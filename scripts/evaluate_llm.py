"""
Evaluate the extraction engine against CUAD's lawyer annotations.

Every contract goes through the same code path as the app (api.jobs.run_analysis writing to a
database), so the numbers describe what a user actually sees. Scoring rules are defined and
justified in scripts/eval_metrics.py and were fixed before the first LLM run.

Reported (with 95% Wilson intervals and the support n of each figure):
  clauses     precision / recall / F1 of "clause found with a correct excerpt" (strict rule)
  excerpts    rejected rate (quotes not found verbatim: never shown), then, among shown excerpts,
              correct / mislabelled / unsupported rates
  robustness  precision and recall at matching thresholds 0.25, 0.5 (primary) and 0.75
  cost        tokens, estimated dollars and seconds per contract

Usage:
    python -m scripts.evaluate_llm --extractor gold --n 20     # free sanity check: replays the
                                                                # lawyers' annotations (the ceiling)
    python -m scripts.evaluate_llm --n 20                       # paid: one LLM call per contract
    python -m scripts.evaluate_llm --n 20 --runs 2              # paid x2: run-to-run variance
    python -m scripts.evaluate_llm --n 5 --save-to-db           # keep results in the app DB (offline demo)
"""

import argparse
import json
import random
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import selectinload

from api.dependencies.providers import get_extractor
from api.jobs import run_analysis
from configs.config import settings
from core.clause_types import CLAUSE_NAMES
from core.extraction import ExtractionResult
from core.verification import QuoteVerifier
from data.db.models import Contract
from data.db.session import make_session_factory
from scripts.eval_metrics import (
    PRIMARY_THRESHOLD,
    RULE_INSIDE,
    SENSITIVITY_THRESHOLDS,
    rate,
    score_contract,
)

CUAD_TEST = Path("ml/data/raw/cuad_extracted/test.json")
REPORTS_DIR = Path("reports")
# USD per million tokens (input, output). Models not listed: cost is not estimated.
PRICES = {"claude-opus-5-5": (4.0, 20.0)}


def clause_name(question: str) -> str:
    return question.split('"')[1]


def load_gold(cuad_contract: dict) -> dict[str, list[tuple[int, int]]]:
    """Lawyer-annotated character spans per clause type for one CUAD contract."""
    return {
        clause_name(qa["question"]): [
            (a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]
        ]
        for qa in cuad_contract["paragraphs"][0]["qas"]
    }


class GoldExtractor:
    """Replays the lawyers' annotations as if they were model output: the evaluation ceiling."""

    def __init__(self, contracts: list[dict]):
        self._by_text = {c["paragraphs"][0]["context"]: c for c in contracts}

    def extract(self, contract_text: str) -> ExtractionResult:
        qas = self._by_text[contract_text]["paragraphs"][0]["qas"]
        excerpts = {clause_name(qa["question"]): [a["text"] for a in qa["answers"]] for qa in qas}
        return ExtractionResult(excerpts, model="gold-annotations", input_tokens=0, output_tokens=0)


def analyse(cuad: dict, session_factory, extractor) -> dict:
    """Run one contract through the app's job and return the raw record used for scoring."""
    text = cuad["paragraphs"][0]["context"]
    with session_factory() as db:
        contract = Contract(filename=f"{cuad['title']}.txt", text=text)
        db.add(contract)
        db.commit()
        contract_id = contract.id

    started = time.perf_counter()
    run_analysis(contract_id, session_factory, extractor)
    seconds = round(time.perf_counter() - started, 1)

    verifier = QuoteVerifier(text)
    with session_factory() as db:
        contract = db.get(Contract, contract_id, options=[selectinload(Contract.findings)])
        record = {"title": cuad["title"], "status": contract.status, "error": contract.error, "seconds": seconds}
        if contract.status != "done":
            return record
        price_in, price_out = PRICES.get(contract.model, (None, None))
        record.update({
            "model": contract.model,
            "input_tokens": contract.input_tokens,
            "output_tokens": contract.output_tokens,
            "cost_usd": round((contract.input_tokens * price_in + contract.output_tokens * price_out) / 1e6, 4)
            if price_in else None,
            "returned": len(contract.findings),
            "rejected": sum(not f.verified for f in contract.findings),
            # Every occurrence of each verified quote: the excerpt is judged on its text (eval_metrics).
            "excerpts": [
                (f.clause_type, [(s.start, s.end) for s in verifier.locate_all(f.quote)])
                for f in contract.findings if f.verified
            ],
            "gold": load_gold(cuad),
        })
        return record


def score(records: list[dict], threshold: float, rule: str = RULE_INSIDE) -> dict:
    """Aggregate clause and excerpt outcomes over all analysed contracts."""
    totals = Counter()
    per_type = {name: Counter() for name in CLAUSE_NAMES}
    for record in records:
        excerpts = [(t, [tuple(span) for span in occurrences]) for t, occurrences in record["excerpts"]]
        gold = {t: [tuple(span) for span in spans] for t, spans in record["gold"].items()}
        for clause_type, counts in score_contract(gold, excerpts, list(CLAUSE_NAMES), threshold, rule).items():
            totals.update(counts)
            per_type[clause_type].update(counts)

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision, recall = rate(tp, tp + fp), rate(tp, tp + fn)
    p, r = precision["value"], recall["value"]
    shown = totals["correct"] + totals["mislabelled"] + totals["unsupported"]
    return {
        "clauses": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": round(2 * p * r / (p + r), 3) if p and r else None,
        },
        "shown_excerpts": {
            "correct": rate(totals["correct"], shown),
            "mislabelled": rate(totals["mislabelled"], shown),
            "unsupported": rate(totals["unsupported"], shown),
        },
        "per_clause_type": {
            name: {
                "support": c["tp"] + c["fn"],
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "recall": rate(c["tp"], c["tp"] + c["fn"]),
            }
            for name, c in per_type.items()
        },
    }


def summarise_run(records: list[dict]) -> dict:
    done = [r for r in records if r["status"] == "done"]
    returned = sum(r["returned"] for r in done)
    costs = [r["cost_usd"] for r in done if r["cost_usd"] is not None]
    return {
        "contracts_analysed": len(done),
        "contracts_failed": len(records) - len(done),
        "rejected_excerpts": rate(sum(r["rejected"] for r in done), returned),
        "primary": score(done, PRIMARY_THRESHOLD),
        "sensitivity": {
            str(t): {k: score(done, t)["clauses"][k] for k in ("precision", "recall")}
            for t in SENSITIVITY_THRESHOLDS
        },
        "mean_seconds_per_contract": round(sum(r["seconds"] for r in done) / len(done), 1) if done else None,
        "mean_cost_usd_per_contract": round(sum(costs) / len(costs), 4) if costs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="number of CUAD test contracts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole evaluation (LLM variance)")
    parser.add_argument("--extractor", choices=["claude", "gold"], default="claude")
    parser.add_argument("--save-to-db", action="store_true", help="store results in the app database")
    args = parser.parse_args()

    contracts = json.loads(CUAD_TEST.read_text(encoding="utf-8"))["data"]
    sample = random.Random(args.seed).sample(contracts, args.n)
    extractor = GoldExtractor(contracts) if args.extractor == "gold" else get_extractor()

    runs = []
    for run in range(1, args.runs + 1):
        db_url = settings.review_db_url if args.save_to_db else f"sqlite:///{tempfile.mkdtemp()}/eval.db"
        session_factory = make_session_factory(db_url)
        records = []
        for i, cuad in enumerate(sample, 1):
            record = analyse(cuad, session_factory, extractor)
            records.append(record)
            status = f"{record['seconds']:5.0f}s  ${record.get('cost_usd') or 0:.3f}" if record["status"] == "done" \
                else f"FAILED: {record['error']}"
            print(f"[run {run} · {i}/{args.n}] {status}  {cuad['title'][:60]}")
        runs.append({"summary": summarise_run(records), "records": records})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f"python -m scripts.evaluate_llm --extractor {args.extractor} --n {args.n} "
                   f"--seed {args.seed} --runs {args.runs}",
        "extractor": args.extractor,
        "model": settings.llm_model if args.extractor == "claude" else "gold-annotations",
        "effort": settings.llm_effort if args.extractor == "claude" else None,
        "matching_rule": f"excerpt correct if >= {PRIMARY_THRESHOLD:.0%} of its characters lie in a gold span "
                         "of the same clause type (see scripts/eval_metrics.py)",
        "runs": runs,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"eval_{args.extractor}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for i, run in enumerate(runs, 1):
        s = run["summary"]
        c = s["primary"]["clauses"]
        print(f"\nRun {i}: {s['contracts_analysed']} contracts analysed, {s['contracts_failed']} failed")
        print(f"  clauses   precision {c['precision']['value']} {c['precision']['ci95']}  "
              f"recall {c['recall']['value']} {c['recall']['ci95']}  F1 {c['f1']}")
        print(f"  excerpts  rejected {s['rejected_excerpts']['value']} (n={s['rejected_excerpts']['n']})  "
              f"correct {s['primary']['shown_excerpts']['correct']['value']}  "
              f"mislabelled {s['primary']['shown_excerpts']['mislabelled']['value']}")
        print(f"  per contract: {s['mean_seconds_per_contract']} s, ${s['mean_cost_usd_per_contract']}")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
