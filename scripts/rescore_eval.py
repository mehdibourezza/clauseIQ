"""
Re-score a saved evaluation report under both matching rules, without any new LLM call.

The report keeps the raw per-contract records (quote positions and lawyer annotations), so the
scores can be recomputed for free. The pre-registered rule ("inside") stays the official result;
the symmetric rule is an exploratory, post-hoc analysis (see scripts/eval_metrics.py).

    python -m scripts.rescore_eval reports/eval_claude_20261011_1645.json
"""

import argparse
import json
from pathlib import Path

from scripts.eval_metrics import PRIMARY_THRESHOLD, RULE_INSIDE, RULE_SYMMETRIC, SENSITIVITY_THRESHOLDS
from scripts.evaluate_llm import score

WATCHED_TYPES = ("Agreement Date", "Effective Date", "Parties", "Document Name")


def describe(proportion: dict) -> str:
    low, high = proportion["ci95"]
    return f"{proportion['value']:.3f} [{low:.3f}, {high:.3f}] (n={proportion['n']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"{report['command']}  ({report['generated_at']}, model {report['model']})")
    for number, run in enumerate(report["runs"], 1):
        done = [r for r in run["records"] if r["status"] == "done"]
        print(f"\nRun {number}: {len(done)} contracts")
        for rule, label in ((RULE_INSIDE, "pre-registered"), (RULE_SYMMETRIC, "exploratory, post-hoc")):
            primary = score(done, PRIMARY_THRESHOLD, rule)
            clauses = primary["clauses"]
            print(f"  {rule:9s} ({label})")
            print(f"    precision {describe(clauses['precision'])}  recall {describe(clauses['recall'])}  F1 {clauses['f1']}")
            sensitivity = {t: score(done, t, rule)["clauses"] for t in SENSITIVITY_THRESHOLDS}
            print("    by threshold (P / R): " + "  ".join(
                f"{t}: {c['precision']['value']:.3f} / {c['recall']['value']:.3f}" for t, c in sensitivity.items()))
            per_type = primary["per_clause_type"]
            print("    recall: " + "  ".join(
                f"{name} {per_type[name]['recall']['value']:.2f} (n={per_type[name]['support']})" for name in WATCHED_TYPES))


if __name__ == "__main__":
    main()
