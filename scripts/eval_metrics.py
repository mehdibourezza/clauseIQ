"""
Scoring rules for the evaluation, fixed BEFORE running the LLM (pre-registered on 2026-10-11).

Why fix them first: if the matching rule is chosen after seeing the results, it can be tuned
(consciously or not) to flatter the model. The rules below are written down, tested, and the
evaluation script only applies them.

Vocabulary (per contract and clause type):
  gold spans     passages annotated by CUAD lawyers for that clause type
  excerpt        a VERIFIED quote returned by the engine (found verbatim). An excerpt is judged
                 on its TEXT: when that text occurs several times in the contract (e.g. a title),
                 all its occurrences are considered. Found while validating the measure with the
                 lawyers' own annotations (2026-10-11, before any LLM run): judging only the first
                 occurrence, which the app highlights, failed 7 "Document Name" cases out of 20
                 contracts because lawyers annotated a later copy of the title.

Excerpt-level rule (threshold t, primary t = 0.5):
  correct        for at least one occurrence, at least t of its characters lie inside gold spans
                 of the SAME type
  mislabelled    not correct, but some occurrence has at least t inside gold spans of ANOTHER type
  unsupported    neither: the quoted text was not annotated by the lawyers for any clause
                 (not necessarily wrong: annotations can be incomplete)

Why "share of the excerpt inside the gold span" and not CUAD's own rule (word-level Jaccard
>= 0.5 between prediction and annotation): our prompt asks for the shortest passage (usually one
sentence) while CUAD annotations can span a whole paragraph. Jaccard would penalise a correct
one-sentence excerpt taken from a long annotated clause. For a lawyer, the question is "is the
highlighted text part of the relevant clause?", which is what this rule measures.

Clause-level rule (strict, as in named-entity evaluation):
  found (TP)     clause annotated AND at least one correct excerpt of that type
  missed (FN)    clause annotated AND no correct excerpt of that type
  false alarm    at least one excerpt of that type but no correct one. If the clause is annotated,
  (FP)           this is counted as BOTH a false alarm and a miss: flagging a clause with the
                 wrong passage is not a success.

Uncertainty: Wilson 95% score interval on every proportion (better than the textbook
p ± 1.96·sqrt(p(1-p)/n) for small n or p near 0/1). Caveat: the (contract, clause) pairs of one
contract are not independent, so the intervals are slightly optimistic.
"""

import math

Span = tuple[int, int]

PRIMARY_THRESHOLD = 0.5
SENSITIVITY_THRESHOLDS = (0.25, 0.5, 0.75)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion; None when there is no data."""
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return round(max(0.0, centre - half_width), 3), round(min(1.0, centre + half_width), 3)


def rate(successes: int, total: int) -> dict:
    """A proportion with its support and 95% interval, ready for the JSON report."""
    return {
        "value": round(successes / total, 3) if total else None,
        "k": successes,
        "n": total,
        "ci95": wilson_interval(successes, total),
    }


def _merge(spans: list[Span]) -> list[Span]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def inside_fraction(excerpt: Span, gold: list[Span]) -> float:
    """Share of the excerpt's characters covered by the gold spans (overlaps counted once)."""
    length = excerpt[1] - excerpt[0]
    if length <= 0:
        return 0.0
    covered = sum(max(0, min(excerpt[1], e) - max(excerpt[0], s)) for s, e in _merge(gold))
    return covered / length


def classify_excerpt(
    occurrences: list[Span], clause_type: str, gold_by_type: dict[str, list[Span]], threshold: float
) -> str:
    """'correct', 'mislabelled' or 'unsupported' for one excerpt given all its occurrences."""
    own_gold = gold_by_type.get(clause_type, [])
    if any(inside_fraction(span, own_gold) >= threshold for span in occurrences):
        return "correct"
    other_gold = [span for t, spans in gold_by_type.items() if t != clause_type for span in spans]
    if any(inside_fraction(span, other_gold) >= threshold for span in occurrences):
        return "mislabelled"
    return "unsupported"


def score_contract(
    gold_by_type: dict[str, list[Span]],
    excerpts: list[tuple[str, list[Span]]],
    clause_types: list[str],
    threshold: float,
) -> dict[str, dict[str, int]]:
    """
    Per clause type outcome counts for one contract: tp, fn, fp (0/1 each) and excerpt
    categories (correct / mislabelled / unsupported counts). Each excerpt is given as
    (clause_type, all occurrences of its text).
    """
    per_type: dict[str, dict[str, int]] = {}
    for clause_type in clause_types:
        own = [occurrences for t, occurrences in excerpts if t == clause_type]
        categories = [classify_excerpt(occ, clause_type, gold_by_type, threshold) for occ in own]
        annotated = bool(gold_by_type.get(clause_type))
        has_correct = "correct" in categories
        per_type[clause_type] = {
            "tp": int(annotated and has_correct),
            "fn": int(annotated and not has_correct),
            "fp": int(bool(own) and not has_correct),
            "correct": categories.count("correct"),
            "mislabelled": categories.count("mislabelled"),
            "unsupported": categories.count("unsupported"),
        }
    return per_type
