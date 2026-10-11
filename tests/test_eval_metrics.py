"""The pre-registered scoring rules (scripts/eval_metrics.py) on hand-made cases."""

import pytest

from scripts.eval_metrics import classify_excerpt, inside_fraction, score_contract, wilson_interval

GOLD = {"Governing Law": [(100, 200)], "Cap On Liability": [(300, 400)], "Parties": []}
TYPES = list(GOLD)


def test_wilson_interval_on_three_out_of_four():
    assert wilson_interval(3, 4) == pytest.approx((0.301, 0.954), abs=0.001)
    assert wilson_interval(0, 0) is None


def test_inside_fraction_counts_overlapping_gold_spans_once():
    assert inside_fraction((100, 200), [(100, 150), (120, 200)]) == 1.0
    assert inside_fraction((180, 220), [(100, 200)]) == 0.5
    assert inside_fraction((0, 50), [(100, 200)]) == 0.0


@pytest.mark.parametrize(
    "occurrences, clause_type, expected",
    [
        ([(120, 180)], "Governing Law", "correct"),             # fully inside its own annotation
        ([(190, 230)], "Governing Law", "unsupported"),         # only 25% inside: below the 50% rule
        ([(310, 390)], "Governing Law", "mislabelled"),         # inside the Cap On Liability annotation
        ([(500, 600)], "Governing Law", "unsupported"),         # annotated for nothing
        ([(10, 40), (120, 150)], "Governing Law", "correct"),   # repeated text: one occurrence is enough
    ],
)
def test_excerpt_classification(occurrences, clause_type, expected):
    assert classify_excerpt(occurrences, clause_type, GOLD, threshold=0.5) == expected


def test_clause_flagged_with_the_wrong_passage_is_both_a_false_alarm_and_a_miss():
    outcome = score_contract(GOLD, [("Governing Law", [(310, 390)])], TYPES, threshold=0.5)
    assert outcome["Governing Law"] == {
        "tp": 0, "fn": 1, "fp": 1, "correct": 0, "mislabelled": 1, "unsupported": 0,
    }


def test_found_clause_unannotated_clause_and_correct_absence():
    excerpts = [("Governing Law", [(120, 180)]), ("Parties", [(0, 40)])]
    outcome = score_contract(GOLD, excerpts, TYPES, threshold=0.5)
    assert outcome["Governing Law"]["tp"] == 1 and outcome["Governing Law"]["fp"] == 0
    assert outcome["Parties"]["fp"] == 1 and outcome["Parties"]["fn"] == 0       # flagged, not annotated
    assert outcome["Cap On Liability"]["fn"] == 1                                # annotated, never flagged
