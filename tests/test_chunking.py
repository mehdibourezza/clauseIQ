"""Splitting long contracts into overlapping parts, and merging per-part extractions."""

import pytest

from core.chunking import ChunkingExtractor, split_into_chunks
from core.extraction import ExtractionError, ExtractionResult

# ~85,000 characters of line-broken text, long enough to need several 10,000-character parts.
TEXT = "\n".join(f"Section {i}. " + "The parties agree that " * 9 for i in range(400))
MAX, OVERLAP = 10_000, 1_000


def test_short_text_is_a_single_part():
    assert split_into_chunks("short contract", MAX, OVERLAP) == [(0, 14)]


def test_parts_cover_the_text_within_size_and_overlap_limits():
    parts = split_into_chunks(TEXT, MAX, OVERLAP)
    assert len(parts) > 1
    assert parts[0][0] == 0 and parts[-1][1] == len(TEXT)
    assert all(end - start <= MAX for start, end in parts)
    for (start, end), (next_start, _) in zip(parts, parts[1:]):
        assert start < next_start <= end - OVERLAP


def test_every_clause_up_to_the_overlap_length_fits_whole_in_some_part():
    parts = split_into_chunks(TEXT, MAX, OVERLAP)
    for clause_start in range(0, len(TEXT) - OVERLAP, 37):
        clause_end = clause_start + OVERLAP
        assert any(s <= clause_start and clause_end <= e for s, e in parts), clause_start


def test_sizes_that_cannot_guarantee_progress_are_rejected():
    with pytest.raises(ValueError):
        split_into_chunks(TEXT, 2_000, 1_000)


class RecordingExtractor:
    """Returns one quote shared by every part and one quote unique to each part."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[str] = []
        self._fail_on_call = fail_on_call

    def extract(self, contract_text: str) -> ExtractionResult:
        self.calls.append(contract_text)
        if len(self.calls) == self._fail_on_call:
            raise ExtractionError("LLM API error (529): overloaded")
        n = len(self.calls)
        return ExtractionResult(
            {"Governing Law": ["shared quote", f"quote from part {n}"], "Parties": []},
            model="fake-model", input_tokens=100, output_tokens=10,
        )


def test_short_contract_goes_through_in_one_call():
    inner = RecordingExtractor()
    ChunkingExtractor(inner, MAX, OVERLAP).extract("short contract")
    assert inner.calls == ["short contract"]


def test_long_contract_is_extracted_part_by_part_and_merged():
    inner = RecordingExtractor()
    result = ChunkingExtractor(inner, MAX, OVERLAP).extract(TEXT)
    n = len(inner.calls)
    assert n == len(split_into_chunks(TEXT, MAX, OVERLAP)) > 1
    assert result.excerpts["Governing Law"] == ["shared quote"] + [f"quote from part {i}" for i in range(1, n + 1)]
    assert result.excerpts["Parties"] == []
    assert (result.input_tokens, result.output_tokens) == (100 * n, 10 * n)


def test_one_failed_part_fails_the_whole_analysis():
    with pytest.raises(ExtractionError, match="Part 2/"):
        ChunkingExtractor(RecordingExtractor(fail_on_call=2), MAX, OVERLAP).extract(TEXT)
