"""ClaudeExtractor tests against a stub client: they check our handling of the API response, not Claude."""

from types import SimpleNamespace

import pytest

from core.clause_types import CLAUSE_NAMES
from core.extraction import ExtractionError
from engines.claude import SYSTEM_PROMPT, CategoryExcerpts, ClaudeExtractor, ExtractionOutput


class _StubStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._message


def _extractor_returning(stop_reason="end_turn", categories=()):
    message = SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=ExtractionOutput(categories=list(categories)),
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )
    calls = []

    def stream(**kwargs):
        calls.append(kwargs)
        return _StubStream(message)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return ClaudeExtractor(client, model="claude-opus-5-5", effort="high"), calls


def test_catalogue_has_41_unique_categories_all_listed_in_prompt():
    assert len(CLAUSE_NAMES) == 41 == len(set(CLAUSE_NAMES))
    assert all(f"- {name}:" in SYSTEM_PROMPT for name in CLAUSE_NAMES)


def test_output_schema_restricts_categories_to_the_catalogue():
    schema = ExtractionOutput.model_json_schema()
    assert schema["$defs"]["CategoryExcerpts"]["properties"]["clause_type"]["enum"] == list(CLAUSE_NAMES)


def test_every_category_is_returned_and_missing_ones_are_empty():
    extractor, calls = _extractor_returning(categories=[
        CategoryExcerpts(clause_type="Governing Law", excerpts=["governed by the laws of Delaware"]),
        CategoryExcerpts(clause_type="Governing Law", excerpts=["Section 12"]),
    ])
    result = extractor.extract("contract text")

    assert set(result.excerpts) == set(CLAUSE_NAMES)
    assert result.excerpts["Governing Law"] == ["governed by the laws of Delaware", "Section 12"]
    assert result.excerpts["Cap On Liability"] == []
    assert (result.input_tokens, result.output_tokens) == (1200, 300)
    assert calls[0]["output_format"] is ExtractionOutput
    assert "<contract>\ncontract text\n</contract>" in calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_answers_raise_extraction_error(stop_reason):
    extractor, _ = _extractor_returning(stop_reason=stop_reason)
    with pytest.raises(ExtractionError):
        extractor.extract("contract text")
