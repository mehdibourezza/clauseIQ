"""
Quote verification: proves that an excerpt returned by the LLM really exists in the contract.

Why this exists:
  An LLM can paraphrase or invent text while claiming to quote it. A lawyer must be able
  to trust that every highlighted sentence is in the contract, so we only accept quotes
  found verbatim in the source text and we reject everything else.

Matching is strict, with one exception: text extracted from PDFs has irregular whitespace
(line breaks, double spaces, non-breaking spaces) and typographic punctuation (curly quotes,
long dashes) that the model may normalise. We normalise both sides for those characters
only, then require an exact match. Fuzzy matching is deliberately NOT used: it would
accept a reworded quote, which is exactly what we want to catch.

The verifier keeps a map from each normalised character back to its index in the original
text, so a match is returned as (start, end) offsets in the ORIGINAL text, ready for
highlighting.
"""

from dataclasses import dataclass

# One-to-one character substitutions: each keeps the string length, so offsets stay aligned.
_PUNCTUATION = str.maketrans({
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en dash, em dash
})

# Shorter quotes ("a", "of") would match almost anywhere and prove nothing.
MIN_QUOTE_CHARS = 3


@dataclass(frozen=True)
class Span:
    """Half-open character range [start, end) in the original contract text."""
    start: int
    end: int


def _normalise(text: str) -> tuple[str, list[int]]:
    """
    Collapse every whitespace run into a single space and unify punctuation.

    Returns the normalised string and, for each of its characters, the index of the
    original character it came from.
    """
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text.translate(_PUNCTUATION)):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        origin.append(i)
    return "".join(chars), origin


class QuoteVerifier:
    """Locates quotes in one contract. Build once per contract, call locate() per quote."""

    def __init__(self, contract_text: str):
        self._normalised, self._origin = _normalise(contract_text)

    def locate(self, quote: str) -> Span | None:
        """Return the span of the first exact occurrence of `quote`, or None if absent."""
        spans = self.locate_all(quote)
        return spans[0] if spans else None

    def locate_all(self, quote: str) -> list[Span]:
        """Every non-overlapping exact occurrence of `quote`, in reading order ([] if absent)."""
        needle, _ = _normalise(quote)
        needle = needle.strip()
        if len(needle) < MIN_QUOTE_CHARS:
            return []

        spans = []
        position = self._normalised.find(needle)
        while position != -1:
            last = position + len(needle) - 1
            spans.append(Span(start=self._origin[position], end=self._origin[last] + 1))
            position = self._normalised.find(needle, position + len(needle))
        return spans
