"""
Long contracts: split into overlapping parts for extraction, then merge the results.

ChunkingExtractor is a decorator: it implements the ClauseExtractor port and wraps any other
extractor, so the same splitting works for Claude today and for a self-hosted engine later.
Short contracts go straight to the wrapped extractor in one call.

Why quote positions stay correct: parts are only used for EXTRACTION. Verification runs on
the full contract text (core.analysis), so each quote is located in the full text directly.

The overlap guarantee. Write each part as [start, end). The splitter keeps one invariant:
the next part starts at or before end - overlap. Take a clause of length L <= overlap and the
LAST part whose start is at or before the clause start s. If the clause crossed that part's
end, then s > end - L >= end - overlap >= next start, so a later part would also start at or
before s: a contradiction. Hence that part contains the whole clause. The default overlap
(4,000 characters) exceeds the longest clause annotated in CUAD (3,884 characters).
"""

from core.extraction import ClauseExtractor, ExtractionError, ExtractionResult

# How far back from an ideal cut point we look for a line break or a space, so that parts
# do not end or start in the middle of a word.
_SNAP_WINDOW = 1_000


def _cut_at_or_before(text: str, position: int, floor: int) -> int:
    """A cut point <= position (and >= floor), just after a line break or else a space."""
    lowest = max(floor, position - _SNAP_WINDOW)
    for separator in ("\n", " "):
        index = text.rfind(separator, lowest, position)
        if index != -1:
            return index + 1
    return position


def split_into_chunks(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """
    (start, end) offsets of parts covering the whole text: each part is at most max_chars
    long, and each part starts at or before the previous part's end minus `overlap`.
    """
    if max_chars <= 2 * overlap:
        raise ValueError("max_chars must be more than twice the overlap")
    if len(text) <= max_chars:
        return [(0, len(text))]

    parts: list[tuple[int, int]] = []
    start = 0
    while start + max_chars < len(text):
        # Cut where a part ends no earlier than half its maximum size: guarantees progress.
        end = _cut_at_or_before(text, start + max_chars, floor=start + max_chars // 2)
        parts.append((start, end))
        # Snapping only ever moves the next start LEFT, which keeps the overlap invariant.
        start = _cut_at_or_before(text, end - overlap, floor=start + 1)
    parts.append((start, len(text)))
    return parts


class ChunkingExtractor:
    """Runs the wrapped extractor on each part of a long contract and merges the excerpts."""

    def __init__(self, inner: ClauseExtractor, max_chars: int, overlap: int):
        if max_chars <= 2 * overlap:
            raise ValueError("max_chars must be more than twice the overlap")
        self._inner = inner
        self._max_chars = max_chars
        self._overlap = overlap

    def extract(self, contract_text: str) -> ExtractionResult:
        parts = split_into_chunks(contract_text, self._max_chars, self._overlap)
        if len(parts) == 1:
            return self._inner.extract(contract_text)

        results: list[ExtractionResult] = []
        for number, (start, end) in enumerate(parts, 1):
            try:
                results.append(self._inner.extract(contract_text[start:end]))
            except ExtractionError as exc:
                # No partial result: it would show "clause absent" for text nobody read.
                raise ExtractionError(f"Part {number}/{len(parts)} of the contract failed: {exc}") from exc

        merged: dict[str, list[str]] = {}
        for result in results:
            for clause_type, quotes in result.excerpts.items():
                bucket = merged.setdefault(clause_type, [])
                for quote in quotes:
                    if quote not in bucket:   # the same quote can come from two overlapping parts
                        bucket.append(quote)

        return ExtractionResult(
            excerpts=merged,
            model=results[0].model,
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
        )
