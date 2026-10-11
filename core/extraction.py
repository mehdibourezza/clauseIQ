"""
The extraction port: what the rest of the app expects from any clause-extraction engine.

ClauseExtractor is an interface (a "port"). The analysis core, the API and the evaluation
depend on it, never on a concrete engine. Engines are adapters living outside core/
(engines/claude.py today; a self-hosted Legal-BERT engine later), and core.chunking wraps any
of them for contracts too long for one call.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExtractionResult:
    excerpts: dict[str, list[str]]   # every one of the 41 names is a key; [] means absent
    model: str                       # the model that actually answered (traceability)
    input_tokens: int
    output_tokens: int


class ExtractionError(Exception):
    """The engine could not produce a usable answer. The message is user-facing."""


class ClauseExtractor(Protocol):
    def extract(self, contract_text: str) -> ExtractionResult: ...
