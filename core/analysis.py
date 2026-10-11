"""
Contract analysis = extraction (any engine) + verification of every excerpt.

This is the business core of the app. It has no knowledge of HTTP, databases or Claude:
the API, the evaluation script and the tests all call analyze_contract() the same way.
"""

from dataclasses import dataclass

from core.extraction import ClauseExtractor
from core.verification import QuoteVerifier, Span


@dataclass(frozen=True)
class Finding:
    clause_type: str
    quote: str
    span: Span | None   # None = the quote was not found in the contract (never highlighted)

    @property
    def verified(self) -> bool:
        return self.span is not None


@dataclass(frozen=True)
class AnalysisResult:
    findings: list[Finding]
    model: str
    input_tokens: int
    output_tokens: int


def analyze_contract(contract_text: str, extractor: ClauseExtractor) -> AnalysisResult:
    extraction = extractor.extract(contract_text)
    verifier = QuoteVerifier(contract_text)

    findings: list[Finding] = []
    for clause_type, quotes in extraction.excerpts.items():
        for quote in dict.fromkeys(quotes):   # drop exact duplicates, keep order
            findings.append(Finding(clause_type, quote, verifier.locate(quote)))

    return AnalysisResult(
        findings=findings,
        model=extraction.model,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
    )
