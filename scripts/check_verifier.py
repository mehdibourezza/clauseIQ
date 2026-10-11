"""
Checks the quote verifier against every lawyer annotation of the CUAD test split (no LLM, free).

For each annotated excerpt, asks QuoteVerifier to find it in its own contract and counts:
  located      found verbatim (after whitespace/typography normalisation)
  too short    not located because shorter than MIN_QUOTE_CHARS (rejected by design)
  elsewhere    located, but the first occurrence is not the annotated one (repeated text)

    python -m scripts.check_verifier
"""

import json
from pathlib import Path

from core.verification import MIN_QUOTE_CHARS, QuoteVerifier

CUAD_TEST = Path("ml/data/raw/cuad_extracted/test.json")


def main() -> None:
    contracts = json.loads(CUAD_TEST.read_text(encoding="utf-8"))["data"]
    total = located = too_short = elsewhere = 0
    for contract in contracts:
        paragraph = contract["paragraphs"][0]
        verifier = QuoteVerifier(paragraph["context"])
        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                total += 1
                start, end = answer["answer_start"], answer["answer_start"] + len(answer["text"])
                span = verifier.locate(answer["text"])
                if span is None:
                    too_short += len(answer["text"].strip()) < MIN_QUOTE_CHARS
                    continue
                located += 1
                elsewhere += not (span.start < end and start < span.end)

    print(f"contracts: {len(contracts)}  annotated excerpts: {total}")
    print(f"located: {located}  not located: {total - located} (of which shorter than {MIN_QUOTE_CHARS} chars: {too_short})")
    print(f"located elsewhere (repeated text, first occurrence highlighted): {elsewhere} ({elsewhere / total:.1%})")


if __name__ == "__main__":
    main()
