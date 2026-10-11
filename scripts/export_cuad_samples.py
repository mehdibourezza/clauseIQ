"""
Write a few CUAD test contracts to samples/*.txt, ready to upload in the app during a demo.

CUAD contracts are public (CC BY 4.0), so they can be shown without confidentiality concerns.

    python -m scripts.export_cuad_samples --n 3
"""

import argparse
import json
import re
from pathlib import Path

CUAD_TEST = Path("ml/data/raw/cuad_extracted/test.json")
SAMPLES_DIR = Path("samples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=40_000, help="skip very long contracts (slower demo)")
    args = parser.parse_args()

    contracts = json.loads(CUAD_TEST.read_text(encoding="utf-8"))["data"]
    candidates = [c for c in contracts if len(c["paragraphs"][0]["context"]) <= args.max_chars]

    SAMPLES_DIR.mkdir(exist_ok=True)
    for contract in candidates[: args.n]:
        name = re.sub(r"[^\w.-]+", "_", contract["title"])[:80] + ".txt"
        # newline="": keep the text byte-identical to CUAD (no \n -> \r\n conversion on Windows)
        (SAMPLES_DIR / name).write_text(contract["paragraphs"][0]["context"], encoding="utf-8", newline="")
        print(f"samples/{name}")


if __name__ == "__main__":
    main()
