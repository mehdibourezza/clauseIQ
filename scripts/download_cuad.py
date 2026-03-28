"""
Download and explore the CUAD dataset directly from the Atticus Project GitHub.

CUAD = Contract Understanding Atticus Dataset
- 510 real commercial contracts
- 41 clause types labeled by legal experts
- 13,000+ question-answer pairs (SQuAD-style format)

Run: python -m scripts.download_cuad
"""

import json
import urllib.request
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("ml/data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Direct links to the official CUAD JSON files
DATA_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
ZIP_PATH = OUTPUT_DIR / "data.zip"
EXTRACTED_DIR = OUTPUT_DIR / "cuad_extracted"


def download():
    # Download data.zip if not already cached
    if ZIP_PATH.exists():
        print("  data.zip already downloaded, using cache...")
    else:
        print(f"  Downloading data.zip (~18MB) from GitHub...")
        urllib.request.urlretrieve(DATA_ZIP_URL, ZIP_PATH)
        print(f"  Saved to {ZIP_PATH}")

    # Extract
    if not EXTRACTED_DIR.exists():
        import zipfile
        print("  Extracting data.zip...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACTED_DIR)
        print(f"  Extracted to {EXTRACTED_DIR}")

    # Find the JSON files inside
    json_files = list(EXTRACTED_DIR.rglob("*.json"))
    print(f"  Found JSON files: {[f.name for f in json_files]}")

    # Load train and test splits
    data = {}
    for jf in json_files:
        name = jf.stem  # e.g. "train_separate_questions" or "test"
        split = "train" if "train" in name else "test"
        with open(jf, encoding="utf-8") as f:
            data[split] = json.load(f)
        print(f"  Loaded {split} ← {jf.name}")

    return data


def flatten(raw_data: dict) -> list[dict]:
    """Convert SQuAD-style nested JSON into flat list of QA examples."""
    rows = []
    for article in raw_data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                rows.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "id": qa["id"],
                    "answers": qa["answers"],
                    "is_impossible": qa.get("is_impossible", len(qa["answers"]) == 0),
                })
    return rows


def explore(data: dict):
    print("\n" + "=" * 60)
    print("CUAD DATASET EXPLORATION")
    print("=" * 60)

    for split_name, raw in data.items():
        rows = flatten(raw)
        print(f"\n--- Split: {split_name} ---")
        print(f"  Contracts: {len(raw['data'])}")
        print(f"  QA examples: {len(rows)}")

    # Work with train split
    train_rows = flatten(data["train"])
    example = train_rows[0]

    print("\n--- Example QA row ---")
    print(f"  Contract: {example['title']}")
    print(f"  Context length: {len(example['context'])} chars")
    print(f"  Context preview: {example['context'][:200]}...")
    print(f"  Question (clause type): {example['question']}")
    print(f"  Answer: {example['answers']}")
    print(f"  Is negative (clause absent): {example['is_impossible']}")

    # All 41 clause types
    print("\n--- All clause types in CUAD ---")
    questions = [r["question"] for r in train_rows]
    unique_questions = sorted(set(questions))
    for i, q in enumerate(unique_questions, 1):
        print(f"  {i:2d}. {q}")

    # Positive vs negative balance per clause
    print("\n--- Positive examples per clause type (top 15) ---")
    clause_counts = Counter()
    for row in train_rows:
        if not row["is_impossible"]:
            clause_counts[row["question"]] += 1

    total = len(train_rows)
    positives = sum(clause_counts.values())
    print(f"  Total examples: {total}")
    print(f"  Positive (clause present): {positives} ({100*positives/total:.1f}%)")
    print(f"  Negative (clause absent):  {total - positives} ({100*(total-positives)/total:.1f}%)")
    print()
    for clause, count in clause_counts.most_common(15):
        print(f"  {count:4d}  {clause}")

    # Save summary
    summary = {
        "train_contracts": len(data["train"]["data"]),
        "train_examples": len(train_rows),
        "unique_clause_types": len(unique_questions),
        "clause_types": unique_questions,
        "positive_counts": dict(clause_counts.most_common()),
    }
    summary_path = OUTPUT_DIR / "cuad_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return unique_questions


def save_flat(data: dict):
    print("\nSaving flattened splits...")
    for split_name, raw in data.items():
        rows = flatten(raw)
        out_path = OUTPUT_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    print("Downloading CUAD dataset...")
    data = download()
    clause_types = explore(data)
    save_flat(data)
    print("\nAll done. Next step: run scripts/preprocess_cuad.py")
