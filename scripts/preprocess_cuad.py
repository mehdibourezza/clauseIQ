"""
Preprocess CUAD data into tokenized features for Legal-BERT fine-tuning.

What this does:
  1. Loads flat train/test QA examples from ml/data/raw/
  2. Tokenizes with Legal-BERT tokenizer using a sliding window
     (contracts are too long for 512 tokens — we chunk them with overlap)
  3. Maps character-level answer spans to token-level start/end positions
  4. Saves processed features to ml/data/processed/

The sliding window explained:
  - max_length = 512 tokens (BERT's hard limit)
  - doc_stride = 128 tokens (overlap between windows)
  - A 4000-token contract produces ~8 overlapping windows
  - Each window is a separate training example

Run: python -m scripts.preprocess_cuad
"""

import gc
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

RAW_DIR = Path("ml/data/raw")
PROCESSED_DIR = Path("ml/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512
DOC_STRIDE = 128       # overlap between windows
MAX_QUERY_LENGTH = 64  # questions won't exceed this


def load_examples(split: str) -> list[dict]:
    path = RAW_DIR / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_clause_label(question: str) -> str:
    """Extract short clause label from the verbose CUAD question string.

    e.g. 'Highlight the parts ... related to "Cap On Liability" ...'
         -> 'Cap On Liability'
    """
    import re
    match = re.search(r'"([^"]+)"', question)
    return match.group(1) if match else question[:50]


def process_and_save(
    examples: list[dict],
    tokenizer: AutoTokenizer,
    split: str,
):
    """
    Convert raw QA examples into tokenized features with sliding windows.
    Writes to disk incrementally (JSONL) to avoid OOM on large datasets.
    """
    from collections import Counter

    features_path = PROCESSED_DIR / f"{split}_features.jsonl"
    meta_path = PROCESSED_DIR / f"{split}_meta.jsonl"

    total = 0
    skipped = 0
    n_possible = 0
    clause_counts = Counter()
    total_active_tokens = 0

    with open(features_path, "w", encoding="utf-8") as f_feat, \
         open(meta_path, "w", encoding="utf-8") as f_meta:

        for example in tqdm(examples, desc=f"Processing {split}"):
            question = example["question"]
            context = example["context"]
            answers = example["answers"]
            is_impossible = example["is_impossible"]
            clause_label = get_clause_label(question)

            tokenized = tokenizer(
                question,
                context,
                max_length=MAX_LENGTH,
                truncation="only_second",
                stride=DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            for i, input_ids in enumerate(tokenized["input_ids"]):
                offset_mapping = tokenized["offset_mapping"][i]
                sequence_ids = tokenized.sequence_ids(i)

                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

                offset_mapping = [
                    o if sequence_ids[k] == 1 else (0, 0)
                    for k, o in enumerate(offset_mapping)
                ]

                start_position = 0
                end_position = 0

                if not is_impossible and answers:
                    answer_text = answers[0]["text"]
                    answer_start_char = answers[0]["answer_start"]
                    answer_end_char = answer_start_char + len(answer_text)

                    window_start_char = offset_mapping[context_start][0]
                    window_end_char = offset_mapping[context_end][1]

                    if answer_start_char >= window_start_char and answer_end_char <= window_end_char:
                        token_start = context_start
                        while token_start <= context_end and offset_mapping[token_start][0] <= answer_start_char:
                            token_start += 1
                        start_position = token_start - 1

                        token_end = context_end
                        while token_end >= context_start and offset_mapping[token_end][1] >= answer_end_char:
                            token_end -= 1
                        end_position = token_end + 1

                        if start_position > end_position:
                            skipped += 1
                            continue

                feature_impossible = is_impossible or (start_position == 0 and end_position == 0)

                # Write training feature (no context/offsets — keeps file small)
                feature = {
                    "input_ids": input_ids,
                    "attention_mask": tokenized["attention_mask"][i],
                    "start_position": start_position,
                    "end_position": end_position,
                    "is_impossible": feature_impossible,
                    "clause_label": clause_label,
                    "example_id": example["id"],
                }
                f_feat.write(json.dumps(feature) + "\n")

                # Write metadata (for inference post-processing)
                meta = {
                    "example_id": example["id"],
                    "clause_label": clause_label,
                    "offset_mapping": offset_mapping,
                    "context": context,
                    "question": question,
                }
                f_meta.write(json.dumps(meta) + "\n")

                # Track stats
                total += 1
                active = sum(1 for x in tokenized["attention_mask"][i] if x == 1)
                total_active_tokens += active
                if not feature_impossible:
                    n_possible += 1
                    clause_counts[clause_label] += 1

            # Explicitly free tokenizer output and force GC every 200 examples
            del tokenized
            if total > 0 and total % 200 == 0:
                gc.collect()

    print(f"\n--- {split} feature stats ---")
    print(f"  Total windows:     {total:,}")
    print(f"  With answer:       {n_possible:,} ({100*n_possible/total:.1f}%)")
    print(f"  Without answer:    {total - n_possible:,} ({100*(total-n_possible)/total:.1f}%)")
    print(f"  Skipped (invalid): {skipped}")
    print(f"  Avg tokens/window: {total_active_tokens/total:.0f}")
    print(f"\n  Top 5 clause types:")
    for label, count in clause_counts.most_common(5):
        print(f"    {count:4d}  {label}")
    print(f"\n  Saved to {features_path}")
    print(f"  Metadata to {meta_path}")


if __name__ == "__main__":
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    for split in ["train", "test"]:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        examples = load_examples(split)
        print(f"  Loaded {len(examples):,} examples")

        process_and_save(examples, tokenizer, split)

    print("\nPreprocessing complete.")
    print("Next step: python -m ml.training.train")
