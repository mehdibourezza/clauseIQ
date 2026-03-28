"""
PyTorch Dataset for tokenized CUAD features stored as JSONL.

Why a custom dataset instead of HuggingFace datasets.load_dataset?
  - Our data is already tokenized and stored as JSONL (one feature per line)
  - We need to stream from disk without loading everything into RAM
  - We control exactly what goes into each batch

Memory strategy:
  - We DO load everything into RAM here (train_features.jsonl is ~2-3 GB)
  - If that becomes a problem, switch to IterableDataset + line-by-line streaming
  - For now, list-of-dicts is simpler and faster (random access for shuffling)

The dataset handles the 1.4% class imbalance problem in two ways:
  1. It reports class weights for use in the loss function
  2. It can optionally return only positive examples (for debugging)
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class CUADFeatureDataset(Dataset):
    """
    Loads pre-tokenized CUAD features from a JSONL file.

    Each line in the JSONL contains:
      - input_ids: list of 512 ints (token IDs)
      - attention_mask: list of 512 ints (1=real token, 0=padding)
      - start_position: int (ground truth start token index)
      - end_position: int (ground truth end token index)
      - is_impossible: bool (True = no clause in this window)
      - clause_label: str (e.g. "Cap On Liability")
      - example_id: str (for grouping windows from the same contract)

    On __getitem__, returns a dict of tensors ready for the model's forward().
    """

    def __init__(
        self,
        features_path: Path,
        only_positive: bool = False,
    ):
        """
        Args:
            features_path: path to {split}_features.jsonl
            only_positive: if True, keep only windows where the clause is present.
                           Useful for debugging: can the model find spans at all?
        """
        self.features: list[dict] = []

        with open(features_path, encoding="utf-8") as f:
            for line in f:
                feat = json.loads(line)
                if only_positive and feat["is_impossible"]:
                    continue
                self.features.append(feat)

        n_total = len(self.features)
        n_positive = sum(1 for f in self.features if not f["is_impossible"])
        n_negative = n_total - n_positive

        print(f"Loaded {n_total:,} features from {features_path.name}")
        print(f"  Positive (clause present): {n_positive:,} ({100*n_positive/n_total:.1f}%)")
        print(f"  Negative (no clause):      {n_negative:,} ({100*n_negative/n_total:.1f}%)")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        feat = self.features[idx]

        return {
            "input_ids":      torch.tensor(feat["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(feat["start_position"], dtype=torch.long),
            "end_positions":   torch.tensor(feat["end_position"],   dtype=torch.long),
            "is_impossible":   torch.tensor(feat["is_impossible"],  dtype=torch.bool),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class weights to counter the 1.4% positive rate.

        Used in WeightedRandomSampler to over-sample positive windows.

        Weight formula:
          w_positive  = n_total / (2 * n_positive)
          w_negative  = n_total / (2 * n_negative)

        This is the standard sklearn-style balanced class weight formula.
        It ensures the model sees roughly equal numbers of positive and
        negative examples per epoch, regardless of the actual 1:70 ratio.

        Returns a tensor of shape (n_samples,) with one weight per example.
        This is what WeightedRandomSampler expects.
        """
        n_total = len(self.features)
        n_positive = sum(1 for f in self.features if not f["is_impossible"])
        n_negative = n_total - n_positive

        w_pos = n_total / (2.0 * n_positive) if n_positive > 0 else 1.0
        w_neg = n_total / (2.0 * n_negative) if n_negative > 0 else 1.0

        weights = [
            w_neg if feat["is_impossible"] else w_pos
            for feat in self.features
        ]

        return torch.tensor(weights, dtype=torch.float)
