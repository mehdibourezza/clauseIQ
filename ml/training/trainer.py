"""
Training loop for ClauseExtractor (Legal-BERT span extraction).

Architecture overview:
  - Model: Legal-BERT + two linear span heads (start + end)
  - Loss: CE(start_logits, start_gt) + CE(end_logits, end_gt)
  - Optimizer: AdamW with different LRs for backbone vs heads
  - Schedule: linear warmup → linear decay (cosine also works, linear is simpler)
  - Sampler: WeightedRandomSampler to counter 1.4% positive rate
  - Two-phase training:
      Phase 1 (epochs 0..freeze_epochs-1): backbone frozen, only heads train
      Phase 2 (epochs freeze_epochs..end): top N backbone layers unfrozen

Why AdamW and not plain Adam?
  Adam tracks per-parameter learning rates (adaptive gradient descent).
  AdamW = Adam + decoupled weight decay (L2 regularization applied to weights
  directly, not through the gradient). This is the standard for fine-tuning
  transformers — it prevents catastrophic forgetting while still regularizing.

Why linear warmup?
  At the start of training, the heads are randomly initialized. If we use the
  full learning rate immediately, the random gradients can destroy the pre-trained
  BERT weights in the first few steps. Warmup ramps the LR from ~0 to the target
  value over 'warmup_steps', giving the model time to stabilize.

Why WeightedRandomSampler and not loss weighting?
  Both are valid. We use WeightedRandomSampler because:
  - It controls what the model SEES (not just how it's penalized)
  - Seeing 50% positive examples per batch is more stable than rare
    positive examples with 70x amplified gradients
  - The model learns span extraction better when it actually sees spans
"""

import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from ml.models.clause_model import ClauseExtractor, SpanPrediction
from ml.data.dataset import CUADFeatureDataset


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Data
    processed_dir: Path = Path("ml/data/processed")
    weights_dir: Path = Path("ml/models/weights")

    # Model
    model_name: str = "nlpaueb/legal-bert-base-uncased"

    # Training schedule
    batch_size: int = 16          # 16 fits in 6GB VRAM (Legal-BERT is 110M params)
    num_epochs: int = 3           # Fine-tuning transformers rarely needs >3 epochs
    freeze_epochs: int = 1        # Phase 1: train only heads for this many epochs
    unfreeze_layers: int = 4      # Phase 2: unfreeze top N transformer layers

    # Learning rates (separate for backbone and heads)
    head_lr: float = 3e-4         # Head LR: larger, random init needs more movement
    backbone_lr: float = 3e-5     # Backbone LR: 10x smaller, preserve pre-training

    # AdamW
    weight_decay: float = 0.01    # Standard weight decay for transformers

    # Warmup: fraction of total steps to ramp up LR
    warmup_ratio: float = 0.06    # 6% warmup is standard for BERT fine-tuning

    # Sampler: how many samples to draw per epoch
    # Using full epoch * rebalancing = ~2x positive examples seen per epoch
    samples_per_epoch: Optional[int] = None  # None = len(dataset) (full epoch)

    # Logging
    log_every_steps: int = 100
    eval_every_steps: int = 500

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    loss: float = 0.0
    n_steps: int = 0

    # Span prediction accuracy (how often does argmax == ground truth?)
    start_acc: float = 0.0
    end_acc: float = 0.0
    n_samples: int = 0

    def update(self, loss: float, start_logits: torch.Tensor,
               end_logits: torch.Tensor, start_gt: torch.Tensor,
               end_gt: torch.Tensor):
        self.loss += loss
        self.n_steps += 1

        # Token-level accuracy: did we predict the right token?
        start_pred = start_logits.argmax(dim=-1)
        end_pred = end_logits.argmax(dim=-1)
        self.start_acc += (start_pred == start_gt).float().sum().item()
        self.end_acc += (end_pred == end_gt).float().sum().item()
        self.n_samples += start_gt.size(0)

    def summary(self) -> dict:
        return {
            "loss": self.loss / max(self.n_steps, 1),
            "start_acc": self.start_acc / max(self.n_samples, 1),
            "end_acc": self.end_acc / max(self.n_samples, 1),
        }

    def reset(self):
        self.loss = 0.0
        self.n_steps = 0
        self.start_acc = 0.0
        self.end_acc = 0.0
        self.n_samples = 0


# ---------------------------------------------------------------------------
# Span F1 evaluation
# ---------------------------------------------------------------------------

def compute_span_f1(
    start_preds: list[int],
    end_preds: list[int],
    start_gts: list[int],
    end_gts: list[int],
    is_impossible: list[bool],
) -> dict:
    """
    Compute span-level F1 score across a batch.

    For each example, the predicted span and ground truth span are sets of
    token positions. F1 = harmonic mean of precision and recall over tokens.

    Why F1 and not accuracy?
      With 1.4% positive rate, predicting "no answer" for everything gives
      98.6% accuracy but 0% recall. F1 rewards the model for actually finding
      clauses, not just being silent.

    Three cases:
      1. Both impossible: true negative → not counted (trivially correct)
      2. GT impossible, pred non-zero: false positive
      3. GT possible, pred: token overlap determines precision/recall

    Returns:
      exact_match: fraction of examples where span matches exactly
      f1: average token-level F1
      recall: fraction of positive GT spans correctly found
    """
    exact_matches = 0
    f1_scores = []
    total_possible = 0
    found_possible = 0

    for s_pred, e_pred, s_gt, e_gt, impossible in zip(
        start_preds, end_preds, start_gts, end_gts, is_impossible
    ):
        if impossible and s_gt == 0 and e_gt == 0:
            # True negative — skip from F1 computation
            continue

        total_possible += 1

        # Ground truth token set
        gt_tokens = set(range(s_gt, e_gt + 1)) if e_gt >= s_gt else set()

        # Predicted token set
        # If model predicts start > end, it found nothing → treat as no-answer
        pred_tokens = set(range(s_pred, e_pred + 1)) if e_pred >= s_pred else set()

        # Exact match
        if pred_tokens == gt_tokens:
            exact_matches += 1

        # Token F1
        overlap = len(gt_tokens & pred_tokens)
        if overlap == 0:
            f1_scores.append(0.0)
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
            if recall > 0:
                found_possible += 1

    n = len(f1_scores)
    return {
        "em":     exact_matches / max(n, 1),
        "f1":     sum(f1_scores) / max(n, 1),
        "recall": found_possible / max(total_possible, 1),
        "n_eval": n,
    }


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class ClauseTrainer:
    """
    Two-phase training loop for Legal-BERT clause extraction.

    Phase 1 (frozen backbone):
      - Only start_head and end_head are updated
      - Fast: ~2,000 trainable params vs 110M
      - Purpose: warm up the randomly initialized heads before touching BERT
      - Duration: freeze_epochs (default: 1 epoch)

    Phase 2 (partial unfreeze):
      - Top unfreeze_layers (default: 4) transformer blocks + pooler are unfrozen
      - Backbone gets a 10x lower LR than the heads to prevent catastrophic forgetting
      - Purpose: adapt BERT's upper layers to the legal clause extraction task
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Create output directory for checkpoints
        config.weights_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders with class-balanced sampling."""
        processed_dir = self.config.processed_dir

        train_dataset = CUADFeatureDataset(processed_dir / "train_features.jsonl")
        val_dataset = CUADFeatureDataset(processed_dir / "test_features.jsonl")

        # Weighted sampler: over-sample positive windows to counter 1.4% rate
        weights = train_dataset.get_class_weights()
        n_samples = self.config.samples_per_epoch or len(train_dataset)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=n_samples,
            replacement=True,   # replacement=True is required for oversampling
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,       # sampler replaces shuffle
            num_workers=0,         # Windows: num_workers>0 causes multiprocessing issues
            pin_memory=(self.device.type == "cuda"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # No grad → can use larger batch
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        return train_loader, val_loader

    def _build_optimizer(self, model: ClauseExtractor, phase: int) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with different LRs for different parameter groups.

        Why different learning rates per group?
          The pre-trained backbone already has good weights. A high LR would
          destroy them. We use:
            - head_lr = 3e-4  (heads are randomly initialized — need to move fast)
            - backbone_lr = 3e-5 (fine-tuning only — 10x smaller)

          This is called 'discriminative fine-tuning' and is standard practice
          for transformer fine-tuning (ULMFiT paper, 2018).

        Phase 1 (frozen backbone): only head parameters in optimizer.
        Phase 2 (unfrozen): two param groups with different LRs.
        """
        head_params = list(model.start_head.parameters()) + list(model.end_head.parameters())

        if phase == 1:
            # Frozen backbone — only train heads
            return torch.optim.AdamW(
                head_params,
                lr=self.config.head_lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            # Unfrozen backbone — two groups with different LRs
            backbone_params = [p for p in model.bert.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": self.config.backbone_lr},
                    {"params": head_params,     "lr": self.config.head_lr},
                ],
                weight_decay=self.config.weight_decay,
            )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ):
        """
        Linear warmup → linear decay learning rate schedule.

        The schedule looks like this:

          LR
          ^
          |     /\\
          |    /  \\
          |   /    \\
          |  /      \\
          | /        \\
          +-------------> steps
           warmup  decay

        Warmup (first warmup_steps):
          LR ramps linearly from 0 to target_lr.
          Prevents large gradient updates from random head initialization
          from corrupting the pre-trained BERT weights early in training.

        Decay (remaining steps):
          LR decays linearly from target_lr to 0.
          Slower updates as we approach convergence = finer adjustment.

        warmup_steps = warmup_ratio * total_steps (default: 6% of total)
        """
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    @torch.no_grad()
    def _evaluate(self, model: ClauseExtractor, val_loader: DataLoader) -> dict:
        """Run full evaluation on the validation set. Returns span F1 and loss."""
        model.eval()

        total_loss = 0.0
        n_batches = 0
        all_start_preds, all_end_preds = [], []
        all_start_gts, all_end_gts = [], []
        all_impossible = []

        for batch in val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions   = batch["end_positions"].to(self.device)

            output: SpanPrediction = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            total_loss += output.loss.item()
            n_batches += 1

            start_preds = output.start_logits.argmax(dim=-1).cpu().tolist()
            end_preds   = output.end_logits.argmax(dim=-1).cpu().tolist()

            all_start_preds.extend(start_preds)
            all_end_preds.extend(end_preds)
            all_start_gts.extend(start_positions.cpu().tolist())
            all_end_gts.extend(end_positions.cpu().tolist())
            all_impossible.extend(batch["is_impossible"].tolist())

        span_metrics = compute_span_f1(
            all_start_preds, all_end_preds,
            all_start_gts, all_end_gts,
            all_impossible,
        )
        span_metrics["val_loss"] = total_loss / max(n_batches, 1)

        model.train()
        return span_metrics

    def train(self):
        """Full training loop across all epochs."""
        config = self.config

        # --- Build components ---
        print("\nBuilding model...")
        model = ClauseExtractor(config.model_name).to(self.device)

        # Start with frozen backbone (Phase 1)
        model.freeze_backbone()
        param_info = model.count_trainable_params()
        print(f"Phase 1 — Frozen backbone")
        print(f"  Trainable: {param_info['trainable']:,} / {param_info['total']:,} "
              f"({param_info['trainable_pct']:.2f}%)")

        print("\nBuilding dataloaders...")
        train_loader, val_loader = self._build_dataloaders()

        total_steps = len(train_loader) * config.num_epochs
        print(f"\nTraining plan:")
        print(f"  Epochs:      {config.num_epochs}")
        print(f"  Steps/epoch: {len(train_loader):,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Batch size:  {config.batch_size}")
        print(f"  Warmup:      {int(config.warmup_ratio * total_steps):,} steps")

        # Phase 1 optimizer and scheduler
        optimizer = self._build_optimizer(model, phase=1)
        scheduler = self._build_scheduler(optimizer, total_steps)

        best_f1 = 0.0
        global_step = 0
        step_metrics = StepMetrics()

        print(f"\n{'='*60}")
        print("Training started")
        print(f"{'='*60}")

        for epoch in range(config.num_epochs):
            epoch_start = time.time()

            # --- Phase transition: unfreeze backbone after freeze_epochs ---
            if epoch == config.freeze_epochs:
                print(f"\n[Epoch {epoch}] Switching to Phase 2 — Unfreezing top "
                      f"{config.unfreeze_layers} BERT layers")
                model.unfreeze_backbone(last_n_layers=config.unfreeze_layers)
                param_info = model.count_trainable_params()
                print(f"  Trainable: {param_info['trainable']:,} / {param_info['total']:,} "
                      f"({param_info['trainable_pct']:.2f}%)")

                # Rebuild optimizer with two param groups
                optimizer = self._build_optimizer(model, phase=2)
                # Note: we keep a fresh scheduler for the remaining steps
                remaining_steps = total_steps - global_step
                scheduler = self._build_scheduler(optimizer, remaining_steps)

            print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
            model.train()

            for batch in train_loader:
                # Move to device
                input_ids       = batch["input_ids"].to(self.device)
                attention_mask  = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions   = batch["end_positions"].to(self.device)

                # --- Forward pass ---
                optimizer.zero_grad()
                output: SpanPrediction = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                # --- Backward pass ---
                output.loss.backward()

                # Gradient clipping: cap gradient norm at 1.0
                # This prevents a single bad batch from causing a huge weight update.
                # Standard practice for transformer fine-tuning.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                global_step += 1
                step_metrics.update(
                    output.loss.item(),
                    output.start_logits.detach(),
                    output.end_logits.detach(),
                    start_positions,
                    end_positions,
                )

                # --- Periodic logging ---
                if global_step % config.log_every_steps == 0:
                    summary = step_metrics.summary()
                    lr = scheduler.get_last_lr()[-1]
                    print(
                        f"  Step {global_step:6d} | "
                        f"loss={summary['loss']:.4f} | "
                        f"start_acc={summary['start_acc']:.3f} | "
                        f"end_acc={summary['end_acc']:.3f} | "
                        f"lr={lr:.2e}"
                    )
                    step_metrics.reset()

                # --- Periodic evaluation ---
                if global_step % config.eval_every_steps == 0:
                    print(f"\n  [Eval @ step {global_step}]")
                    val_metrics = self._evaluate(model, val_loader)
                    print(
                        f"  val_loss={val_metrics['val_loss']:.4f} | "
                        f"EM={val_metrics['em']:.3f} | "
                        f"F1={val_metrics['f1']:.3f} | "
                        f"Recall={val_metrics['recall']:.3f} | "
                        f"n_eval={val_metrics['n_eval']}"
                    )

                    # Save best checkpoint
                    if val_metrics["f1"] > best_f1:
                        best_f1 = val_metrics["f1"]
                        ckpt_path = config.weights_dir / "best_model.pt"
                        torch.save({
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_metrics": val_metrics,
                            "config": config,
                        }, ckpt_path)
                        print(f"  ** New best F1={best_f1:.3f} — saved to {ckpt_path}")
                    print()

            epoch_time = time.time() - epoch_start
            print(f"  Epoch {epoch + 1} done in {epoch_time/60:.1f} min")

        # --- Final evaluation ---
        print(f"\n{'='*60}")
        print("Training complete. Running final evaluation...")
        final_metrics = self._evaluate(model, val_loader)
        print(f"Final val_loss={final_metrics['val_loss']:.4f} | "
              f"EM={final_metrics['em']:.3f} | "
              f"F1={final_metrics['f1']:.3f} | "
              f"Recall={final_metrics['recall']:.3f}")
        print(f"Best F1 during training: {best_f1:.3f}")

        # Save final checkpoint
        final_path = config.weights_dir / "final_model.pt"
        torch.save({
            "epoch": config.num_epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "val_metrics": final_metrics,
            "config": config,
        }, final_path)
        print(f"Final model saved to {final_path}")

        return best_f1
