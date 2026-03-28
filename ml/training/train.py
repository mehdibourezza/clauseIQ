"""
Entry point for ClauseIQ model training.

Run:
    conda activate clauseiq
    python -m ml.training.train

Or with custom settings:
    python -m ml.training.train --epochs 5 --batch-size 8 --freeze-epochs 2
"""

import argparse
from pathlib import Path
from ml.training.trainer import ClauseTrainer, TrainingConfig


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train ClauseExtractor on CUAD")

    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--freeze-epochs", type=int,   default=1,
                        help="Epochs to keep backbone frozen (Phase 1)")
    parser.add_argument("--unfreeze-layers", type=int, default=4,
                        help="Top N BERT layers to unfreeze in Phase 2")
    parser.add_argument("--head-lr",       type=float, default=3e-4)
    parser.add_argument("--backbone-lr",   type=float, default=3e-5)
    parser.add_argument("--samples-per-epoch", type=int, default=None,
                        help="Samples to draw per epoch (default: full dataset)")
    parser.add_argument("--processed-dir", type=str,  default="ml/data/processed")
    parser.add_argument("--weights-dir",   type=str,  default="ml/models/weights")

    args = parser.parse_args()

    return TrainingConfig(
        processed_dir=Path(args.processed_dir),
        weights_dir=Path(args.weights_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
        unfreeze_layers=args.unfreeze_layers,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        samples_per_epoch=args.samples_per_epoch,
    )


if __name__ == "__main__":
    config = parse_args()

    print("ClauseIQ Training")
    print("=" * 60)
    print(f"Model:          {config.model_name}")
    print(f"Epochs:         {config.num_epochs} (freeze first {config.freeze_epochs})")
    print(f"Batch size:     {config.batch_size}")
    print(f"Head LR:        {config.head_lr}")
    print(f"Backbone LR:    {config.backbone_lr}")
    print(f"Unfreeze top:   {config.unfreeze_layers} layers")
    print(f"Device:         {config.device}")
    print()

    trainer = ClauseTrainer(config)
    best_f1 = trainer.train()

    print(f"\nDone. Best span F1: {best_f1:.4f}")
