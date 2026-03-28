"""
Legal-BERT fine-tuned for extractive clause QA.

Architecture:
  - Backbone: nlpaueb/legal-bert-base-uncased (110M parameters)
  - Head: two linear projections over all token hidden states
      start_logits[i] = W_s · h[i]   (is token i the start of the answer?)
      end_logits[i]   = W_e · h[i]   (is token i the end of the answer?)

The model outputs a probability distribution over all 512 token positions
for both start and end. The predicted span is (argmax(start), argmax(end)).

For "no answer" (clause absent), the correct label is position 0 ([CLS] token).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass


@dataclass
class SpanPrediction:
    start_logits: torch.Tensor   # shape: (batch, seq_len)
    end_logits: torch.Tensor     # shape: (batch, seq_len)
    loss: torch.Tensor | None    # scalar, only during training


class ClauseExtractor(nn.Module):
    """
    Legal-BERT with a span extraction head.

    The backbone produces a hidden state h[i] for every token i.
    Two independent linear layers project each h[i] to a scalar score:
      start_score[i] = dot(W_start, h[i]) + b_start
      end_score[i]   = dot(W_end,   h[i]) + b_end

    These scores are passed through softmax to get a probability distribution
    over token positions. Training minimizes cross-entropy at both positions.
    """

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size  # 768 for BERT-base

        # Two independent span boundary predictors
        # Each maps from hidden_size (768) → 1 scalar per token
        self.start_head = nn.Linear(self.hidden_size, 1)
        self.end_head = nn.Linear(self.hidden_size, 1)

        # Dropout for regularization (applied before the heads)
        self.dropout = nn.Dropout(p=0.1)

        # Initialize heads with small weights (don't overpower the pre-trained backbone)
        nn.init.normal_(self.start_head.weight, std=0.02)
        nn.init.normal_(self.end_head.weight, std=0.02)
        nn.init.zeros_(self.start_head.bias)
        nn.init.zeros_(self.end_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        attention_mask: torch.Tensor,   # (batch, seq_len)
        start_positions: torch.Tensor | None = None,  # (batch,) ground truth
        end_positions: torch.Tensor | None = None,    # (batch,) ground truth
    ) -> SpanPrediction:

        # --- BERT forward pass ---
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
        # Every token gets a 768-dimensional representation
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)

        # --- Span head ---
        # (batch, seq_len, hidden_size) @ (hidden_size, 1) → (batch, seq_len, 1)
        # squeeze(-1) → (batch, seq_len)
        start_logits = self.start_head(sequence_output).squeeze(-1)
        end_logits = self.end_head(sequence_output).squeeze(-1)

        # Mask padding tokens — they should never be predicted as span boundaries
        # Set their logits to -inf before softmax so they get probability ~0
        padding_mask = (attention_mask == 0)
        start_logits = start_logits.masked_fill(padding_mask, float("-inf"))
        end_logits = end_logits.masked_fill(padding_mask, float("-inf"))

        # --- Loss computation (only during training) ---
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self._compute_loss(
                start_logits, end_logits,
                start_positions, end_positions,
                seq_len=input_ids.size(1),
            )

        return SpanPrediction(
            start_logits=start_logits,
            end_logits=end_logits,
            loss=loss,
        )

    def _compute_loss(
        self,
        start_logits: torch.Tensor,   # (batch, seq_len)
        end_logits: torch.Tensor,     # (batch, seq_len)
        start_positions: torch.Tensor,  # (batch,)
        end_positions: torch.Tensor,    # (batch,)
        seq_len: int,
    ) -> torch.Tensor:
        """
        Cross-entropy loss over token positions for both span boundaries.

        Loss = CE(start_logits, start_positions) + CE(end_logits, end_positions)

        Both are treated as independent multi-class classification problems
        over seq_len=512 classes (token positions).

        For impossible examples (no answer), ground truth is position 0 ([CLS]).
        The model learns that predicting the [CLS] token = "I see no clause here."

        Why sum and not average? Both boundaries are equally important.
        Using sum means total loss ≈ 2x a single CE loss, which is consistent
        across all training steps.
        """
        # Clamp positions to [0, seq_len-1] — safety guard against label errors
        start_positions = start_positions.clamp(0, seq_len - 1)
        end_positions = end_positions.clamp(0, seq_len - 1)

        loss_fn = nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)

        return start_loss + end_loss

    def freeze_backbone(self):
        """Freeze all BERT weights. Only train the span heads."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_layers: int = 4):
        """
        Unfreeze the last N transformer layers of BERT for fine-tuning.

        Why not unfreeze everything at once?
        - Lower layers encode general syntax/vocabulary (already good)
        - Upper layers encode task-specific semantics (need to adapt)
        - Unfreezing everything risks catastrophic forgetting of pre-trained knowledge
        - Standard practice: unfreeze top 2-4 layers, use LR 10x smaller than head
        """
        # First freeze everything
        for param in self.bert.parameters():
            param.requires_grad = False

        # Then unfreeze the top N encoder layers
        encoder_layers = self.bert.encoder.layer
        total_layers = len(encoder_layers)
        for i in range(total_layers - last_n_layers, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Always unfreeze the pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

    def count_trainable_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": 100 * trainable / total,
        }
