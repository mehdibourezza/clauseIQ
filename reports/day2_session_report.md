# ClauseIQ — Session Report: Day 2
**Date:** 2026-03-10
**Phase:** 2 — Model Architecture & Training Loop
**Status:** Training ready to run

---

## 1. What We Built Today

- `ml/models/clause_model.py` — Legal-BERT + two linear span heads (ClauseExtractor)
- `ml/data/dataset.py` — PyTorch Dataset for JSONL features + class weight computation
- `ml/training/trainer.py` — Full two-phase training loop with AdamW + warmup schedule
- `ml/training/train.py` — CLI entry point with argument parsing

---

## 2. The Model: ClauseExtractor

### Architecture

```
Input: [CLS] question [SEP] context [SEP] [PAD]...
         |
         v
Legal-BERT (110M params)
  - 12 transformer layers
  - 768-dimensional hidden states
  - Trained on EU legislation, US court cases, contracts
         |
         v
Dropout(p=0.1) applied to all token representations
         |
    _____|_____
    |         |
    v         v
start_head  end_head
Linear(768→1) Linear(768→1)
    |         |
    v         v
start_logits  end_logits
shape: (batch, 512)    shape: (batch, 512)
    |         |
    v         v
argmax     argmax
    |         |
    v         v
start_token  end_token
```

### What the heads learn

Each head applies a single linear layer across every token position:

```
start_score[i] = dot(W_start, h[i]) + b_start
end_score[i]   = dot(W_end,   h[i]) + b_end
```

Where `h[i]` is the 768-dimensional hidden state of token `i` from BERT.

These are NOT independent per-token classifiers — they share the BERT
backbone, so every token's score is informed by all other tokens (via
self-attention). The head just "reads" the final representation.

### Padding masking

Before taking argmax, padding positions get -inf logits:

```python
padding_mask = (attention_mask == 0)  # True where padded
start_logits = start_logits.masked_fill(padding_mask, float("-inf"))
```

Without this, the model might predict a padding token as the start of a
clause, which is nonsensical. Setting to -inf ensures softmax gives them
probability ~0.

---

## 3. The Loss Function

### Cross-entropy over token positions

The span extraction task is framed as two independent classification problems,
each with 512 classes (one per token position):

```
loss = CE(start_logits, start_gt) + CE(end_logits, end_gt)
```

Cross-entropy loss for K classes at one position:

```
CE(logits, target) = -log( exp(logits[target]) / sum_j exp(logits[j]) )
            = -logits[target] + log(sum_j exp(logits[j]))
```

The model is maximizing the log-probability of the correct token position.

### Why sum and not average?

Both boundaries are equally important. Summing means:
- total loss ~ 2x a single CE loss
- Consistent across all training steps (no variation from single vs double answers)
- Standard practice in SQuAD QA models (original BERT paper)

### "No answer" convention

For windows where the clause is absent, ground truth is:
```
start_gt = 0, end_gt = 0
```

Position 0 is always the [CLS] token. The model learns:
- "If the clause is here, predict its token span"
- "If the clause is absent, predict (0, 0)"

This is elegant because it turns "no answer detection" into a natural
side effect of span prediction, not a separate classification task.

---

## 4. The Class Imbalance Problem

### Scale of the problem

From yesterday's preprocessing:
```
Train features (windows): 907,520
  Positive (clause present): 12,782  (1.41%)
  Negative (no clause):     894,738  (98.59%)

Ratio: 1 positive for every 70 negatives
```

### Why this is catastrophic for training

If you feed the model batches with this natural distribution:
- A batch of 16 contains on average 0.22 positive examples
- Most batches have ZERO positive examples
- The model never sees a clause span during training
- It learns to always predict (0, 0) → achieves 98.6% accuracy
- But span F1 = 0.0 (never finds any clause)

### Solution: WeightedRandomSampler

Instead of fixing the loss function, we fix what the model SEES:

```python
w_positive = n_total / (2 * n_positive)  = 907520 / (2 * 12782)  = 35.5
w_negative = n_total / (2 * n_negative)  = 907520 / (2 * 894738) = 0.507
```

Each positive example is weighted ~70x more than each negative.
When WeightedRandomSampler draws samples, it samples proportional to weight.

Result: each batch has roughly 50% positive, 50% negative examples.

```
Without sampler:  batch of 16 has 0-1 positive examples
With sampler:     batch of 16 has ~8 positive examples
```

### Why this and not loss reweighting?

Both approaches are valid. WeightedRandomSampler wins because:
- Gradient signal: 8 positive examples per batch produce 8 independent
  span gradients, not 1 gradient amplified 70x
- Stability: amplified gradients can cause loss spikes
- Intuition: we want the model to genuinely SEE clauses, not just be
  penalized more when it misses them

---

## 5. The Optimizer: AdamW

### Why Adam at all?

Stochastic Gradient Descent (SGD) uses one global learning rate for all
parameters. The problem: different parameters need very different update sizes.

- BERT's embedding layer (learned from billions of words): needs tiny updates
- The new span heads (randomly initialized): need large updates

Adam solves this with per-parameter adaptive learning rates:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # gradient momentum
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        # squared gradient momentum
theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + eps)
```

- Parameters with consistently large gradients get a SMALLER effective LR
- Parameters with small, noisy gradients get a LARGER effective LR
- This is "gradient normalization" — every parameter moves at its own pace

Standard values (and the ones we use):
```
beta1 = 0.9    (momentum decay — smooth 90% of old gradient)
beta2 = 0.999  (variance decay — track long-term gradient scale)
eps   = 1e-8   (numerical stability)
```

### AdamW: decoupled weight decay

Plain Adam has a subtle bug: L2 regularization (weight decay) interacts
with the adaptive learning rate in a way that makes regularization ineffective
for parameters with large gradients.

AdamW fixes this by applying weight decay DIRECTLY to the weights, not
through the gradient:

```
# Plain Adam with L2:   g_t += lambda * theta   (adds to gradient)
# AdamW:                theta_t -= lr * lambda * theta   (direct subtraction)
```

The distinction matters because: in plain Adam, large-gradient parameters
(which have small effective LR) barely get regularized. In AdamW, all
parameters are regularized equally regardless of gradient scale.

We use `weight_decay=0.01`, which is standard for BERT fine-tuning.

### Discriminative learning rates

We use two different learning rates:

```
head_lr     = 3e-4   (heads: randomly initialized → need to move fast)
backbone_lr = 3e-5   (BERT: already trained → small adjustments only)
```

This is called "discriminative fine-tuning" (Howard & Ruder, 2018).
The intuition: the backbone is like a skilled expert who just needs
minor recalibration. The heads are new employees who need to learn
everything from scratch.

With a 10x lower LR for the backbone:
- It adapts to legal clause extraction
- It retains its pre-trained syntactic and semantic knowledge
- No "catastrophic forgetting" of what it learned during pre-training

---

## 6. The Learning Rate Schedule: Linear Warmup + Decay

### The warmup phase

At step 0, the span heads are randomly initialized. Their gradients
point in random directions. If we use the full LR immediately, these
random gradients can push BERT's weights away from the pre-trained optimum
in the first few steps.

Warmup ramps the LR from 0 to target over warmup_steps:

```
LR at step t = target_lr * (t / warmup_steps)   for t < warmup_steps
```

We use warmup_ratio = 0.06, so 6% of total steps are warmup:
```
total_steps = 3 epochs * ~56,720 steps/epoch = ~170,160 steps
warmup_steps = 0.06 * 170,160 = ~10,210 steps
```

During warmup, the model explores slowly, letting the heads form coherent
weights before the backbone starts being nudged.

### The decay phase

After warmup, LR decays linearly to 0:

```
LR at step t = target_lr * (1 - (t - warmup) / (total - warmup))
```

Why decay at all? As training progresses, the model gets closer to a good
optimum. Large steps at this point would overshoot. Decaying LR = finer
adjustments as we approach convergence.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the gradient norm exceeds 1.0, it's scaled down to 1.0. This prevents
a single bad batch (malformed input, very long span) from causing a huge
weight update that wrecks the model. Standard for all transformer training.

---

## 7. Evaluation: Why Not Accuracy?

### The F1 score for span prediction

With 98.6% negatives, a model predicting (0,0) for everything achieves:
```
Accuracy = 98.6%
F1 = 0.0   (never found a single clause)
```

We evaluate with span-level F1 instead:

```
For each positive example:
  GT span    = {token_i, token_{i+1}, ..., token_j}
  Pred span  = {token_k, token_{k+1}, ..., token_l}

  overlap   = |GT ∩ Pred|
  precision = |GT ∩ Pred| / |Pred|
  recall    = |GT ∩ Pred| / |GT|
  F1        = 2 * precision * recall / (precision + recall)
```

Then we average F1 across all positive examples.

Why token-level overlap and not character-level?
- We're working in token space during training
- Token F1 is what the CUAD leaderboard uses for fair comparison
- Character F1 would require re-running the tokenizer on predictions

### Exact Match (EM)

EM = 1 if the predicted span matches the ground truth token-for-token, 0 otherwise.
Useful as a strict upper-bound check: "did we get it exactly right?"

EM is harder to optimize than F1 — a prediction that's off by one token
gets EM=0 but F1≈0.9.

### Recall

We also track recall separately:
```
recall = fraction of positive examples where the model predicted
         a non-zero span (i.e. it tried to find a clause)
```

A model that's always conservative (predicts no-answer) will have low recall.
Watching recall tells us if the model is learning to "look" for clauses at all.

---

## 8. Two-Phase Training Strategy

### Phase 1: Frozen backbone (epoch 0)

```
Trainable: ~1,538 params (start_head + end_head only)
Frozen:    ~110M params (all of Legal-BERT)
```

Purpose: warm up the heads to a reasonable state before touching BERT.

Why not unfreeze everything from the start?
- 1 epoch = ~56,720 steps with our sampler
- Early gradients from random heads are noisy
- Those noisy gradients flowing back through 12 BERT layers = catastrophic forgetting
- After 1 epoch, heads have stable gradients → safe to update backbone

### Phase 2: Partial unfreeze (epochs 1-2)

```
Unfrozen: top 4 transformer layers + pooler
Frozen:   bottom 8 layers

Trainable: ~28M / 110M = 25% of model
```

Why only the top layers?

BERT learns hierarchically:
- Layers 1-4: Basic syntax (part-of-speech, noun phrases)
- Layers 5-8: Semantics (word meaning, coreference)
- Layers 9-12: Task-specific (complex reasoning, long-range dependencies)

For clause extraction, we need to adapt layers 9-12 to recognize legal span
boundaries. Layers 1-8 are already good at syntax/semantics — touching them
risks destroying general language understanding that helps us.

Fine-tuning only the top N layers is the standard recipe from the BERT paper
and confirmed by most transformer transfer learning research.

---

## 9. ML Decisions Made Today

| Decision | Choice | Rationale |
|---|---|---|
| Class imbalance | WeightedRandomSampler | Controls what model SEES, not just loss penalty — more stable |
| Optimizer | AdamW | Decoupled weight decay → effective regularization + adaptive LR |
| LR schedule | Linear warmup + decay | Prevents catastrophic forgetting early, fine adjustment late |
| Backbone LR | 3e-5 (10x lower) | Discriminative fine-tuning — preserve pre-training |
| Head LR | 3e-4 | Random init needs large initial updates |
| Gradient clipping | max_norm=1.0 | Prevents bad batches from corrupting weights |
| Phase 1 freeze | 1 epoch | Stabilize heads before unfreezing backbone |
| Unfreeze layers | top 4 | Task-specific layers; preserve general syntax/semantics below |
| Evaluation metric | Span F1 + Recall | Accuracy is misleading with 1.4% positive rate |

---

## 10. Tomorrow

- Launch training run: `python -m ml.training.train`
- Monitor Phase 1 → Phase 2 transition (watch if F1 improves after unfreeze)
- First real results: what does the model learn after 1 epoch?
- Build inference engine: post-processing to select best window per contract
- Begin FastAPI backend: `/v1/analyze` endpoint (async + Celery)

---

*Report generated during ClauseIQ Phase 2 development.*
