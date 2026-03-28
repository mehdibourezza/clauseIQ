# ClauseIQ — Session Report: Day 1
**Date:** 2026-03-10
**Phase:** 1 — Data Pipeline & Preprocessing
**Status:** In progress (preprocessing running)

---

## 1. What We Built Today

- Downloaded the CUAD dataset (408 contracts, 22,450 QA examples, 41 clause types)
- Wrote a tokenization pipeline converting raw legal text into model-ready features
- Debugged two memory failures and fixed them with incremental disk writes + GC

---

## 2. The Task: Extractive Question Answering

### What the model does mathematically

Given a question Q and a context passage C, the model predicts two integers:

```
start_position ∈ {0, 1, ..., L-1}
end_position   ∈ {0, 1, ..., L-1}
```

Where L = 512 (max sequence length). The predicted answer span is then:

```
answer = tokens[start_position : end_position + 1]
```

For each of the 41 clause types, we ask the model:
- "Does this paragraph contain a Cap On Liability clause?"
- "If yes, which exact tokens form it?"

If the clause is absent, the model predicts `(start=0, end=0)` — the `[CLS]` token position, which conventionally means "no answer".

---

## 3. The Input Representation: How BERT Reads Text

### Tokenization

Raw text is broken into subword tokens using WordPiece tokenization. For example:

```
"indemnification" → ["ind", "##emni", "##fication"]
"liability"       → ["liability"]   (common word, stays whole)
```

The `##` prefix means "continuation of previous token." This allows the model to handle any word, including rare legal terms it has never seen exactly.

### The input sequence structure

Every input to Legal-BERT looks like this:

```
[CLS] question tokens [SEP] context tokens [SEP] [PAD] [PAD] ...
```

Concretely for one training example:

```
Position:  0    1   2   3  ...  67   68  69  70  71  ... 511
Token:   [CLS] What is  a  ... [SEP] The party shall ... [PAD]
Segment:   0    0   0   0  ...   0    1   1   1   1  ...   0
```

- Segment 0 = question
- Segment 1 = context paragraph
- The model uses this to distinguish "what I'm being asked" from "where the answer might be"

---

## 4. The Core Problem: Contracts Are Too Long

### Why 512 tokens is a hard limit

BERT uses self-attention, which computes a similarity score between every pair of tokens. For a sequence of length L, that's L × L = L² attention scores.

```
L = 512  → 512² = 262,144 attention pairs per layer
L = 2048 → 2048² = 4,194,304 attention pairs per layer (16x more memory)
```

This quadratic scaling is why transformers have a length limit. A full contract of 40 pages is roughly 15,000–50,000 tokens — completely impossible to process at once.

### The sliding window solution

We chop each contract into overlapping windows:

```
Parameters:
  max_length = 512 tokens
  doc_stride = 128 tokens (overlap)

Window 0: tokens[0   : 512]
Window 1: tokens[384 : 896]   (512 - 128 = 384 start)
Window 2: tokens[768 : 1280]
...
```

The overlap (stride = 128) ensures that:
- No clause is cut off right at a window boundary
- The model sees the same text in multiple windows from different positions
- At inference time, we take the window with the highest confidence answer

For a 5,000-token contract: `ceil((5000 - 512) / (512 - 128)) + 1 ≈ 13 windows`

Each window becomes an **independent training example**. This is why 22,450 raw QA pairs expand to many more tokenized features.

---

## 5. Answer Span Labeling: The Tricky Part

### Character positions → token positions

The CUAD dataset gives us answers as character offsets:

```json
{
  "text": "This Agreement shall be governed by the laws of Delaware",
  "answer_start": 4521
}
```

But the model works in token space, not character space. We need to convert:

```
character_start = 4521
character_end   = 4521 + len("This Agreement...") = 4577

→ find token t_start such that offset_mapping[t_start][0] >= 4521
→ find token t_end   such that offset_mapping[t_end][1]  <= 4577
```

The `offset_mapping` from the tokenizer maps each token index to its `(start_char, end_char)` in the original text. This is what `return_offsets_mapping=True` gives us.

### Answer in window check

For each window we check: is the answer fully contained in this chunk?

```
window_covers_answer = (answer_start_char >= window_start_char)
                       AND
                       (answer_end_char <= window_end_char)
```

If not, this window is labeled as "impossible" (no answer present) — even though the answer exists elsewhere in the document. The model learns that most windows don't contain the clause.

---

## 6. Dataset Statistics & ML Implications

### Class imbalance

From our exploration:

```
Total QA examples:  22,450
Positive (clause present): ~8,300  (~37%)
Negative (clause absent):  ~14,150  (~63%)
```

At the window level this gets worse — most windows of a 20-page contract won't contain the specific clause we're looking for. The effective positive rate at window level is likely 10–20%.

**Why this matters:** A naive model can achieve 80%+ accuracy just by always predicting "no answer." We need to evaluate with F1 score on the span, not raw accuracy.

### The 41 clause types: very different frequencies

```
Parties:               2,011 positive examples  (easiest — nearly every contract)
Governing Law:           374 positive examples  (common)
Source Code Escrow:       61 positive examples  (rare — software contracts only)
Unlimited License:        26 positive examples  (very rare)
```

Rare clauses will be harder for the model. We may need to weight the loss function or use specialized evaluation per clause type.

---

## 7. Why Legal-BERT Instead of Plain BERT

### Pre-training domain matters

Standard BERT was pre-trained on Wikipedia + BookCorpus — general English. Legal text has very different statistical properties:

- **Vocabulary:** "indemnify," "hereinafter," "notwithstanding," "ipso facto"
- **Sentence structure:** Extremely long, nested clauses with many subclauses
- **Semantics:** "shall" means obligation (not future tense), "party" means a signing entity (not a social event)

Legal-BERT (`nlpaueb/legal-bert-base-uncased`) was pre-trained on:
- EU legislation
- US court cases
- Contracts and legal filings

Result: the token embeddings already encode legal meaning before we even start fine-tuning. Fine-tuning on CUAD then teaches it the specific task of clause extraction.

**Analogy:** It's like hiring someone who already knows legal vocabulary (Legal-BERT) vs someone who has to learn what "indemnification" means from scratch (vanilla BERT).

---

## 8. ML Decisions Made Today

| Decision | Choice | Rationale |
|---|---|---|
| Base model | Legal-BERT | Domain-matched pre-training beats general BERT on legal tasks |
| Task formulation | Extractive QA | Returns exact evidence span, not just yes/no — more useful and explainable |
| Window size | 512 tokens | BERT's hard limit |
| Stride | 128 tokens | 25% overlap prevents clause boundary cuts; standard in SQuAD literature |
| Tokenizer truncation | `only_second` | Never truncate the question — only the context gets windowed |
| Storage format | JSONL (streaming) | Prevents OOM; allows future dataset sizes beyond RAM |

---

## 9. Tomorrow

- Run the model training loop (Legal-BERT fine-tuning on CUAD)
- Understand the loss function: cross-entropy on start and end positions independently
- Understand the optimizer: AdamW with weight decay and linear warmup schedule
- First evaluation: Exact Match (EM) and F1 on span predictions

---

*Report generated during ClauseIQ Phase 1 development.*
