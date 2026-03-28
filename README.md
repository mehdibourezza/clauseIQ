# ClauseIQ

Legal contract intelligence engine powered by Legal-BERT. Extracts 41 types of clauses from commercial contracts using extractive question answering — returns exact text spans, not generated summaries.

Built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (408 contracts, 22,450 labeled QA pairs).

---

## What It Does

Given a contract and a clause type, ClauseIQ finds and returns the exact sentence(s) in the contract that correspond to that clause.

```
Input:  contract text + "Does this contract contain a Cap on Liability clause?"
Output: "In no event shall either party's total liability exceed $500,000."
        start_token: 4821  end_token: 4849  confidence: 0.94
```

It handles 41 clause types including:
- Cap on Liability
- Governing Law
- Termination for Convenience
- Indemnification
- Non-Compete
- Renewal Term
- Source Code Escrow
- ... and 34 more

---

## Why Not Just Use ChatGPT?

| | ChatGPT / Claude API | ClauseIQ |
|---|---|---|
| Data privacy | Contract sent to external server | Runs fully on-premise |
| Output format | Prose paragraph | Structured JSON with character positions |
| Portfolio analysis | One contract at a time | Query across hundreds of contracts |
| Accuracy benchmark | No published benchmark | CUAD F1 score per clause type |
| Cost at scale (600 contracts × 41 clauses) | ~$1,200 | ~$0.60 |
| Speed | ~30 min | ~2 min |

Enterprise legal teams and law firms cannot send client contracts to third-party APIs due to attorney-client privilege and compliance requirements. ClauseIQ is designed to run inside the client's own infrastructure.

---

## Project Status

### Done
- [x] Data pipeline — CUAD dataset download and preprocessing (`scripts/`)
- [x] Tokenization — sliding window approach (512 tokens, 128-token stride)
- [x] Model architecture — Legal-BERT + dual span extraction heads (`ml/models/`)
- [x] Training pipeline — two-phase fine-tuning with AdamW, warmup/decay, class-balanced sampling (`ml/training/`)
- [x] Evaluation metrics — Span F1, Exact Match, Recall

### In Progress
- [ ] Training run — executing on GPU (Colab T4), weights not yet saved
- [ ] Streaming dataset — rewriting `dataset.py` to avoid loading 4GB into RAM

### Not Started
- [ ] Inference engine — sliding window inference + multi-window answer aggregation (`ml/inference/`)
- [ ] FastAPI backend — REST endpoints for contract upload and clause extraction (`api/`)
- [ ] Database layer — SQLAlchemy models for contracts, extractions, API keys (`data/db/`)
- [ ] Storage client — S3/MinIO integration for PDF uploads (`data/storage/`)
- [ ] Celery workers — async background processing (`workers/`)
- [ ] Frontend — contract upload UI, clause highlighting (`frontend/`)
- [ ] Docker — containerization for deployment (`docker/`)
- [ ] Tests — unit and integration tests (`tests/`)

---

## Architecture

```
Contract PDF
     ↓
PDF Parser (pdfplumber)
     ↓
Sliding Window Tokenizer (Legal-BERT tokenizer, 512 tokens, 128 stride)
     ↓
ClauseExtractor (Legal-BERT + dual linear span heads)
     ↓
┌─────────────────────────────┐
│  start_logits: [512 scores] │  → argmax → predicted start token
│  end_logits:   [512 scores] │  → argmax → predicted end token
└─────────────────────────────┘
     ↓
Multi-window aggregation (pick highest confidence window per clause)
     ↓
{ clause_type, extracted_text, start_char, end_char, confidence }
```

### Model

- **Backbone:** [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) — 110M parameters, pre-trained on EU legislation, US court cases, and contracts
- **Heads:** Two independent linear layers (768 → 1) for start and end token classification
- **Task:** Extractive QA — predict the (start, end) token span of the answer

### Training

- **Dataset:** CUAD — 907,520 tokenized windows (after sliding window expansion)
- **Class imbalance:** 1.4% positive (clause present) / 98.6% negative — handled with WeightedRandomSampler
- **Phase 1 (1 epoch):** Freeze backbone, train only span heads (~1,500 params)
- **Phase 2 (2 epochs):** Unfreeze top 4 BERT layers, discriminative LRs (head: 3e-4, backbone: 3e-5)
- **Schedule:** Linear warmup (6%) + linear decay
- **Gradient clipping:** max norm 1.0

---

## Quickstart

### Requirements

- Python 3.9+
- Conda (recommended) or virtualenv
- 8GB+ RAM for training (16GB recommended)
- GPU strongly recommended for training (CPU training will take days)

### Setup

```bash
git clone https://github.com/mehdibourezza/clauseiq
cd clauseiq

conda create -n clauseiq python=3.9
conda activate clauseiq
pip install -r requirements.txt

cp .env.example .env
```

### Step 1 — Download the CUAD dataset

```bash
python -m scripts.download_cuad
```

Downloads 408 commercial contracts + 22,450 labeled QA pairs (~18MB) into `ml/data/raw/`.

### Step 2 — Preprocess (tokenize)

```bash
python -m scripts.preprocess_cuad
```

Tokenizes the dataset into sliding window features. Outputs:
- `ml/data/processed/train_features.jsonl` (~4GB, 907K windows)
- `ml/data/processed/test_features.jsonl` (~538MB, 120K windows)

This takes 20-40 minutes.

### Step 3 — Train

```bash
python -m ml.training.train
```

**Requires a GPU.** On a T4 GPU (Google Colab): ~3-5 hours.

Custom options:
```bash
python -m ml.training.train --epochs 3 --batch-size 16 --freeze-epochs 1
```

Saves checkpoints to `ml/models/weights/best_model.pt`.

### Training on Google Colab (free T4 GPU)

If you don't have a local GPU, use Google Colab:

1. Upload `train_features.jsonl` and `test_features.jsonl` to Google Drive
2. Open a new Colab notebook, set runtime to T4 GPU
3. Mount Drive, copy code files, run training — see `reports/colab_training.md` for step-by-step instructions

> **Note:** Free Colab has 12GB RAM. The full dataset (4GB JSONL → ~12GB in RAM as Python objects) causes OOM crashes. Use a subsampled dataset (~113K examples) or upgrade to Colab Pro.

---

## Repository Structure

```
clauseiq/
├── ml/
│   ├── models/
│   │   └── clause_model.py       # ClauseExtractor: Legal-BERT + span heads
│   ├── data/
│   │   └── dataset.py            # PyTorch Dataset for tokenized JSONL features
│   ├── training/
│   │   ├── train.py              # CLI entry point
│   │   └── trainer.py            # Full training loop, metrics, optimizer
│   ├── inference/                # TODO: sliding window inference engine
│   └── evaluation/               # TODO: per-clause F1 analysis
│
├── api/                          # TODO: FastAPI REST endpoints
│   └── v1/routers/               # TODO: /analyze, /contracts, /results
│
├── data/
│   ├── db/                       # TODO: SQLAlchemy ORM models
│   └── storage/                  # TODO: S3/MinIO client
│
├── workers/                      # TODO: Celery async tasks
├── frontend/                     # TODO: React UI
├── docker/                       # TODO: Dockerfile + docker-compose
├── tests/                        # TODO: pytest unit + integration tests
│
├── scripts/
│   ├── download_cuad.py          # Download CUAD from GitHub
│   └── preprocess_cuad.py        # Tokenize into sliding window JSONL
│
├── configs/
│   └── config.py                 # Pydantic settings (DB, Redis, ML, S3)
│
├── alembic/                      # Database migrations (framework ready, no migrations yet)
├── requirements.txt
└── .env.example
```

---

## Roadmap

### Phase 3 — Inference Engine
- Load trained checkpoint
- Apply sliding window inference across full contracts
- Aggregate predictions across windows (select highest confidence per clause)
- Post-processing: map token positions back to character positions

### Phase 4 — Backend API
- `POST /v1/contracts` — upload a contract PDF
- `GET /v1/contracts/{id}/clauses` — get all extracted clauses
- `POST /v1/analyze` — synchronous single-contract analysis
- JWT authentication, API key management, rate limiting

### Phase 5 — Portfolio Analytics
- Store all extractions in PostgreSQL
- Query: "show all contracts missing a liability cap"
- Query: "flag contracts where governing law is not Delaware"
- Dashboard with clause coverage across contract portfolio

### Phase 6 — Deployment
- Docker + docker-compose for local deployment
- On-premise deployment guide (no data leaves client infrastructure)

---

## Contributing

This project is in active early development. The ML training pipeline is complete; everything above Phase 2 is scaffolded but not implemented.

If you want to contribute, the highest-value areas are:
1. **Streaming dataset** — rewrite `dataset.py` as `IterableDataset` to avoid OOM on large files
2. **Inference engine** — `ml/inference/` is entirely empty
3. **FastAPI endpoints** — `api/v1/routers/` is entirely empty

---

## Dataset

This project uses the **CUAD (Contract Understanding Atticus Dataset)**:

> Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", 2021.
> [https://arxiv.org/abs/2103.06268](https://arxiv.org/abs/2103.06268)

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## License

MIT
