# ClauseIQ

Contract review tool for lawyers. ClauseIQ finds 41 types of clauses in commercial contracts (cap on liability, governing law, non-compete, change of control...) and returns the **exact sentences** from the contract, each one **verified against the contract text** before it is shown.

Built on the [CUAD dataset](https://www.atticusprojectai.org/cuad): 510 commercial contracts annotated by lawyers for 41 clause categories.

Two extraction engines sit behind one interface:

| Engine | Status | Trade-off |
|---|---|---|
| **LLM** (Claude, structured outputs + verbatim verification) | Working, used by the app | Fast to put in place; the contract text is sent to the model provider (Anthropic API, US). Claude is also offered on Microsoft Foundry, but without an EU data zone as of July 2026 |
| **Legal-BERT** (fine-tuned on CUAD) | Training pipeline built, model not yet trained | Self-hosted: no contract leaves the infrastructure, but needs training and evaluation first |

---

## Contract Review App

A lawyer uploads a contract (PDF or TXT) and sees, for each of the 41 clause types, whether it is present, with the exact sentences highlighted in the contract.

### Architecture

```
[Angular 17 frontend]
   │ POST /api/v1/contracts  (PDF or TXT, max 10 MB)
   ▼
[FastAPI] ── text extraction (pypdf) ──► contracts table (status = processing)
   │ 202 Accepted {id}, then background task
   ▼
[ClauseExtractor]  interface (core/)
   │   ChunkingExtractor (core/): contracts > 400,000 chars are split into overlapping parts
   │   ClaudeExtractor (engines/): ONE call per part, full text + 41 category definitions
   │ ◄── JSON constrained by a schema: 41 × {clause_type, excerpts[]}
   ▼
[QuoteVerifier]  every excerpt must exist verbatim in the contract
   │   found     → (start_char, end_char), verified = true
   │   not found → verified = false (stored for measurement, never highlighted)
   ▼
[SQLite via SQLAlchemy]  contracts, clause_findings (status = done)
   ▲
   │ GET /api/v1/contracts/{id}, polled every 2 s until done
[Angular]  clause list ◄──► contract text with highlighted excerpts
```

Design choices:

- **Verification, not trust.** An LLM can reword a sentence while presenting it as a quote. Only excerpts found verbatim in the contract are highlighted (whitespace and typographic quotes/dashes are normalised, nothing else; no fuzzy matching). Rejected excerpts are kept and counted, so the model's invention rate is measurable.
- **One call for 41 categories.** The contract's tokens are paid for once, instead of 41 times.
- **202 Accepted + polling.** An analysis takes tens of seconds (25 s on average over 20 CUAD contracts, see Evaluation); the upload returns immediately and the frontend polls the status.
- **Ports and adapters.** `core/` (extraction port, chunking, verification, analysis) knows nothing about HTTP, databases or model providers; `tests/test_architecture.py` fails if it ever imports them. FastAPI, SQLite and Claude (`engines/claude.py`) are adapters around it: the same core can run behind an Azure Function, another engine can replace Claude, and the tests plug in a fake extractor.
- **Long contracts.** Above `LLM_MAX_CHARS_PER_CALL` (default 400,000 characters, above the longest CUAD contract), the contract is extracted in parts that overlap by 4,000 characters (above the longest CUAD annotated clause, 3,884), so every clause appears whole in at least one part. Verification still runs on the full text, so offsets need no adjustment. If one part fails, the whole analysis fails: a partial result would report clauses as absent in text nobody read.
- **Untrusted input.** The contract is passed as data between `<contract>` tags and the model is told to ignore any instructions it contains.

### Run Locally

Backend (Python 3.11):

```bash
conda activate clauseiq
pip install -r requirements-app.txt
cp .env.example .env            # then set ANTHROPIC_API_KEY
uvicorn api.main:app --reload --port 8000
```

Interactive API documentation: http://localhost:8000/docs

Frontend (Node 18, Angular 17):

```bash
cd frontend
npm install
npx ng serve                     # http://localhost:4200, /api is proxied to the backend
```

Demo contracts (public CUAD contracts written to `samples/`):

```bash
python -m scripts.export_cuad_samples --n 3
```

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/contracts` | Upload a contract (multipart `file`). Returns `202` with the contract id |
| `GET` | `/api/v1/contracts` | List contracts, most recent first |
| `GET` | `/api/v1/contracts/{id}` | Status, text and findings (with character offsets) |
| `GET` | `/api/v1/clause-types` | The 41 clause categories and their definitions |
| `GET` | `/api/v1/health` | Liveness check |

### Tests and CI

```bash
python -m pytest                                              # 41 backend tests, never call the paid API
cd frontend && npx ng test --watch=false --browsers=ChromeHeadless
```

`azure-pipelines.yml` runs both test suites and builds the frontend. It is written but has not yet run on Azure DevOps.

### Evaluation

```bash
python -m scripts.evaluate_llm --n 20     # costs money: every contract is one LLM call
```

The script sends CUAD test contracts through the same code path as the app and scores the results against the lawyers' annotations. Every figure comes with its sample size and a 95% Wilson interval. Reports (including the raw per-contract records) are written to `reports/`.

**Scoring rules, fixed before any LLM run** ([scripts/eval_metrics.py](scripts/eval_metrics.py)):

- An excerpt is **correct** when at least 50% of its characters fall inside a passage the lawyers annotated for the same clause, **mislabelled** when they fall inside another clause's passage, **unsupported** otherwise. CUAD's own scorer uses word-level Jaccard ≥ 0.5 instead; it would penalise our deliberately short, one-sentence excerpts taken from long annotated paragraphs.
- A clause counts as **found** only with at least one correct excerpt. Flagging a clause with the wrong passage counts as both a false alarm and a miss.
- Robustness: precision and recall are also reported at thresholds 25% and 75%.
- Rejected excerpts (not found verbatim, never shown) are reported separately: they measure how often the model tried to quote text that is not in the contract.

**The measurement was validated before measuring the model**, with the lawyers' own annotations replayed as a perfect extractor (2026-10-11):

| Check | Result | Command |
|---|---|---|
| Verifier on every annotated excerpt of the 102 test contracts | 2,638 / 2,643 located; the 5 misses are shorter than 3 characters (rejected by design) | `python -m scripts.check_verifier` |
| Same, annotation whose first occurrence is elsewhere (repeated text) | 47 (1.8%) | `python -m scripts.check_verifier` |
| Perfect extractor, 20 contracts (seed 0), 232 annotated clauses | precision 1.0 [0.984, 1.0], recall 1.0 [0.984, 1.0] at every threshold; 3/510 excerpts rejected (too short) | `python -m scripts.evaluate_llm --extractor gold --n 20` |

This check caught a flaw in the first version of the measure: it judged only the first occurrence of a quote (the one the app highlights), so 7 "Document Name" cases failed because lawyers had annotated a later copy of the title. Excerpts are now judged on their text, across all occurrences.

**LLM results (2026-10-11).** `python -m scripts.evaluate_llm --n 20 --runs 2`: 20 CUAD test contracts (seed 0, 232 annotated clauses), `claude-opus-5-5`, effort `high`, evaluated twice to measure run-to-run variation. Full report: `reports/eval_claude_20261011_1645.json`.

| Metric (pre-registered rule) | Run 1 | Run 2 |
|---|---|---|
| Clause precision | 0.626 [0.569, 0.680] | 0.626 [0.568, 0.680] |
| Clause recall | 0.780 [0.723, 0.829] | 0.772 [0.713, 0.821] |
| F1 | 0.695 | 0.691 |
| Rejected quotes (not found verbatim, never shown) | 0 / 476 (≤ 0.8%) | 0 / 471 (≤ 0.8%) |
| Shown quotes: correct / mislabelled / unsupported | 0.576 / 0.059 / 0.366 | 0.573 / 0.068 / 0.359 |
| Mean per contract | 24.8 s, $0.114 | 25.3 s, $0.116 |

How to read it:

- **The model never quoted text that is not in the contract** (0 of 947 quotes over both runs). The verifier stayed a safety net that this sample never triggered.
- **The two runs agree within 0.01**, well inside the confidence intervals.
- **Most errors come from how short annotations are scored.** For "Agreement Date", lawyers annotated only the date ("May 18, 2010") while the model quoted the sentence containing it; the pre-registered rule counts that quote as wrong (recall 0.05 on 19 contracts). The rule is asymmetric, and the perfect-extractor check could not reveal it because its quotes have exactly the annotations' length.
- **Exploratory, post-hoc analysis** (not the official result): with a symmetric rule (a match when the overlap covers at least half of the shorter of quote and annotation), run 1 gives precision 0.723 [0.669, 0.772], recall 0.901 [0.856, 0.933], F1 0.802, stable across thresholds 0.25 to 0.75. Command: `python -m scripts.rescore_eval reports/eval_claude_20261011_1645.json`. This rule should be fixed in advance for the next evaluation.
- **False alarms are not all errors.** Of the 108 clauses flagged without a correct quote in run 1, 68 concern clauses the lawyers did not annotate in that contract. Spot checks found defensible ones (trademark obligations after expiry, flagged as "Post-Termination Services"); judging them needs a lawyer. The other 40 are annotated clauses quoted at the wrong place.
- **Limits:** 20 contracts; CUAD has been public since 2021 and may be in the model's training data; these are US English contracts, so the figures do not transfer to French contracts. Re-running this harness on contracts annotated by a firm's own lawyers is the real test.

### Known Limitations

| Limitation | Why it is acceptable now | Next step |
|---|---|---|
| Background task runs in the API process; analyses interrupted by a restart are marked failed at startup (single-process assumption) | Single local user | Queue-triggered Azure Function, or a lease per job with several instances |
| SQLite, tables created without migrations | Local prototype | Alembic + Azure SQL |
| No authentication | Local demo tool | Entra ID single sign-on, per-matter access rights |
| Scanned PDFs rejected (no OCR) | Demo uses CUAD text contracts | Azure Document Intelligence |
| Contract text sent to an external API (US) | Demo uses public CUAD contracts only | An EU-hosted model behind the same `ClauseExtractor` interface (Claude on Foundry had no EU data zone as of July 2026), or the self-hosted Legal-BERT engine |
| Repeated text: only the first occurrence is highlighted (the evaluation judges all occurrences) | 1.8% of annotated excerpts are repeated text (measured) | Highlight every occurrence (`QuoteVerifier.locate_all` already exists) |

---

## Legal-BERT Engine (Training Pipeline)

Extractive question answering: given a contract and a clause type, the model predicts the start and end token of the answer span, so it returns exact text, not generated summaries.

Illustration of the input and output format (the model has not been trained, so these values are not real predictions):

```
Input:  contract text + "Does this contract contain a Cap on Liability clause?"
Output: "In no event shall either party's total liability exceed $500,000."
        start_token: 4821  end_token: 4849  confidence: 0.94
```

### Model

- **Backbone:** [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) — 110M parameters, pre-trained on EU legislation, US court cases, and contracts
- **Heads:** Two independent linear layers (768 → 1) for start and end token classification
- **Task:** Extractive QA — predict the (start, end) token span of the answer

```
Contract text
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
Multi-window aggregation (pick highest confidence window per clause)   [not built yet]
```

### Training

- **Dataset:** CUAD — 907,520 tokenized windows (after sliding window expansion)
- **Class imbalance:** 1.4% positive (clause present) / 98.6% negative — handled with WeightedRandomSampler
- **Phase 1 (1 epoch):** Freeze backbone, train only span heads (~1,500 params)
- **Phase 2 (2 epochs):** Unfreeze top 4 BERT layers, discriminative LRs (head: 3e-4, backbone: 3e-5)
- **Schedule:** Linear warmup (6%) + linear decay
- **Gradient clipping:** max norm 1.0

### Quickstart

Requires `requirements.txt` (PyTorch, Transformers...) and a GPU for training.

```bash
python -m scripts.download_cuad      # 408 training contracts + 22,450 labeled QA pairs (~18MB) into ml/data/raw/
python -m scripts.preprocess_cuad    # sliding-window features into ml/data/processed/ (~4GB, 20-40 min)
python -m ml.training.train          # GPU required; never run to completion, duration not measured
```

Custom options:
```bash
python -m ml.training.train --epochs 3 --batch-size 16 --freeze-epochs 1
```

> **Note:** Free Colab has 12GB RAM. The full dataset (4GB JSONL → ~12GB in RAM as Python objects) causes OOM crashes. Use a subsampled dataset (~113K examples) or upgrade to Colab Pro.

---

## Project Status

### Done
- [x] Contract review app: FastAPI backend, Angular frontend, LLM extraction engine with verbatim verification
- [x] Backend and frontend tests; Azure DevOps pipeline definition
- [x] Evaluation script against CUAD lawyer annotations (measurement validated)
- [x] LLM evaluation on 20 CUAD test contracts, two runs (2026-10-11, see Evaluation)
- [x] Legal-BERT data pipeline, model architecture and two-phase training loop

### Not Done
- [ ] Legal-BERT training run (weights not saved yet) and its inference engine
- [ ] Deployment (Docker, Azure), authentication, OCR
- [ ] Portfolio queries ("all contracts without a liability cap")

---

## Repository Structure

```
clauseiq/
├── core/                         # Business core: no framework, database or model provider
│   ├── clause_types.py           # The 41 CUAD categories (single source of truth)
│   ├── extraction.py             # ClauseExtractor interface (the port)
│   ├── chunking.py               # Overlapping parts for long contracts (decorator)
│   ├── verification.py           # QuoteVerifier: verbatim matching → character offsets
│   ├── analysis.py               # extraction + verification
│   └── text_extraction.py        # PDF / TXT → text
├── engines/
│   └── claude.py                 # ClaudeExtractor: Anthropic API adapter (prompt, schema)
├── api/                          # FastAPI adapter
│   ├── main.py                   # App entry point
│   ├── v1/routers/contracts.py   # Endpoints
│   ├── jobs.py                   # Background analysis job
│   ├── schemas/                  # HTTP response models
│   └── dependencies/             # Dependency providers (DB, extractor)
├── data/db/                      # SQLAlchemy models and session factory
├── frontend/                     # Angular 17 app
├── ml/                           # Legal-BERT engine (data, model, training)
├── scripts/                      # CUAD download/preprocessing, evaluation, demo samples
├── tests/                        # pytest
├── configs/config.py             # Settings (pydantic-settings, read from .env)
├── azure-pipelines.yml
├── requirements-app.txt          # App dependencies
└── requirements.txt              # Legal-BERT training dependencies
```

---

## Dataset

This project uses the **CUAD (Contract Understanding Atticus Dataset)**:

> Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", 2021.
> [https://arxiv.org/abs/2103.06268](https://arxiv.org/abs/2103.06268)

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## License

MIT
