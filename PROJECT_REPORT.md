# ClauseIQ — Project Report

> Internal engineering document (not the README). Single source of truth for what was built,
> why, how the pieces connect, and what remains. Update after every significant change.
> Last full review: 2026-10-11.

---

## 1. Project Overview

ClauseIQ is a contract review tool for lawyers. A user uploads a commercial contract (PDF or
TXT); for each of the 41 clause categories of CUAD (Contract Understanding Atticus Dataset: 510
commercial contracts annotated by lawyers) the app says whether the clause is present and
highlights the exact sentences in the contract. Every excerpt returned by the extraction engine
is verified verbatim against the contract text before display, so the tool never shows a
sentence the contract does not contain.

Two extraction engines behind one interface (`core.extraction.ClauseExtractor`):
- **LLM engine** (Claude via the Anthropic API, structured outputs): built, used by the app, not yet run for real (API credit exhausted on 2026-10-10).
- **Legal-BERT engine** (fine-tuned extractive question answering): training pipeline built in March 2026, model never trained, no inference code.

Context: portfolio project of Mehdi Bourezza, built with Python, TypeScript/Angular and SQL,
with Azure and Azure DevOps as the deployment target.

**Stack:** Python 3.11, FastAPI 0.143, SQLAlchemy 2.1 (SQLite), pypdf 6, anthropic SDK 1.13,
Angular 17.3 / TypeScript 5.4 (Node 18.17), pytest, Karma/Jasmine. Legal-BERT side: PyTorch,
Transformers.

**Repository:** https://github.com/mehdibourezza/clauseIQ. Working branch:
`feature/review-app` (uncommitted as of 2026-10-11).

---

## 2. Architecture Map

```
                          [Browser: Angular 17 SPA]  frontend/            BUILT
                                   │  /api/v1/* (dev proxy → 127.0.0.1:8000)
                                   ▼
┌──────────────────────── [FastAPI app] api/ ─────────────────────────────┐ BUILT
│ routers/contracts.py ── upload → 202 + id ── BackgroundTasks ──┐        │
│        ▲ GET status/detail (polled every 2 s)                  ▼        │
│        │                                               jobs.run_analysis │
└────────┼───────────────────────────────────────────────────────┼────────┘
         │                                                       ▼
         │                         ┌──────── core/ (no framework/DB/provider) ┐ BUILT
         │                         │ text_extraction → analysis.analyze_contract
         │                         │      ├─ ClauseExtractor (port)            │
         │                         │      │    └─ ChunkingExtractor (decorator: long contracts → parts)
         │                         │      └─ QuoteVerifier (verbatim check, full text)
         │                         └──────────────────────────────────────────┘
         │                         engines/claude.py: ClaudeExtractor ─────► Anthropic API (external)   BUILT
         │                         Legal-BERT extractor                                                PLANNED (ml/ has training only)
         ▼
[SQLite: contracts, clause_findings]  data/db/                                BUILT
                                                     
[scripts/evaluate_llm.py] ── same job path ── compares with CUAD annotations  BUILT (never run with the LLM)
[ml/] Legal-BERT data prep + training loop                                    BUILT, never trained
[azure-pipelines.yml] tests + frontend build                                  WRITTEN, never executed
Deployment (cloud hosting, auth, queue)                                       PLANNED
```

---

## 3. Component Registry

#### Clause catalogue (`core/clause_types.py`)
- **Purpose:** The 41 CUAD categories (name + lawyer-facing definition), single source of truth.
- **Inputs/Outputs:** static data → used by the prompt, the output schema enum and `GET /api/v1/clause-types`.
- **Key decisions:** generated once from `ml/data/raw/cuad_extracted/test.json` questions, verbatim.
- **Status:** Built. **Last updated:** 2026-10-10

#### Text extraction (`core/text_extraction.py`)
- **Purpose:** Uploaded bytes → contract text.
- **Inputs:** filename + bytes (.pdf via pypdf, .txt as UTF-8). **Outputs:** str, or `DocumentError` (user-facing message).
- **Key decisions:** scanned PDFs (no text layer) rejected explicitly instead of analysing an empty contract.
- **Status:** Built. **Last updated:** 2026-10-10

#### Extraction port (`core/extraction.py`)
- **Purpose:** The contract every engine fulfils: `ClauseExtractor.extract(text) -> ExtractionResult` (dict name → excerpts, model, token counts), or `ExtractionError` (user-facing message).
- **Key decisions:** a Protocol (port), so engines are swappable; `tests/test_architecture.py` fails if any `core/` module imports FastAPI, SQLAlchemy, anthropic or another app package.
- **Status:** Built. **Last updated:** 2026-10-11 (Claude adapter moved out to `engines/`)

#### Long-contract chunking (`core/chunking.py`)
- **Purpose:** `ChunkingExtractor` decorates any `ClauseExtractor`: contracts above `LLM_MAX_CHARS_PER_CALL` (default 400,000 chars) are extracted in parts overlapping by `LLM_CHUNK_OVERLAP_CHARS` (default 4,000), excerpts merged and de-duplicated, tokens summed.
- **Key decisions:** overlap > longest CUAD annotated clause (3,884 chars) so every clause appears whole in one part (proof in the module docstring); cut points snap left to a line break/space; verification stays on the full text (offsets unchanged); one failed part fails the whole analysis (no misleading "absent"); parts processed sequentially. Longest CUAD contract (338,211 chars): 1 part at default, 4 parts at 100,000.
- **Status:** Built, 7 tests (`tests/test_chunking.py`). **Last updated:** 2026-10-11

#### Claude engine (`engines/claude.py`)
- **Purpose:** One Anthropic API call per contract (or per part) returns verbatim excerpts for all 41 categories.
- **Key decisions:** `claude-opus-5-5`, effort `high`, streaming, structured outputs (Pydantic schema, `clause_type` constrained to a 41-value enum), server-side `fallbacks="default"` (beta `server-side-fallback-2026-07-01`); SDK errors translated to `ExtractionError`; contract wrapped in `<contract>` tags with an instruction to ignore embedded instructions (prompt-injection mitigation).
- **Status:** Built, unit-tested with a stub client (`tests/test_claude_engine.py`); never executed against the real API. **Last updated:** 2026-10-11

#### Quote verifier (`core/verification.py`)
- **Purpose:** Prove each excerpt exists in the contract; return offsets in the original text.
- **Inputs:** contract text, quote. **Outputs:** `Span(start, end)` or None.
- **Key decisions:** normalise whitespace runs and curly quotes/dashes on both sides (1:1 char map kept to recover original offsets), then exact, case-sensitive match; no fuzzy matching (would accept reworded quotes); quotes < 3 chars rejected; first occurrence wins.
- **Status:** Built. **Last updated:** 2026-10-10

#### Analysis (`core/analysis.py`)
- **Purpose:** extraction + verification → `AnalysisResult` (findings with optional span, model, tokens).
- **Key decisions:** exact duplicate quotes dropped; unverified quotes kept (span None) for measurement.
- **Status:** Built. **Last updated:** 2026-10-10

#### Persistence (`data/db/models.py`, `data/db/session.py`)
- **Purpose:** Tables `contracts` (id UUID, filename, text, status processing/done/failed, error, model, input/output tokens, created/completed_at) and `clause_findings` (contract FK ON DELETE CASCADE, clause_type, quote, start/end_char, verified).
- **Key decisions:** SQLite + `create_all` (no migrations); indexes on created_at, contract_id, clause_type.
- **Status:** Built. **Last updated:** 2026-10-10

#### API (`api/main.py`, `api/v1/routers/contracts.py`, `api/jobs.py`, `api/schemas/contracts.py`, `api/dependencies/providers.py`)
- **Purpose:** `POST /api/v1/contracts` (202), `GET /api/v1/contracts`, `GET /api/v1/contracts/{id}`, `GET /api/v1/clause-types`, `GET /api/v1/health`.
- **Key decisions:** 202 + polling; sync endpoints (thread pool); background job always records failures (`status=failed`, message) so nothing stays "processing"; dependency providers overridable in tests; 10 MB upload cap.
- **Status:** Built. **Last updated:** 2026-10-10

#### Frontend (`frontend/src/app/`)
- **Purpose:** Upload + contract list (`contract-list/`); detail with 41-clause list and highlighted text (`contract-detail/`); `highlight.ts` cuts text at verified offsets (overlaps supported).
- **Key decisions:** standalone components, signals, route param bound as signal input, polling with `timer` + `exhaustMap` + `takeWhile`; French UI; dev proxy targets `127.0.0.1` (Node resolves `localhost` to IPv6).
- **Status:** Built. **Last updated:** 2026-10-10

#### Evaluation (`scripts/evaluate_llm.py`)
- **Purpose:** Run N CUAD test contracts through `jobs.run_analysis` and score against lawyer annotations: presence precision/recall/F1 (overall + per type), grounding rate, on-target rate, cost, latency; JSON report in `reports/`. `--save-to-db` keeps results in the app DB (offline demo).
- **Status:** Built; validated with a gold extractor; never run with the LLM. **Last updated:** 2026-10-10

#### Scoring rules (`scripts/eval_metrics.py`) and verifier check (`scripts/check_verifier.py`)
- **Purpose:** Pre-registered matching rules (excerpt correct if ≥ 50% of its characters lie in a gold span of the same type, judged over all occurrences of its text; strict clause rule: wrong passage = false alarm + miss), Wilson 95% intervals, sensitivity thresholds 0.25/0.5/0.75. `check_verifier` reproduces the verifier figures on all CUAD test annotations.
- **Key decisions:** rule chosen over CUAD's word-level Jaccard ≥ 0.5 because our excerpts are one sentence while annotations can be paragraphs; rules fixed before the first LLM run.
- **Status:** Built, unit-tested (`tests/test_eval_metrics.py`). **Last updated:** 2026-10-11

#### Demo samples (`scripts/export_cuad_samples.py`)
- **Purpose:** Writes N public CUAD test contracts to `samples/*.txt` (byte-identical, `newline=""`).
- **Status:** Built. **Last updated:** 2026-10-10

#### Legal-BERT training pipeline (`ml/`, `scripts/download_cuad.py`, `scripts/preprocess_cuad.py`, `scripts/colab_train.py`)
- **Purpose:** Fine-tune `nlpaueb/legal-bert-base-uncased` with start/end span heads on CUAD (512-token windows, stride 128, WeightedRandomSampler, two-phase unfreezing).
- **Status:** Built (2026-03), never trained; no inference engine; `scripts/colab_train.py` untracked.

#### CI (`azure-pipelines.yml`)
- **Purpose:** Backend pytest + frontend Karma tests + `ng build` + artifact publish.
- **Status:** Written, YAML validated, never executed (no Azure DevOps organisation).

---

## 4. Evaluation & Metrics

| Date | What | Result | Notes |
|---|---|---|---|
| 2026-10-10 | Backend tests (`python -m pytest`) | 20/20 pass | no API calls (fake extractor, stub client) |
| 2026-10-10 | Frontend tests (Karma, ChromeHeadless) | 3/3 pass | `highlight.spec.ts`; `ng build` OK |
| 2026-10-10 | Verifier on all CUAD test annotations (102 contracts) | 2,638 / 2,643 located | 5 misses are < 3 chars (below MIN_QUOTE_CHARS) |
| 2026-10-10 | Same, annotations whose first occurrence is elsewhere | 47 (1.8%) | repeated text; caps on-target rate |
| 2026-10-10 | Eval harness, gold extractor, 8 contracts (seed 0) | presence P=R=F1=1.0; grounding 0.994; on-target 0.972 | validates the measurement, not the model |
| 2026-10-10 | LLM smoke test (1 contract) | not run | HTTP 400 "credit balance is too low" |
| 2026-10-11 | Backend tests | 30/30 pass | + scoring-rule tests, `locate_all` test |
| 2026-10-11 | `python -m scripts.check_verifier` (102 contracts, 2,643 annotations) | 2,638 located; 5 misses all < 3 chars; 47 (1.8%) first occurrence elsewhere | reproducible command (2026-10-10 figures came from ad-hoc scripts) |
| 2026-10-11 | Gold extractor v1 of the measure (first occurrence only), 20 contracts seed 0 | P = R = 0.97 | 7 "Document Name" failures: lawyers annotated a later copy of the title → measure fixed |
| 2026-10-11 | `python -m scripts.evaluate_llm --extractor gold --n 20` (measure v2: all occurrences) | P = 1.0 [0.984, 1.0], R = 1.0 [0.984, 1.0] on 232 annotated clauses, same at thresholds 0.25/0.5/0.75; 3/510 excerpts rejected (< 3 chars) | measurement validated; rules frozen before any LLM run |
| 2026-10-11 | Backend tests | 38/38 pass | + architecture rule (core imports), + 7 chunking tests |
| 2026-10-11 | API key check (Max plan included credits) | auth OK, `claude-opus-5-5` reachable; 1-word call 18 in / 5 out tokens, ~$0.0002 | |
| 2026-10-11 | First real analysis, end to end through the UI (Loha supply agreement, 11,475 chars) | 202 in 0.17 s; done in 16 s; 7,315 in / 2,017 out tokens ≈ $0.070; 10/41 clause types, 14 excerpts, 14 verified, 0 rejected | single contract: says nothing about quality yet |
| 2026-10-11 | `python -m scripts.evaluate_llm --n 20 --runs 2` (seed 0, 232 annotated clauses), pre-registered rule | run 1: P 0.626 [0.569, 0.680], R 0.780 [0.723, 0.829], F1 0.695; run 2: P 0.626, R 0.772, F1 0.691 | report `reports/eval_claude_20261011_1645.json`; runs agree within 0.01 |
| 2026-10-11 | Same, rejected quotes | 0/476 and 0/471 (≤ 0.8% each, Wilson 95%) | the model never quoted text absent from the contract |
| 2026-10-11 | Same, shown quotes correct / mislabelled / unsupported | run 1: 0.576 / 0.059 / 0.366 | 108 false alarms in run 1 = 68 on clauses not annotated in that contract + 40 annotated but wrong passage |
| 2026-10-11 | Same, cost and latency | 24.8 s and $0.114 per contract (run 1); $2.27 + $2.32 for both runs; 10.7 to 74.9 s | |
| 2026-10-11 | Recall per type (pre-registered rule) | Agreement Date 0.05 (n=19), Parties 0.60 (20), Effective Date 0.69 (13), Document Name 0.80 (20); Governing Law, Anti-Assignment, License Grant, Expiration Date 1.0 | short annotations (a date) inside longer correct quotes are scored wrong |
| 2026-10-11 | `python -m scripts.rescore_eval reports/eval_claude_20261011_1645.json`, symmetric rule (EXPLORATORY, post-hoc) | run 1: P 0.723 [0.669, 0.772], R 0.901 [0.856, 0.933], F1 0.802; flat across thresholds 0.25-0.75 | not the official figure; pre-register this rule next time |
| — | Legal-BERT | never trained | training metric bug: span F1 skips all negative windows (false positives never counted) |

---

## 5. Technical Decisions Log

```
[2026-03-10] Extractive QA with Legal-BERT (training pipeline)
  CHOSE: Legal-BERT + start/end span heads, 512-token windows with 128 stride, WeightedRandomSampler, two-phase unfreezing
  OVER: plain BERT, loss re-weighting
  BECAUSE: domain pre-training; 1.4% positive windows; see reports/day1/day2 session reports

[2026-10-10] LLM engine first, Legal-BERT later
  CHOSE: Claude with structured outputs + local verbatim verification
  OVER: training Legal-BERT now; Anthropic native Citations
  BECAUSE: no trained weights and 1-3 days available; native Citations are incompatible with structured outputs

[2026-10-10] One LLM call for all 41 categories
  CHOSE: single call  OVER: 41 calls
  BECAUSE: contract tokens paid once (~30x cheaper on input); recall on rare types to be measured

[2026-10-10] Strict verbatim verification
  CHOSE: whitespace/typography normalisation + exact match  OVER: fuzzy matching
  BECAUSE: fuzzy matching would accept reworded quotes, the failure we want to catch

[2026-10-10] 202 Accepted + polling with in-process BackgroundTasks
  CHOSE: FastAPI BackgroundTasks  OVER: Celery/Redis, synchronous request
  BECAUSE: analysis takes minutes; no Docker locally; job lost on restart accepted for local use

[2026-10-10] FastAPI locally, core independent of framework (ports and adapters)
  CHOSE: FastAPI  OVER: Azure Functions locally
  BECAUSE: known framework, no Core Tools/storage emulator risk; core portable to Functions

[2026-10-10] SQLite via SQLAlchemy  OVER: PostgreSQL  BECAUSE: zero install; URL swap later

[2026-10-10] Angular 17 on Node 18.17  OVER: Node 22 + latest Angular  BECAUSE: user choice, no install risk

[2026-10-10] Separate requirements-app.txt  BECAUSE: app and CI must not install torch/transformers

[2026-10-11] Roadmap after competitive analysis + deployment research + LLM council (5 advisors, peer review, chairman)
  CHOSE: measure first (pre-registered metrics, then LLM eval),
         code-ownership pass, one session with a legal user, Azure DevOps pipeline green on a self-hosted agent,
         deployment only after hardening (shared access code + daily cap, durable job row, idempotent upload, Postgres)
  OVER: positioning against Harvey/Legora; building French/tax gold sets without lawyers; full Azure deployment at this stage
  BECAUSE: trust before features; unmeasured claims collapse under questioning;
           every new AI-written feature widens the "defend every line" gap; a public URL without auth is a cost liability.
  Wording: "exact quote, found in your contract" instead of "verified" (a verbatim match proves existence,
           not the right label nor completeness).

[2026-10-11] Pre-registered evaluation rules
  CHOSE: excerpt correct if >= 50% of its characters fall in a same-type gold span, judged on its text
         (all occurrences); strict clause rule; Wilson intervals; sensitivity at 0.25/0.75
  OVER: any-overlap rule (too lenient), CUAD word-Jaccard >= 0.5 (penalises one-sentence excerpts),
        first-occurrence-only judging (failed 7 "Document Name" cases with the lawyers' own annotations)
  BECAUSE: rules chosen after seeing model results could flatter the model; validated with a perfect extractor first

[2026-10-11] Asymmetric rule found AFTER the first LLM run: keep it official, add a labelled post-hoc analysis
  CHOSE: report the pre-registered rule as the result; add RULE_SYMMETRIC (overlap coefficient) as exploratory,
         recomputed from the saved report by scripts/rescore_eval.py (no new API call)
  OVER: silently switching to the rule that scores better
  BECAUSE: switching rules after seeing results is the "garden of forking paths"; the flaw (short annotations such as
           dates inside longer correct quotes) was invisible to the perfect-extractor check, whose quotes equal the annotations

[2026-10-11] Claude adapter moved out of core/
  CHOSE: core/extraction.py = port only; engines/claude.py = Anthropic adapter; architecture test on core imports
  OVER: port and adapter in the same core module
  BECAUSE: "the core knows neither FastAPI nor Claude" must be literally true and checked automatically

[2026-10-11] Automatic chunking of long contracts (decorator)
  CHOSE: ChunkingExtractor wrapping any engine; 400,000-char threshold; 4,000-char overlap; fail whole analysis on a failed part
  OVER: chunking inside ClaudeExtractor; failing cleanly without chunking; partial results
  BECAUSE: engine-agnostic; threshold above every CUAD contract keeps evaluation unchanged; overlap above the longest
           CUAD clause guarantees each clause is whole in one part; a partial result would mislead the lawyer
```

---

## 6. Tech Debt & Known Issues

Findings of the 2026-10-11 architecture review (system-architect protocol) included.

1. **No authentication, no rate limit, no quota.** Fine locally. **Blocks any public deployment**: anyone could trigger paid LLM calls (cost attack) and read every contract.
2. **POST /contracts is not idempotent.** A double click or retry creates a second contract and a second paid LLM call. Fix: content hash (SHA-256 of the text) + reuse of an existing analysis.
3. **In-process background tasks.** A restart mid-analysis loses the job; since 2026-10-11 the row is marked failed at startup (`api.jobs.fail_interrupted_analyses`) instead of staying `processing` forever, but the user must upload again. Assumes a single API process: several instances need a lease per job. Breaks on scale-to-zero platforms. Fix: durable queue + worker.
4. **SQLite + `create_all`, no migrations.** Breaks on the first schema change with data to keep. Fix: PostgreSQL + Alembic.
5. **`GET /contracts` unpaginated** and loads every finding of every contract. Fine below ~1,000 contracts. Fix: cursor pagination + aggregate count query.
6. **No delete / soft delete, no ownership column.** Required before multi-user use.
7. **Single LLM provider** (server-side fallback only covers refusals). No timeout budget per analysis beyond SDK defaults.
8. **No observability:** unstructured logs, no metrics (latency, cost, failure rate), no tracing.
9. **Verifier highlights the first occurrence of repeated text** (1.8% of annotated excerpts).
10. **Contract text sent to an external API (Anthropic, US).** Demo with public CUAD contracts only. Real use needs an EU-hosted model behind `ClauseExtractor` or the self-hosted Legal-BERT engine. Correction 2026-10-11: Claude on Microsoft Foundry had **no EU data zone** as of July 2026 (InfoQ), so "Foundry in the firm's tenant" is not by itself an answer to EU residency.
11. **Scanned PDFs unsupported** (no OCR).
12. **Node 18 is end-of-life; Angular 17 out of support.**
13. **Legal-BERT training metric bug:** span F1 ignores negative windows, so false positives are invisible.
14. **CI never executed; no Docker image.**

---

## 7. What's Next

Order recommended on 2026-10-11:

1. **[DONE 2026-10-11] Pre-register the evaluation** (no cost): fix the excerpt↔gold matching rule, define hit / mislabel / unsupported / miss, add per-clause support counts and confidence intervals, plan two runs for variance. Must be done BEFORE seeing LLM results.
2. [DONE 2026-10-11] API credits (Max plan) → smoke test → evaluation (20 contracts, 2 runs) → metrics recorded here and in README.
3. [DONE] Commit `feature/review-app` (small commits).
4. Code-ownership pass: Mehdi explains every module without the assistant; remove what cannot be explained.
5. One 30-45 min session with a legal user; fix one thing (likely: distinguish "nothing found, check manually" from "quote rejected"; never display "absent").
6. Azure DevOps pipeline green on a self-hosted agent (free; the hosted-job grant can take days; Azure DevOps billing does not accept free-trial subscriptions).
7. Hardening before any public URL: shared access code + daily call cap, durable job row + restart sweeper, idempotent upload (file hash), PostgreSQL + Alembic.
8. Deployment: option B (Render Starter ~$7/mo + Neon free + Cloudflare Pages) or option A (Azure Static Web Apps + Container Apps + Storage Queue + Flex Functions worker + Postgres B1ms, ~$0-1/mo year 1). Requires 7.
9. Next: recall work (W4) on clauses where the eval shows weak recall; Legal-BERT vs LLM on the same harness; French/tax categories only with lawyer-annotated data; Node/Angular upgrade.

---

## 8. Environment & Setup

```bash
# Backend (conda env "clauseiq", Python 3.11)
pip install -r requirements-app.txt
cp .env.example .env                      # set ANTHROPIC_API_KEY
uvicorn api.main:app --reload --port 8000  # docs: http://localhost:8000/docs
python -m pytest                          # 41 tests, no API calls

# Frontend (Node 18, Angular 17)
cd frontend && npm install && npx ng serve            # http://localhost:4200
npx ng test --watch=false --browsers=ChromeHeadless

# Demo data and evaluation
python -m scripts.export_cuad_samples --n 3            # samples/*.txt
python -m scripts.evaluate_llm --n 20                  # paid: one LLM call per contract
python -m scripts.evaluate_llm --n 5 --save-to-db      # offline demo data in clauseiq_review.db

# Legal-BERT (requirements.txt, GPU)
python -m scripts.download_cuad && python -m scripts.preprocess_cuad && python -m ml.training.train
```

Settings (`configs/config.py`, read from `.env`): `ANTHROPIC_API_KEY`, `LLM_MODEL` (default
`claude-opus-5-5`), `LLM_EFFORT` (default `high`), `LLM_MAX_CHARS_PER_CALL` (default 400000;
set 100000 to demonstrate chunking on long CUAD contracts), `LLM_CHUNK_OVERLAP_CHARS` (default
4000), `REVIEW_DB_URL` (default `sqlite:///clauseiq_review.db`).

API credits: Mehdi's Claude Max plan includes monthly API credits ($100 on Max 5x, $200 on
Max 20x, no rollover), claimed on claude.ai › Settings › Billing › "API credits" by linking a
Claude Console organization (one organization per plan, not changeable without support). The API
key in `.env` must belong to that organization.
