# Recruitment AI — Automated CV Evaluation System

An AI-powered recruitment tool: automatically parses CVs and Job Descriptions (JDs), matches candidates against requirements, and produces a score plus a detailed explanation to help HR teams screen candidates faster.

Graduation internship project.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Data Generation &amp; Model Training Pipeline](#data-generation--model-training-pipeline)
- [Why a Trained Model Can Outperform the Hardcoded Formula](#why-a-trained-model-can-outperform-the-hardcoded-formula)
- [Roadmap](#roadmap)

---

## Features

- **Automatic CV/JD extraction** via LLM — parses CVs (PDF/DOCX) and JDs into structured data (skills, experience, education, projects...).
- **Two-layer scoring**:
  1. *Rule-based matching*: skill matching via embedding similarity, experience/education scores via a weighted formula.
  2. *LLM scoring*: the LLM reviews the rule-based result, refines the final score, and generates a natural-language explanation (strengths, gaps).
- **Single CV-JD evaluation** (`/api/v1/evaluate`) and **ranking multiple CVs against one JD** (`/api/v1/rank`).
- **Multi-provider support** for LLM/embeddings (Anthropic, OpenAI, open-source models via Ollama/sentence-transformers) — switch via environment variables, no code changes needed.
- **Simple frontend** (plain HTML/CSS/JS) to demo uploading CVs/JDs and viewing results.

## System Architecture

```
                    ┌──────────────┐
   CV (PDF/DOCX) ──▶│ File          │
   JD (text/file) ─▶│ Extraction    │
                    └──────┬───────┘
                           ▼
                 ┌───────────────────┐
                 │ Parsing Service    │  (LLM structured output)
                 │ CV → ParsedCV      │
                 │ JD → ParsedJD      │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Matching Service   │  (embedding similarity +
                 │ → rule_based_score │   weighted formula)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Scoring Service    │  (LLM refines score +
                 │ → final_score      │   explains strengths/gaps)
                 └─────────┬─────────┘
                           ▼
                   CandidateEvaluation
                (returned via API / shown in FE)
```

The whole flow is orchestrated by `RecruitmentPipeline` (`app/services/pipeline.py`).

## Tech Stack

| Component          | Technology                                                                          |
| ------------------ | ----------------------------------------------------------------------------------- |
| Backend            | FastAPI, Pydantic v2                                                                |
| LLM orchestration  | LangChain (langchain-core, langchain-anthropic, langchain-openai, langchain-ollama) |
| LLM provider       | Anthropic Claude (default), OpenAI, or open models via Ollama                       |
| Embedding          | intfloat/multilingual-e5-large (default, open-source) or Voyage AI / OpenAI         |
| CV file reading    | pypdf, docx2txt                                                                     |
| Database (planned) | PostgreSQL + pgvector (via SQLAlchemy)                                              |
| Frontend           | Plain HTML/CSS/JS                                                                   |
| Testing            | pytest                                                                              |

## Installation

**Requirements:** Python 3.11+

```bash
git clone <repo-url>
cd thuc_tap_tot_nghiep

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

> If using `EMBEDDING_PROVIDER=open_source`, the first run will automatically download the `intfloat/multilingual-e5-large` model (~1.1GB) — a stable internet connection is required.

## Environment Configuration

Copy the sample file and fill in real values:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

| Variable                  | Description                                                | Default                            |
| ------------------------- | ---------------------------------------------------------- | ---------------------------------- |
| `LLM_PROVIDER`          | `anthropic` / `openai` / `open_source`               | `anthropic`                      |
| `LLM_MODEL_NAME`        | LLM model name                                             | `claude-sonnet-5`                |
| `LLM_TEMPERATURE`       | LLM randomness (0 = most deterministic)                    | `0.0`                            |
| `ANTHROPIC_API_KEY`     | Anthropic API key (required if`LLM_PROVIDER=anthropic`)  | —                                 |
| `ANTHROPIC_BASE_URL`    | Custom endpoint (if using a proxy)                         | `https://api.anthropic.com`      |
| `EMBEDDING_PROVIDER`    | `voyage` / `openai` / `open_source`                  | `open_source`                    |
| `EMBEDDING_MODEL_NAME`  | Embedding model name                                       | `intfloat/multilingual-e5-large` |
| `SKILL_MATCH_THRESHOLD` | Cosine similarity threshold to consider two skills a match | `0.85`                           |
| `MAX_INPUT_TOKENS`      | Input token limit for the LLM                              | `4000`                           |

> ⚠️ **Never commit a real `.env` file to git.** It is already listed in `.gitignore`. If an API key was ever committed by mistake, **revoke/rotate that key immediately** in the Anthropic Console, purge it from git history (`git filter-repo` or BFG Repo-Cleaner), then push.

### Notes on `.env` for first-time setup (for people cloning the repo)

- Only `.env.example` (a template with no real secrets) is committed. You **must create your own** `.env` locally via `copy .env.example .env` — without it, the app falls back to code defaults and will return a 401 error when calling the LLM due to a missing `ANTHROPIC_API_KEY`.
- Get an API key at [console.anthropic.com](https://console.anthropic.com) → **API Keys** → create a new key, paste it into the `ANTHROPIC_API_KEY=` line in `.env`.
- `.env` is already in `.gitignore`, so `git status` should **not** show it. If it still shows up, it was likely added/committed before `.gitignore` existed. Fix:
  ```bash
  git rm --cached .env
  git commit -m "Remove .env from tracking"
  ```

  (`--cached` only untracks it from git — the local file is untouched and the app keeps working.)
- Restart `uvicorn` after changing any value in `.env` — the server does not hot-reload environment variables.
- Never share `.env` in plaintext over chat/email, even within the team. If a teammate needs a key, share it through a secure channel — not pasted into a group chat or bundled with source code submitted to instructors.

## Running the App

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000/api/v1`
- Frontend demo: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- API docs (Swagger UI, auto-generated by FastAPI): `http://localhost:8000/docs`

## API Endpoints

### `POST /api/v1/evaluate`

Evaluate one CV against one JD.

**Form-data:**

| Field       | Type            | Required                         |
| ----------- | --------------- | -------------------------------- |
| `cv_file` | file (PDF/DOCX) | ✔                               |
| `jd_text` | text            | ✔ (if`jd_file` not provided)  |
| `jd_file` | file            | ✔ (if`jd_text` not provided)  |
| `jd_id`   | text            | optional, defaults to`"JD-01"` |

**Response:**

```json
{
  "evaluation": {
    "cv_id": "candidate.pdf",
    "jd_id": "JD-01",
    "final_score": 82.5,
    "strengths": ["..."],
    "gaps": ["..."],
    "explanation": "..."
  },
  "rule_based_score": 78.0
}
```

### `POST /api/v1/rank`

Evaluate multiple CVs against one JD at once, returning a ranked list.

**Form-data:** same as `/evaluate` but `cv_files` accepts multiple files.

**Response:**

```json
{
  "jd_id": "JD-01",
  "total_candidates": 12,
  "results": [
    { "rank": 1, "evaluation": { ... }, "rule_based_score": 85.0 },
    { "rank": 2, "evaluation": { ... }, "rule_based_score": 79.0 }
  ]
}
```

## Project Structure

```
.
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   └── routers.py          # REST endpoints (/evaluate, /rank)
│   ├── core/
│   │   ├── settings.py         # Reads configuration from environment variables
│   │   ├── model_config.py     # Builds the LLM client based on provider
│   │   └── embedding_config.py # Builds the embedding client based on provider
│   ├── services/
│   │   ├── parsing_service.py  # Parses CV/JD → structured data (LLM)
│   │   ├── matching_service.py # Rule-based matching (embedding similarity)
│   │   ├── scoring_service.py  # LLM refines score + explanation
│   │   └── pipeline.py         # Orchestrates the whole flow
│   ├── schemas/
│   │   └── models.py           # Pydantic models (ParsedCV, ParsedJD, CandidateEvaluation...)
│   └── utils/
│       └── file_extraction.py  # Extracts text from PDF/DOCX
├── frontend/                   # Demo UI (plain HTML/CSS/JS)
├── tests/                      # Unit tests (pytest)
├── requirements.txt
├── .env.example
└── README.md
```

## Data Generation & Model Training Pipeline

To move from a hardcoded scoring formula to a model that learns from data, the project includes an additional set of scripts for dataset generation and training (outside `app/`, not part of the main API):

| Script                       | Purpose                                                                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_synthetic_jd.py` | Generates diverse synthetic Vietnamese JDs (role/seniority/domain) with Claude, used when real JDs can't be crawled (ITviec/TopCV block crawling via robots.txt).      |
| `generate_synthetic_cv.py` | Generates CVs paired with each JD at varying fit levels (`good_fit`/`partial_fit`/`poor_fit`) so the dataset carries a distinguishable signal.                   |
| `run_distillation.py`      | Runs each CV-JD pair through the actual`RecruitmentPipeline` to obtain `final_score` (label) and per-criterion scores (features) → exports `training_data.csv`. |
| `train_ranker.py`          | Trains a lightweight ranking model (XGBoost) on`training_data.csv`, replacing part of the LLM calls when a fast bulk-screening step is needed.                       |

> Generated data (`*.jsonl`, `training_data.csv`) contains **fictional** information (fake names/emails/phone numbers), so it's safe to store, but it's still recommended not to commit large dataset files directly to git (see `.gitignore`) — store them elsewhere (Google Drive, internal school storage...) and document how to regenerate them here instead.

## Why a Trained Model Can Outperform the Hardcoded Formula

The current rule-based formula (`0.5 × skills + 0.3 × experience + 0.2 × education`) has real limits that a learned model (XGBoost) can overcome:

- **No interaction between factors** — the formula treats skills/experience/education as fully independent and just adds them up, while in reality low experience should heavily drag down the overall score regardless of how high skills are (a conditional/non-linear rule, which decision trees capture naturally).
- **Fixed weights, not learned from data** — `0.5/0.3/0.2` is a guess applied identically to every role and seniority level, even though skills vs. experience should plausibly matter differently for a Fresher vs. a Senior position.
- **Distilled from LLM judgment, which is more nuanced** — the training label (`final_score`) comes from the LLM's own refined evaluation, not a simple sum, so the model learns to approximate that more sophisticated judgment instead of a flat linear formula.
- **Honest caveat** — this advantage only shows up with enough diverse training data; with a small dataset (~200 rows), the model may only match or slightly beat the baseline, which is itself a valid, reportable finding ("more data is needed to fully exploit the model's potential").

## Roadmap

- [ ] Asynchronous processing (background task/queue) for `/rank` when handling large numbers of CVs.
- [ ] Persist evaluation results to PostgreSQL + pgvector (embedding cache, evaluation history).
- [ ] Train a lightweight ranker (XGBoost/LightGBM) as a fast pre-filter before detailed LLM scoring.
- [ ] Build an evaluation benchmark (manually labeled CV-JD pairs) to measure system accuracy.
- [ ] Redact demographic information (name, gender, age) from LLM input to reduce bias.
- [ ] More visual explanations (highlighting the exact CV sentence/section matching each JD requirement).

---

## License

Graduation internship project — for academic purposes.
