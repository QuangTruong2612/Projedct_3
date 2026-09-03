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
- [Using the Ranker as a Cheap Pre-filter](#using-the-ranker-as-a-cheap-pre-filter)
- [Calibrating the Scoring Formulas](#calibrating-the-scoring-formulas)
- [Does the Trained Model Beat the Hardcoded Formula?](#does-the-trained-model-beat-the-hardcoded-formula)
- [Roadmap](#roadmap)

---

## Features

- **Automatic CV/JD extraction** via LLM — parses CVs (PDF/DOCX) and JDs into structured data (skills, experience, education, projects...).
- **Two-layer scoring**:
  1. *Rule-based matching*: skill matching via embedding similarity, experience/education scores via a weighted formula.
  2. *LLM scoring*: the LLM reviews the rule-based result, refines the final score, and generates a natural-language explanation (strengths, gaps).
- **Single CV-JD evaluation** (`/api/v1/evaluate`) and **ranking multiple CVs against one JD** (`/api/v1/rank`).
- **Interview question generation** (`/api/v1/interview-questions`) — turns each detected gap into a targeted question plus what to listen for in the answer.
- **Data-completeness warnings** — a score of 0 because the CV could not be parsed is flagged separately from a score of 0 because the candidate genuinely does not qualify.
- **Model as a cheap pre-filter** — the trained ranker scores every candidate for free, and with `RANKER_PREFILTER_TOP_K` set, the LLM is only asked to write up the top-K. Candidates that don't make the cut are clearly marked `scored_by: "model"` so nobody mistakes a model score for an LLM review.
- **Counterfactual gap analysis** — for every evaluation, which missing skill would raise the score the most, and by exactly how many points. Computed in closed form, so it costs no extra LLM or embedding call.
- **JD quality check** (`/api/v1/check-jd`) — flags job descriptions that will produce bad rankings (19 "required" skills, a *Junior* title asking for 6 years, weights that don't sum to 1).
- **Multi-provider support** for LLM/embeddings (Anthropic, OpenAI, open-source models via Ollama/sentence-transformers) — switch via environment variables, no code changes needed.
- **Frontend** (plain HTML/CSS/JS) with two views:
  - *Ranking list* — candidates ordered by score.
  - *Triage board* — three columns (**Nên phỏng vấn / Cân nhắc thêm / Chưa phù hợp**). The AI files each candidate by score; HR can move anyone to another column and that decision is remembered and always wins over the AI's suggestion. Cards show the per-criterion breakdown, a warning when the CV parsed badly, and the CSV export records the final human decision.

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

| Variable                      | Description                                                                                                        | Default                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| `LLM_PROVIDER`              | `anthropic` / `openai` / `open_source`                                                                       | `anthropic`                      |
| `LLM_MODEL_NAME`            | LLM model name                                                                                                     | `claude-sonnet-5`                |
| `LLM_TEMPERATURE`           | LLM randomness (0 = most deterministic)                                                                            | `0.0`                            |
| `ANTHROPIC_API_KEY`         | Anthropic API key (required if`LLM_PROVIDER=anthropic`)                                                          | —                                 |
| `ANTHROPIC_BASE_URL`        | Custom endpoint (if using a proxy)                                                                                 | `https://api.anthropic.com`      |
| `EMBEDDING_PROVIDER`        | `voyage` / `openai` / `open_source`                                                                          | `open_source`                    |
| `EMBEDDING_MODEL_NAME`      | Embedding model name                                                                                               | `intfloat/multilingual-e5-large` |
| `EMBEDDING_PREFIX`          | `none` / `auto` (adds `query: ` for e5 models) / a literal prefix — see note below                          | `none`                           |
| `SKILL_MATCH_THRESHOLD`     | Similarity above which a skill counts as a*clear* match — **display only**, not used to compute the score | `0.85`                           |
| `SIMILARITY_FLOOR`          | Embedding noise floor; similarity below this is treated as unrelated (calibrated, see below)                       | `0.80`                           |
| `EXPERIENCE_RELEVANCE_BASE` | Fraction of the experience score kept when the candidate's background is entirely unrelated                        | `0.3`                            |
| `MAX_INPUT_TOKENS`          | Input token limit for the LLM                                                                                      | `4000`                           |
| `RANK_MAX_WORKERS`          | How many CVs`/rank` processes in parallel (each worker makes its own LLM calls)                                  | `4`                              |

### A note on `EMBEDDING_PREFIX`

The e5 model family is trained with `query: ` / `passage: ` prefixes, so in
principle they should be applied. Measured on 20 skill pairs (10 synonym pairs
that *should* match, 10 cross-domain pairs that should *not*), adding
`query: ` shifted every similarity up by roughly +0.015 but did **not** improve
separation between the two groups (0.081 → 0.078), and at the current `0.85`
threshold the match decisions were identical. Since enabling it shifts every
score — and would desynchronise `training_data.csv` from the live system — the
default is `none`. Set `EMBEDDING_PREFIX=auto` to experiment.

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

> **Startup takes a few seconds.** The pipeline — including the ~1.1GB embedding
> model — is built **once** during app startup (FastAPI `lifespan`) and reused for
> every request, instead of being rebuilt per request. If the LLM client can't be
> built at startup (e.g. missing `ANTHROPIC_API_KEY`), startup still succeeds and
> the error surfaces as a clear 401 on the first API call.
>
> A consequence: `.env` is read only once, so **restart uvicorn** after changing
> any value there.

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
  "rule_based_score": 78.0,
  "criterion_scores": [
    { "criterion": "skills", "score": 0.81, "detail": "Khớp rõ 14/18 kỹ năng bắt buộc...", "missing_data": false }
  ],
  "completeness": {
    "has_skills": true, "has_experience": true, "has_education": true,
    "score": 1.0, "warnings": []
  },
  "skill_gaps": [
    { "skill": "Kafka", "current_coverage": 0.0, "score_gain": 2.2, "is_required": true },
    { "skill": "Next.js", "current_coverage": 0.35, "score_gain": 1.4, "is_required": false }
  ]
}
```

### `POST /api/v1/check-jd`

Inspect a JD *before* using it to score candidates. A JD listing 19 "required"
skills drives every candidate's coverage down, so every application scores badly
— the fault is in the JD, not the applicants.

**Form-data:** `jd_text` or `jd_file`.

**Response:**

```json
{
  "job_title": "Senior Fullstack Developer",
  "n_require_skills": 19,
  "n_preferred_skills": 0,
  "min_experience_years": 4.0,
  "is_healthy": false,
  "warnings": [
    {
      "code": "too_many_required_skills",
      "severity": "high",
      "message": "Có tới 19 kỹ năng bắt buộc. Gần như không ứng viên nào đáp ứng đủ...",
      "suggestion": "Giữ lại 8-12 kỹ năng thực sự bắt buộc, chuyển phần còn lại sang mục ưu tiên."
    }
  ]
}
```

### `POST /api/v1/interview-questions`

Evaluate one CV, then generate interview questions aimed at the gaps found.
Kept separate from `/evaluate` because it costs one extra LLM call — HR usually
only needs questions for the few candidates who make the shortlist.

**Form-data:** same as `/evaluate`, plus optional `num_questions` (1-15, default 5).

**Response:**

```json
{
  "evaluation": { "final_score": 76.0, "gaps": ["Chưa dùng RabbitMQ"], "...": "..." },
  "questions": [
    {
      "question": "Kể về lần bạn phải xử lý một luồng công việc bất đồng bộ...",
      "targets_gap": "Chưa dùng RabbitMQ",
      "what_to_listen_for": "Nêu được cơ chế retry và xử lý tin nhắn trùng"
    }
  ]
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
├── crawl/                      # Synthetic JD/CV generation (needs an API key)
├── data/                        # Saved LLM output: parsed CVs/JDs + evaluations
├── frontend/                   # Demo UI (plain HTML/CSS/JS)
├── tests/                      # Unit tests (pytest) — 53 tests, no API key needed
├── run_distillation.py         # (CV, JD) pairs → training_data.csv
├── train_ranker.py             # Trains + evaluates the ranker against baselines
├── training_data.csv           # 200 rows: features + LLM labels
├── requirements.txt
├── .env.example
└── README.md
```

## Data Generation & Model Training Pipeline

To move from a hardcoded scoring formula to a model that learns from data, the project includes an additional set of scripts for dataset generation and training (outside `app/`, not part of the main API):

| Script                   | Purpose                                                                                                                                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crawl/generate_jd.py` | Generates diverse synthetic Vietnamese JDs (role/seniority/domain) with Claude, used because real JDs can't be crawled (ITviec/TopCV disallow it in robots.txt —`crawl/crawl_jd.py` checks and stops). `--count` is the only flag. |
| `crawl/generate_cv.py` | Generates CVs paired with each JD at varying fit levels (`good_fit`/`partial_fit`/`poor_fit`) so the dataset carries a distinguishable signal. `--per_jd` is the only flag.                                                     |
| `run_distillation.py`  | Runs each CV-JD pair through the actual`RecruitmentPipeline` to obtain `final_score` (label) and per-criterion scores (features) → exports `training_data.csv`.                                                                  |
| `train_ranker.py`      | Trains and evaluates a ranking model on`training_data.csv`, always against two baselines (constant prediction, and the rule-based formula itself).                                                                                    |

Intermediate LLM output is kept under `data/` (parsed CVs/JDs and the evaluations)
so that **changing a scoring formula does not require paying for the API again** —
features can be recomputed from the parsed data and re-joined with the existing
labels. See `data/README.md`.

> Generated data (`*.jsonl`, `training_data.csv`) contains **fictional** information (fake names/emails/phone numbers), so it's safe to store, but it's still recommended not to commit large dataset files directly to git (see `.gitignore`) — store them elsewhere (Google Drive, internal school storage...) and document how to regenerate them here instead.

## Using the Ranker as a Cheap Pre-filter

`RecruitmentPipeline` loads the trained model at startup and scores every
candidate with it — no LLM, a few milliseconds. That score is always returned as
`model_score`, next to `rule_based_score` and the LLM's `final_score`, so the
three can be compared.

Set `RANKER_PREFILTER_TOP_K` to turn it into a filter. For a batch of 100 CVs
with `K=20`:

|                    | LLM calls                            |                     |
| ------------------ | ------------------------------------ | ------------------- |
| Without pre-filter | 100 parse + 100 score =**200** |                     |
| With`K=20`       | 100 parse + 20 score =**120**  | **40% fewer** |

Parsing still runs for every CV — without it there are no features to score
with. The saving is on the write-up step.

Candidates below the cut get the model's score and `scored_by: "model"`, with an
explanation saying so. This distinction is not cosmetic: a model score and an
LLM review are different things, and HR must be able to tell which one they are
looking at.

The whole feature degrades quietly. No model file, no scikit-learn, a corrupt
pickle, a feature the model doesn't recognise — each of these turns the ranker
off and the pipeline behaves exactly as it did before. Feature order is read
*from the pickle*, never hardcoded, so retraining with a different feature set
cannot silently feed the model garbage.

> `pickle.load` executes arbitrary code. Only ever point `RANKER_MODEL_PATH` at
> a file produced by this project's own `train_ranker.py`.

## Calibrating the Scoring Formulas

The rule-based constants were **measured, not guessed**. Method: embed every skill,
job title, project description and major from the 200 parsed CVs and 100 JDs once,
then compare candidate formulas offline against the `fit_level` label that the
synthetic-data generator assigned when each CV was written (independent of any
scoring formula). Metric: AUC separating `good_fit` from `poor_fit`, plus the
share of CV pairs ordered correctly within the same JD.

**The finding that drove everything else:** cosine similarity from e5 on short
skill strings does *not* span [0, 1]. Across ~4,000 randomly paired, completely
unrelated skills the median similarity is already **0.79**, and 5% exceed **0.844**
— so the original `0.85` threshold sat barely above pure noise. Worse, genuine
cross-language synonyms scored *below* unrelated same-language pairs
("Kiểm thử tự động" ↔ "Automation Testing" = 0.780, while
"Adobe Photoshop" ↔ "Kubernetes" = 0.806).

| Criterion            | Before                              | After                                                                               | AUC (good vs poor)      |
| -------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- | ----------------------- |
| **Skills**     | count of`similarity ≥ 0.85`      | exact/substring match first, then similarity rescaled from the noise floor          | 0.982 →**1.000** |
| **Experience** | `0.6 × years + 0.4 × relevance` | `years × (0.3 + 0.7 × relevance)` — years only count when the role is relevant | 0.787 →**0.979** |
| **Education**  | degree rank only                    | *unchanged* — see below                                                          | 0.520                   |

Within-JD ranking accuracy of the rule-based score: **95% → 99%**.

The experience fix matters because addition let irrelevant tenure carry the score:
an accountant with 2.2 years applying for a junior QA role scored **0.93** on
experience. Multiplying instead of adding drops that to 0.30.

**A negative result worth recording:** multiplying the education score by the
similarity between the candidate's major and the job did *not* work. Every variant
tried (major vs. job title, major vs. title + required skills, several floors)
moved AUC only from 0.520 to at best 0.564, and all of them made the criterion
*flatter* (std 0.047 → 0.025) — i.e. less informative for the model downstream.
The formula was left alone. The cause is the data, not the formula: nearly every
synthetic CV holds a Bachelor's and nearly every JD asks for one, so there is
little signal to extract. In the trained model, `education_score` ends up with a
feature importance of **0.000**.

## Does the Trained Model Beat the Hardcoded Formula?

One command does everything — no flags:

```bash
python train_ranker.py
```

It evaluates every model, trains each on the full dataset, and writes to `train/`:

```
train/
├── comparison.csv     machine-readable results
├── comparison.md      the same table, ready to paste into a report
├── rf.pkl             each model saved under its own name
├── extratrees.pkl
├── gboost.pkl
└── histgb.pkl
```

5-fold cross-validation grouped by `jd_id`, so no JD appears in both train and
test. Predictions are clipped to [0, 100].

Scope: **tree-based models only** — bagging (Random Forest, Extra Trees) and
boosting (Gradient Boosting, HistGradientBoosting). `xgboost` and `lightgbm`
join the comparison automatically **if installed**; they are deliberately kept
out of `requirements.txt` so anyone who only runs the API does not download them.

|                                       | MAE            | R²    | Ranked correctly |
| ------------------------------------- | -------------- | ------ | ---------------- |
| Constant prediction (mean)            | 17.55          | 0.000  | 0/100            |
| `rule_based_score` used directly    | 10.65          | +0.639 | **99/100** |
| Random Forest — 3 criteria           | 6.03           | +0.846 | 98/100           |
| Random Forest — + JD context         | 5.47           | +0.874 | 96/100           |
| Extra Trees — 3 criteria             | 6.18           | +0.839 | 97/100           |
| **Extra Trees — + JD context** | **5.45** | +0.872 | **98/100** |
| Gradient Boosting — 3 criteria       | 5.98           | +0.853 | 94/100           |
| Gradient Boosting — + JD context     | 5.64           | +0.864 | 96/100           |
| HistGradientBoosting — 3 criteria    | 5.82           | +0.861 | 98/100           |
| HistGradientBoosting — + JD context  | 5.56           | +0.872 | 98/100           |

The `.pkl` files hold `{model, features, model_key, score_range}` — the feature
order matters, so read it from the file rather than hardcoding it. `*.pkl` is
gitignored (~13 MB total); the two comparison tables are not.

Read honestly, this says two different things:

- **For ranking, the model does not beat the formula.** Both sit at 99/100. Once
  the formula is calibrated there is essentially no headroom left, and with only
  100 comparable pairs the difference between 99% and 100% is a single pair —
  not enough to claim a winner either way.
- **For the absolute score, the model clearly wins:** MAE 10.65 → 5.45. That
  matters whenever the number itself is used — showing a 0-100 score to HR, or
  setting a cut-off for who advances.

Two further observations. Adding trees does not help much over the simplest
tree: at 200 rows there is little structure for a deep ensemble to find, and the
spread across all four tree models is under 0.2 MAE. And adding JD context (weights,
minimum years, required degree) helps precisely because the same triple of
criterion scores means different things under different JDs — without it the
model is asked to predict a variable it cannot see.

**What would actually move the needle** is more CVs *per JD*, not more JDs. The
dataset currently has exactly 2 CVs per JD → 100 ranking pairs. Five CVs per JD
would give 1,000 pairs from only 2.5× the rows.

## License

Graduation internship project — for academic purposes.
