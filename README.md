
# Recruitment AI — Hệ thống đánh giá CV tự động

Công cụ AI hỗ trợ tuyển dụng: tự động phân tích CV và JD (Job Description), đối chiếu mức độ phù hợp, và đưa ra điểm số + giải thích chi tiết giúp HR sàng lọc ứng viên nhanh hơn.

Đồ án thực tập tốt nghiệp.

---

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Cấu hình biến môi trường](#cấu-hình-biến-môi-trường)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [API Endpoints](#api-endpoints)
- [Cấu trúc project](#cấu-trúc-project)
- [Pipeline sinh dữ liệu &amp; training model](#pipeline-sinh-dữ-liệu--training-model)
- [Roadmap](#roadmap)

---

## Tính năng

- **Trích xuất thông tin CV/JD tự động** bằng LLM — parse CV (PDF/DOCX) và JD thành dữ liệu có cấu trúc (kỹ năng, kinh nghiệm, học vấn, dự án...).
- **Chấm điểm 2 lớp**:
  1. *Rule-based matching*: so khớp kỹ năng bằng embedding similarity, tính điểm kinh nghiệm/học vấn theo công thức trọng số.
  2. *LLM scoring*: LLM xem xét lại kết quả rule-based, tinh chỉnh điểm số cuối và sinh giải thích (điểm mạnh, điểm thiếu sót) bằng ngôn ngữ tự nhiên.
- **Đánh giá 1-CV-1-JD** (`/api/v1/evaluate`) và **xếp hạng nhiều CV cho 1 JD** (`/api/v1/rank`).
- **Hỗ trợ nhiều nhà cung cấp LLM/embedding** (Anthropic, OpenAI, mô hình mã nguồn mở qua Ollama/sentence-transformers) — đổi qua biến môi trường, không cần sửa code.
- **Frontend đơn giản** (HTML/CSS/JS thuần) để demo upload CV/JD và xem kết quả trực quan.

## Kiến trúc hệ thống

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
                 │ → rule_based_score │   công thức trọng số)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Scoring Service    │  (LLM tinh chỉnh điểm +
                 │ → final_score      │   giải thích strengths/gaps)
                 └─────────┬─────────┘
                           ▼
                   CandidateEvaluation
                (trả về API / hiển thị FE)
```

Toàn bộ luồng trên được điều phối bởi `RecruitmentPipeline` (`app/services/pipeline.py`).

## Công nghệ sử dụng

| Thành phần         | Công nghệ                                                                           |
| -------------------- | ------------------------------------------------------------------------------------- |
| Backend              | FastAPI, Pydantic v2                                                                  |
| LLM orchestration    | LangChain (langchain-core, langchain-anthropic, langchain-openai, langchain-ollama)   |
| LLM provider         | Anthropic Claude (mặc định), OpenAI, hoặc mô hình mở qua Ollama                |
| Embedding            | intfloat/multilingual-e5-large (mặc định, mã nguồn mở) hoặc Voyage AI / OpenAI |
| Đọc file CV        | pypdf, docx2txt                                                                       |
| Database (dự kiến) | PostgreSQL + pgvector (qua SQLAlchemy)                                                |
| Frontend             | HTML/CSS/JS thuần                                                                    |
| Testing              | pytest                                                                                |

## Cài đặt

**Yêu cầu:** Python 3.11+

```bash
git clone <repo-url>
cd thực_tập_tốt_nghiệp

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

> Nếu dùng `EMBEDDING_PROVIDER=open_source`, lần chạy đầu sẽ tự tải model `intfloat/multilingual-e5-large` (~1.1GB) — cần kết nối mạng ổn định.

## Cấu hình biến môi trường

Copy file mẫu rồi điền giá trị thật:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

| Biến                     | Mô tả                                                       | Giá trị mặc định              |
| ------------------------- | ------------------------------------------------------------- | ---------------------------------- |
| `LLM_PROVIDER`          | `anthropic` / `openai` / `open_source`                  | `anthropic`                      |
| `LLM_MODEL_NAME`        | Tên model LLM                                                | `claude-sonnet-5`                |
| `LLM_TEMPERATURE`       | Độ ngẫu nhiên của LLM (0 = ổn định nhất)             | `0.0`                            |
| `ANTHROPIC_API_KEY`     | API key Anthropic (bắt buộc nếu`LLM_PROVIDER=anthropic`) | —                                 |
| `ANTHROPIC_BASE_URL`    | Endpoint tuỳ chỉnh (nếu dùng proxy)                       | `https://api.anthropic.com`      |
| `EMBEDDING_PROVIDER`    | `voyage` / `openai` / `open_source`                     | `open_source`                    |
| `EMBEDDING_MODEL_NAME`  | Tên model embedding                                          | `intfloat/multilingual-e5-large` |
| `SKILL_MATCH_THRESHOLD` | Ngưỡng cosine similarity để coi 2 kỹ năng là khớp     | `0.85`                           |
| `MAX_INPUT_TOKENS`      | Giới hạn token đầu vào cho LLM                           | `4000`                           |

> ⚠️ **Không bao giờ commit file `.env` thật lên git.** File này đã được thêm vào `.gitignore`. Nếu API key từng bị commit nhầm trong lịch sử git, cần **revoke/rotate key đó ngay** trên Anthropic Console, xoá key khỏi lịch sử commit (`git filter-repo` hoặc BFG Repo-Cleaner), rồi mới push.

## Chạy ứng dụng

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000/api/v1`
- Frontend demo: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- API docs (Swagger UI tự sinh bởi FastAPI): `http://localhost:8000/docs`

## API Endpoints

### `POST /api/v1/evaluate`

Đánh giá 1 CV với 1 JD.

**Form-data:**

| Field       | Kiểu           | Bắt buộc                         |
| ----------- | --------------- | ---------------------------------- |
| `cv_file` | file (PDF/DOCX) | ✔                                 |
| `jd_text` | text            | ✔ (nếu không có`jd_file`)    |
| `jd_file` | file            | ✔ (nếu không có`jd_text`)    |
| `jd_id`   | text            | tuỳ chọn, mặc định`"JD-01"` |

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

Đánh giá nhiều CV cùng lúc với 1 JD, trả về danh sách đã xếp hạng theo `final_score` giảm dần.

**Form-data:** tương tự `/evaluate` nhưng `cv_files` là danh sách nhiều file.

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

## Cấu trúc project

```
.
├── app/
│   ├── main.py                 # Entry point FastAPI
│   ├── api/
│   │   └── routers.py          # REST endpoints (/evaluate, /rank)
│   ├── core/
│   │   ├── settings.py         # Đọc cấu hình từ biến môi trường
│   │   ├── model_config.py     # Khởi tạo LLM client theo provider
│   │   └── embedding_config.py # Khởi tạo embedding client theo provider
│   ├── services/
│   │   ├── parsing_service.py  # Parse CV/JD → structured data (LLM)
│   │   ├── matching_service.py # Rule-based matching (embedding similarity)
│   │   ├── scoring_service.py  # LLM tinh chỉnh điểm + giải thích
│   │   └── pipeline.py         # Điều phối toàn bộ luồng
│   ├── schemas/
│   │   └── models.py           # Pydantic models (ParsedCV, ParsedJD, CandidateEvaluation...)
│   └── utils/
│       └── file_extraction.py  # Đọc text từ PDF/DOCX
├── frontend/                   # Demo UI (HTML/CSS/JS thuần)
├── tests/                      # Unit test (pytest)
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline sinh dữ liệu & training model

Để nâng cấp từ công thức chấm điểm cố định (rule-based cứng) sang model học được từ dữ liệu, project có thêm nhánh scripts hỗ trợ sinh dataset và training (nằm ngoài `app/`, không phải phần API chính):

| Script                             | Vai trò                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_synthetic_jd.py`       | Sinh JD tiếng Việt tổng hợp đa dạng (role/seniority/domain) bằng Claude, dùng khi không crawl được JD thật (robots.txt chặn ITviec/TopCV).            |
| `generate_synthetic_cv.py`       | Sinh CV ghép với từng JD ở nhiều mức độ khớp (`good_fit`/`partial_fit`/`poor_fit`) để dataset có đủ tín hiệu phân biệt.                     |
| `run_distillation.py`            | Chạy từng cặp CV-JD qua chính`RecruitmentPipeline` để lấy `final_score` (label) + điểm theo từng tiêu chí (feature) → xuất `training_data.csv`. |
| `train_ranker.py` *(sắp có)* | Train model ranking nhẹ (XGBoost/LightGBM) trên`training_data.csv`, thay thế một phần việc gọi LLM khi cần lọc nhanh số lượng lớn CV.                |

> Dữ liệu sinh ra (`*.jsonl`, `training_data.csv`) chứa thông tin **hư cấu** (tên/email/SĐT giả), an toàn để lưu trữ, nhưng khuyến nghị không commit trực tiếp file dataset lớn lên git (xem `.gitignore`) — nên lưu ở storage riêng (Google Drive, Drive nội bộ trường...) và ghi rõ cách tái tạo trong README.

## Roadmap

- [ ] Xử lý bất đồng bộ (background task/queue) cho `/rank` khi số lượng CV lớn.
- [ ] Lưu kết quả đánh giá vào PostgreSQL + pgvector (cache embedding, lịch sử đánh giá).
- [ ] Train model ranker nhẹ (XGBoost/LightGBM) làm bộ lọc nhanh trước khi gọi LLM chi tiết.
- [ ] Bộ eval benchmark (CV-JD gán nhãn tay) để đo độ chính xác hệ thống.
- [ ] Ẩn thông tin nhân khẩu học (tên, giới tính, tuổi) khỏi input LLM để giảm thiên vị.
- [ ] Giải thích trực quan hơn (highlight câu/đoạn trong CV khớp với từng yêu cầu JD).

---

## License

Đồ án thực tập tốt nghiệp — mục đích học thuật.
