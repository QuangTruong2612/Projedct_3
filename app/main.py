"""Entry point của backend. Chạy: uvicorn app.main:app --reload"""

from dotenv import load_dotenv

# Phải load .env TRƯỚC khi import bất cứ module nào dùng os.getenv()
# (ví dụ app.core.settings) -> nếu để load_dotenv() sau các import khác,
# settings.py có thể đã đọc os.getenv() với giá trị rỗng trước khi .env kịp nạp.
load_dotenv()

import logging  # noqa: E402
import os
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.api.routers import get_pipeline, router  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp sẵn pipeline (kèm model embedding ~1.1GB) ngay khi khởi động,
    để request đầu tiên không phải gánh thời gian nạp model.

    Cố tình BẮT MỌI LỖI: nếu .env thiếu ANTHROPIC_API_KEY thì việc dựng LLM
    client sẽ hỏng. Không nên vì thế mà server không khởi động được — cứ để
    chạy tiếp, request đầu tiên sẽ dựng lại và trả về thông báo 401 rõ ràng
    trong routers.py.
    """
    try:
        get_pipeline()
        logger.info("Pipeline đã sẵn sàng (model embedding đã nạp).")
    except Exception as e:  # noqa: BLE001
        logger.warning("Chưa nạp sẵn được pipeline lúc khởi động: %s", e)
    yield


app = FastAPI(
    title="Recruitment AI API",
    description="Công cụ AI hỗ trợ tuyển dụng: phân tích và đánh giá hồ sơ ứng viên",
    version="0.1.0",
    lifespan=lifespan,
)

# Cấu hình CORS cho phép frontend kết nối mượt mà
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Khai báo đường dẫn thư mục frontend và phục vụ static files trực tiếp tại root
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

