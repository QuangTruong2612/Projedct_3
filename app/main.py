"""Entry point của backend. Chạy: uvicorn app.main:app --reload"""

from dotenv import load_dotenv

# Phải load .env TRƯỚC khi import bất cứ module nào dùng os.getenv()
# (ví dụ app.core.settings) -> nếu để load_dotenv() sau các import khác,
# settings.py có thể đã đọc os.getenv() với giá trị rỗng trước khi .env kịp nạp.
load_dotenv()

import os
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from app.api.routers import router  # noqa: E402

app = FastAPI(
    title="Recruitment AI API",
    description="Công cụ AI hỗ trợ tuyển dụng: phân tích và đánh giá hồ sơ ứng viên",
    version="0.1.0",
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
