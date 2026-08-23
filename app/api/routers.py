"""
API layer: expose pipeline qua REST endpoint để frontend (dashboard HR) gọi.

Lưu ý: đây là bản rút gọn tập trung vào luồng xử lý chính. Phần thực tế
nên bổ sung thêm: xác thực (auth), lưu kết quả vào DB, xử lý bất đồng bộ
(background task/queue) vì gọi LLM cho nhiều CV cùng lúc có thể mất
vài giây đến vài chục giây.
"""

import threading

from fastapi import APIRouter, Form, File, HTTPException, UploadFile

from app.core.settings import ModelSettings
from app.services.pipeline import RecruitmentPipeline
from app.utils.file_extraction import extract_text_from_file

router = APIRouter(prefix="/api/v1", tags=["recruitment"])

_pipeline: RecruitmentPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RecruitmentPipeline:
    """Trả về pipeline dùng chung cho cả tiến trình, dựng đúng 1 lần.

    Trước đây hàm này dựng RecruitmentPipeline MỚI cho mỗi request. Constructor
    của pipeline gọi get_embedding_client() -> HuggingFaceEmbeddings -> nạp
    model intfloat/multilingual-e5-large (~1.1GB) từ đĩa vào RAM. Nghĩa là mỗi
    lần gọi API đều phải nạp lại toàn bộ model, và cache embedding cũng bị xoá
    sạch theo. Giữ lại một instance duy nhất khắc phục cả hai vấn đề.

    Lưu ý: vì chỉ đọc .env một lần, đổi giá trị trong .env phải khởi động lại
    uvicorn mới có hiệu lực (README đã ghi rõ điều này).
    """
    global _pipeline
    if _pipeline is None:
        # Khoá + kiểm tra 2 lần: endpoint chạy trong threadpool nên nhiều
        # request đầu tiên có thể vào đây cùng lúc và cùng nạp model.
        with _pipeline_lock:
            if _pipeline is None:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                _pipeline = RecruitmentPipeline(ModelSettings())
    return _pipeline


def reset_pipeline() -> None:
    """Xoá instance đang cache -- dùng cho test hoặc khi cần nạp lại cấu hình."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None


@router.post("/evaluate")
def evaluate_candidate(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(None),
    jd_id: str = Form("JD-01"),
    jd_file: UploadFile = File(None),
):
    """Đánh giá 1 CV với 1 JD, trả về điểm số + giải thích.

    Khai báo `def` (không phải `async def`): thân hàm gọi LLM và embedding đều
    là code đồng bộ, chặn luồng. Với `async def`, FastAPI chạy thẳng trên event
    loop -> cả server đứng im trong lúc chờ. Với `def`, FastAPI tự đẩy sang
    threadpool nên các request khác vẫn được phục vụ.
    """
    if jd_file and jd_file.filename:
        try:
            jd_text = extract_text_from_file(jd_file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Lỗi đọc file JD: {e}")

    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung JD hoặc tải lên file JD.")

    try:
        cv_raw_text = extract_text_from_file(cv_file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        pipeline = get_pipeline()
        result = pipeline.run(
            cv_raw_text=cv_raw_text,
            jd_raw_text=jd_text,
            cv_id=cv_file.filename,
            jd_id=jd_id,
        )
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "AuthenticationError" in err_msg or "Invalid token" in err_msg:
            raise HTTPException(
                status_code=401,
                detail="Lỗi 401 (Invalid token): API Key Anthropic chưa hợp lệ hoặc thiếu ANTHROPIC_BASE_URL trong .env."
            )
        raise HTTPException(status_code=500, detail=f"Lỗi gọi LLM: {err_msg}")

    return {
        "evaluation": result.evaluation.model_dump(),
        "rule_based_score": result.rule_based_score,
        # Điểm từng tiêu chí kèm diễn giải -> HR thấy được điểm tổng đến từ đâu
        "criterion_scores": [c.model_dump() for c in result.criterion_scores],
    }


@router.post("/rank")
def rank_candidates(
    cv_files: list[UploadFile] = File(...),
    jd_text: str = Form(None),
    jd_id: str = Form("JD-01"),
    jd_file: UploadFile = File(None),
):
    """Đánh giá nhiều CV cùng lúc với 1 JD, trả về danh sách đã xếp hạng.

    Dùng `def` thay vì `async def` với cùng lý do ở /evaluate — endpoint này
    nặng hơn nhiều nên việc chặn event loop càng nghiêm trọng.
    """
    if jd_file and jd_file.filename:
        try:
            extracted_jd = extract_text_from_file(jd_file)
            if extracted_jd and extracted_jd.strip():
                jd_text = extracted_jd
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Lỗi đọc file JD: {e}")

    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung JD hoặc tải lên file JD.")

    cv_texts = {}
    for f in cv_files:
        try:
            cv_texts[f.filename] = extract_text_from_file(f)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")

    try:
        pipeline = get_pipeline()
        ranked = pipeline.rank_candidates(cv_texts, jd_raw_text=jd_text, jd_id=jd_id)
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "AuthenticationError" in err_msg or "Invalid token" in err_msg:
            raise HTTPException(
                status_code=401,
                detail="Lỗi 401 (Invalid token): API Key Anthropic chưa hợp lệ hoặc thiếu ANTHROPIC_BASE_URL trong .env."
            )
        raise HTTPException(status_code=500, detail=f"Lỗi gọi LLM: {err_msg}")

    return {
        "jd_id": jd_id,
        "total_candidates": len(ranked),
        "results": [
            {
                "rank": i + 1,
                "evaluation": r.evaluation.model_dump(),
                "rule_based_score": r.rule_based_score,
                "criterion_scores": [c.model_dump() for c in r.criterion_scores],
            }
            for i, r in enumerate(ranked)
        ],
    }