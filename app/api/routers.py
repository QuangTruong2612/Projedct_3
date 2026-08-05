"""
API layer: expose pipeline qua REST endpoint để frontend (dashboard HR) gọi.

Lưu ý: đây là bản rút gọn tập trung vào luồng xử lý chính. Phần thực tế
nên bổ sung thêm: xác thực (auth), lưu kết quả vào DB, xử lý bất đồng bộ
(background task/queue) vì gọi LLM cho nhiều CV cùng lúc có thể mất
vài giây đến vài chục giây.
"""

from fastapi import APIRouter, Form, File, HTTPException, UploadFile

from app.core.settings import ModelSettings
from app.services.pipeline import RecruitmentPipeline
from app.utils.file_extraction import extract_text_from_file

router = APIRouter(prefix="/api/v1", tags=["recruitment"])

_pipeline = None


def get_pipeline() -> RecruitmentPipeline:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    return RecruitmentPipeline(ModelSettings())


@router.post("/evaluate")
async def evaluate_candidate(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(None),
    jd_id: str = Form("JD-01"),
    jd_file: UploadFile = File(None),
):
    """Đánh giá 1 CV với 1 JD, trả về điểm số + giải thích."""
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
    }


@router.post("/rank")
async def rank_candidates(
    cv_files: list[UploadFile] = File(...),
    jd_text: str = Form(None),
    jd_id: str = Form("JD-01"),
    jd_file: UploadFile = File(None),
):
    """Đánh giá nhiều CV cùng lúc với 1 JD, trả về danh sách đã xếp hạng."""
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
            }
            for i, r in enumerate(ranked)
        ],
    }