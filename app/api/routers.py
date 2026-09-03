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


def _raise_llm_error(e: Exception) -> None:
    """Chuyển lỗi khi gọi LLM thành HTTPException với thông báo dễ hiểu.
    Gom về một chỗ vì cả ba endpoint đều cần xử lý giống nhau."""
    err_msg = str(e)
    if "401" in err_msg or "AuthenticationError" in err_msg or "Invalid token" in err_msg:
        raise HTTPException(
            status_code=401,
            detail="Lỗi 401 (Invalid token): API Key Anthropic chưa hợp lệ hoặc thiếu ANTHROPIC_BASE_URL trong .env."
        )
    raise HTTPException(status_code=500, detail=f"Lỗi gọi LLM: {err_msg}")


def _read_jd_text(jd_text: str | None, jd_file: UploadFile | None) -> str:
    """Lấy nội dung JD từ file (ưu tiên) hoặc từ ô nhập text."""
    if jd_file and jd_file.filename:
        try:
            extracted = extract_text_from_file(jd_file)
            if extracted and extracted.strip():
                jd_text = extracted
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Lỗi đọc file JD: {e}")

    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung JD hoặc tải lên file JD.")
    return jd_text


@router.post("/check-jd")
def check_jd(
    jd_text: str = Form(None),
    jd_file: UploadFile = File(None),
):
    """Soi chất lượng một JD trước khi dùng nó để chấm ứng viên.

    Một JD liệt kê 19 kỹ năng "bắt buộc" sẽ khiến MỌI ứng viên đều điểm thấp --
    lỗi nằm ở JD chứ không phải ở ứng viên. Nên chạy kiểm tra này trước.
    """
    jd_text = _read_jd_text(jd_text, jd_file)

    try:
        report = get_pipeline().check_jd_quality(jd_text)
    except Exception as e:
        _raise_llm_error(e)

    return {**report.model_dump(), "is_healthy": report.is_healthy}


@router.post("/interview-questions")
def interview_questions(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(None),
    jd_id: str = Form("JD-01"),
    jd_file: UploadFile = File(None),
    num_questions: int = Form(5),
):
    """Chấm điểm 1 CV rồi sinh bộ câu hỏi phỏng vấn nhắm vào đúng các khoảng
    trống tìm được.

    Tách riêng khỏi /evaluate vì việc này tốn thêm một lượt gọi LLM: HR thường
    chỉ cần câu hỏi cho vài ứng viên lọt vòng trong, không phải cho cả lô.
    """
    jd_text = _read_jd_text(jd_text, jd_file)

    try:
        cv_raw_text = extract_text_from_file(cv_file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not 1 <= num_questions <= 15:
        raise HTTPException(status_code=400, detail="num_questions phải nằm trong khoảng 1-15.")

    try:
        result, questions = get_pipeline().generate_interview_questions(
            cv_raw_text=cv_raw_text,
            jd_raw_text=jd_text,
            cv_id=cv_file.filename,
            jd_id=jd_id,
            num_questions=num_questions,
        )
    except Exception as e:
        _raise_llm_error(e)

    return {
        "evaluation": result.evaluation.model_dump(),
        "questions": [q.model_dump() for q in questions],
    }


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
    jd_text = _read_jd_text(jd_text, jd_file)

    try:
        cv_raw_text = extract_text_from_file(cv_file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = get_pipeline().run(
            cv_raw_text=cv_raw_text,
            jd_raw_text=jd_text,
            cv_id=cv_file.filename,
            jd_id=jd_id,
        )
    except Exception as e:
        _raise_llm_error(e)

    return {
        "evaluation": result.evaluation.model_dump(),
        "rule_based_score": result.rule_based_score,
        # Điểm từng tiêu chí kèm diễn giải -> HR thấy được điểm tổng đến từ đâu
        "criterion_scores": [c.model_dump() for c in result.criterion_scores],
        # Cảnh báo nếu CV parse thiếu -> HR biết điểm này chưa chắc đáng tin
        "completeness": result.completeness.model_dump() if result.completeness else None,
        # "Bổ sung kỹ năng nào thì điểm lên bao nhiêu" -- giải thích được điểm số
        "skill_gaps": [g.model_dump() for g in result.skill_gaps],
        # Điểm do mô hình đã train dự đoán, để đối chiếu với điểm LLM
        "model_score": result.model_score,
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
    jd_text = _read_jd_text(jd_text, jd_file)

    cv_texts = {}
    for f in cv_files:
        try:
            cv_texts[f.filename] = extract_text_from_file(f)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")

    try:
        ranked = get_pipeline().rank_candidates(cv_texts, jd_raw_text=jd_text, jd_id=jd_id)
    except Exception as e:
        _raise_llm_error(e)

    return {
        "jd_id": jd_id,
        "total_candidates": len(ranked),
        "results": [
            {
                "rank": i + 1,
                "evaluation": r.evaluation.model_dump(),
                "rule_based_score": r.rule_based_score,
                "criterion_scores": [c.model_dump() for c in r.criterion_scores],
                "completeness": r.completeness.model_dump() if r.completeness else None,
                "skill_gaps": [g.model_dump() for g in r.skill_gaps],
                "model_score": r.model_score,
            }
            for i, r in enumerate(ranked)
        ],
    }