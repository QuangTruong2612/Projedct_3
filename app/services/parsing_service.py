"""
Parsing service: nhận text thô (đã trích từ PDF/DOCX) và dùng LLM
để bóc tách thành dữ liệu có cấu trúc theo schema đã định nghĩa.

Bản này dùng extract_structured() của LangChain — truyền thẳng class
Pydantic (ParsedCV, ParsedJD) làm schema, không cần tự viết chuỗi mô tả
cấu trúc JSON (SCHEMA_HINT) như bản cũ. LangChain tự sinh JSON Schema từ
Pydantic model, ép LLM tuân theo, và validate luôn kết quả trả về.
"""

from app.core.model_config import BaseLLMClient
from app.core.settings import ModelSettings
from app.schemas.models import ParsedCV, ParsedJD

# Ước lượng thô: 1 token ~ 3 ký tự với văn bản tiếng Việt có dấu (tiếng Việt
# tốn token hơn tiếng Anh vì dấu thanh thường bị tách riêng). Cố tình chọn
# thấp để phần cắt bớt thiên về an toàn.
CHARS_PER_TOKEN = 3

TRUNCATION_NOTICE = "\n\n[...nội dung đã được cắt bớt do quá dài...]"


def truncate_for_llm(text: str, max_tokens: int) -> tuple[str, bool]:
    """Cắt bớt văn bản quá dài trước khi đưa vào prompt.

    Trả về (văn bản, đã_bị_cắt).

    Vì sao cần: trước đây `MAX_INPUT_TOKENS` được khai báo trong settings và
    ghi trong README nhưng KHÔNG nơi nào dùng, nên một CV scan 20 trang được
    nhồi nguyên vào prompt -- vừa tốn tiền vừa có thể vượt giới hạn context
    của model.

    Cắt ở ĐẦU và CUỐI, bỏ phần giữa: thông tin quan trọng nhất của một CV
    thường nằm ở đầu (thông tin cá nhân, kỹ năng, kinh nghiệm gần nhất) và
    cuối (học vấn, chứng chỉ); phần giữa thường là mô tả công việc cũ.
    """
    if max_tokens <= 0:
        return text, False

    budget = max_tokens * CHARS_PER_TOKEN
    if len(text) <= budget:
        return text, False

    # Ngân sách còn nhỏ hơn cả dòng ghi chú -> chèn ghi chú vào sẽ khiến kết quả
    # DÀI HƠN giới hạn. Trường hợp này cắt thẳng, không ghi chú.
    if budget <= len(TRUNCATION_NOTICE):
        return text[:budget], True

    keep = budget - len(TRUNCATION_NOTICE)
    head = int(keep * 0.7)
    tail = keep - head
    truncated = text[:head] + TRUNCATION_NOTICE + (text[-tail:] if tail > 0 else "")
    return truncated, True


class CVParsingService:
    def __init__(self, llm_client: BaseLLMClient, settings: ModelSettings | None = None):
        self.llm = llm_client
        self.settings = settings or ModelSettings()

    def parse(self, raw_text: str) -> ParsedCV:
        text, _ = truncate_for_llm(raw_text, self.settings.max_input_tokens)
        prompt = (
            "Bạn là hệ thống trích xuất thông tin CV. Đọc nội dung CV dưới đây "
            "và trích xuất chính xác thông tin, không suy diễn thêm nếu CV không "
            "đề cập. Nếu thiếu thông tin, để giá trị rỗng hoặc 0. "
            "Lưu ý: nếu ứng viên có nhiều bằng cấp (ví dụ Master và Bachelor), "
            "hãy liệt kê ĐẦY ĐỦ tất cả vào mảng educations, không chỉ lấy 1 bằng. "
            "Với các dự án (projects): tách riêng khỏi work_experiences, và đánh dấu "
            "is_academic=true nếu đây là đồ án môn học/dự án cá nhân/dự án học tập, "
            "is_academic=false nếu đây là dự án thực hiện trong công việc chính thức.\n\n"
            f"Nội dung CV:\n{text}"
        )
        # Không cần SCHEMA_HINT nữa -> truyền thẳng class ParsedCV
        return self.llm.extract_structured(prompt, ParsedCV)


class JDParsingService:
    def __init__(self, llm_client: BaseLLMClient, settings: ModelSettings | None = None):
        self.llm = llm_client
        self.settings = settings or ModelSettings()

    def parse(self, raw_text: str) -> ParsedJD:
        text, _ = truncate_for_llm(raw_text, self.settings.max_input_tokens)
        prompt = (
            "Bạn là hệ thống trích xuất yêu cầu tuyển dụng. Đọc mô tả công việc "
            "(JD) dưới đây và tách rõ: kỹ năng bắt buộc (require_skills) khác "
            "với kỹ năng ưu tiên/cộng điểm (preferred_skills). Không gộp chung.\n\n"
            f"Nội dung JD:\n{text}"
        )
        return self.llm.extract_structured(prompt, ParsedJD)
