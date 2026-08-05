"""
Parsing service: nhận text thô (đã trích từ PDF/DOCX) và dùng LLM
để bóc tách thành dữ liệu có cấu trúc theo schema đã định nghĩa.

Bản này dùng extract_structured() của LangChain — truyền thẳng class
Pydantic (ParsedCV, ParsedJD) làm schema, không cần tự viết chuỗi mô tả
cấu trúc JSON (SCHEMA_HINT) như bản cũ. LangChain tự sinh JSON Schema từ
Pydantic model, ép LLM tuân theo, và validate luôn kết quả trả về.
"""

from app.core.model_config import BaseLLMClient
from app.schemas.models import ParsedCV, ParsedJD


class CVParsingService:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def parse(self, raw_text: str) -> ParsedCV:
        prompt = (
            "Bạn là hệ thống trích xuất thông tin CV. Đọc nội dung CV dưới đây "
            "và trích xuất chính xác thông tin, không suy diễn thêm nếu CV không "
            "đề cập. Nếu thiếu thông tin, để giá trị rỗng hoặc 0. "
            "Lưu ý: nếu ứng viên có nhiều bằng cấp (ví dụ Master và Bachelor), "
            "hãy liệt kê ĐẦY ĐỦ tất cả vào mảng education, không chỉ lấy 1 bằng. "
            "Với các dự án (projects): tách riêng khỏi work_experience, và đánh dấu "
            "is_academic=true nếu đây là đồ án môn học/dự án cá nhân/dự án học tập, "
            "is_academic=false nếu đây là dự án thực hiện trong công việc chính thức.\n\n"
            f"Nội dung CV:\n{raw_text}"
        )
        # Không cần SCHEMA_HINT nữa -> truyền thẳng class ParsedCV
        return self.llm.extract_structured(prompt, ParsedCV)


class JDParsingService:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def parse(self, raw_text: str) -> ParsedJD:
        prompt = (
            "Bạn là hệ thống trích xuất yêu cầu tuyển dụng. Đọc mô tả công việc "
            "(JD) dưới đây và tách rõ: kỹ năng bắt buộc (required_skills) khác "
            "với kỹ năng ưu tiên/cộng điểm (preferred_skills). Không gộp chung.\n\n"
            f"Nội dung JD:\n{raw_text}"
        )
        return self.llm.extract_structured(prompt, ParsedJD)