"""
Interview service: biến các khoảng trống (gaps) đã phát hiện thành bộ câu hỏi
phỏng vấn nhắm đúng chỗ cần kiểm chứng.

Vì sao tách thành service riêng thay vì nhét vào ScoringService: sinh câu hỏi
là việc TUỲ CHỌN và tốn thêm một lượt gọi LLM. HR thường chỉ cần nó cho vài
ứng viên lọt vào vòng trong, không phải cho cả trăm hồ sơ vừa quét. Tách ra
giúp bước chấm điểm hàng loạt không phải trả chi phí này.

Đầu vào là kết quả đánh giá đã có, nên không cần parse lại CV/JD.
"""

from app.core.model_config import BaseLLMClient
from app.schemas.models import (
    CandidateEvaluation,
    InterviewGuide,
    InterviewQuestion,
    ParsedJD,
)

DEFAULT_NUM_QUESTIONS = 5


class InterviewService:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def generate(
        self,
        evaluation: CandidateEvaluation,
        jd: ParsedJD,
        num_questions: int = DEFAULT_NUM_QUESTIONS,
    ) -> list[InterviewQuestion]:
        """Sinh câu hỏi phỏng vấn dựa trên gaps + strengths của ứng viên.

        Không nhận CV thô: chỉ dùng kết quả đánh giá đã tổng hợp, nên prompt
        cũng không chứa tên/email/số điện thoại của ứng viên.
        """
        if not evaluation.gaps:
            return []

        gaps = "\n".join(f"- {g}" for g in evaluation.gaps)
        strengths = "\n".join(f"- {s}" for s in evaluation.strengths) or "- (chưa ghi nhận)"

        prompt = f"""
Bạn là chuyên gia tuyển dụng đang chuẩn bị buổi phỏng vấn cho một ứng viên.

Vị trí tuyển dụng: {jd.job_title}
Kỹ năng bắt buộc: {", ".join(jd.require_skills) or "không rõ"}
Yêu cầu kinh nghiệm tối thiểu: {jd.min_experience_years} năm

Điểm mạnh đã ghi nhận của ứng viên:
{strengths}

Khoảng trống cần kiểm chứng:
{gaps}

Nhiệm vụ: soạn đúng {num_questions} câu hỏi phỏng vấn.

Yêu cầu:
1. Mỗi câu hỏi phải nhắm vào MỘT khoảng trống cụ thể ở trên, ghi rõ nó đang
   kiểm chứng khoảng trống nào.
2. Hỏi về tình huống và kinh nghiệm thực tế ("kể về lần bạn..."), không hỏi
   kiểu tra bài định nghĩa lý thuyết.
3. Một khoảng trống có thể chỉ là do CV viết thiếu chứ không phải ứng viên
   không biết -- hãy đặt câu hỏi cho ứng viên cơ hội chứng minh, đừng hỏi
   theo kiểu buộc tội.
4. Với mỗi câu, nêu rõ người phỏng vấn nên nghe thấy điều gì trong câu trả
   lời để kết luận ứng viên thực sự có năng lực đó.
5. Viết bằng tiếng Việt, tự nhiên như người thật hỏi.
""".strip()

        guide: InterviewGuide = self.llm.extract_structured(prompt, InterviewGuide)
        return guide.questions
