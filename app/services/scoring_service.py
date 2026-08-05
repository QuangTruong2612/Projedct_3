"""
Scoring service: nhận kết quả matching (rule-based) + CV/JD gốc,
đưa vào LLM để:
1. Rà soát lại điểm rule-based (LLM có thể phát hiện sai sót mà
   embedding similarity bỏ sót, ví dụ ngữ cảnh đặc thù)
2. Sinh giải thích tự nhiên: điểm mạnh, điểm yếu, lý do

Lưu ý: LLM KHÔNG tự tính điểm từ đầu, mà tinh chỉnh dựa trên điểm
rule-based đã có -> giảm rủi ro LLM "ảo tưởng" (hallucinate) điểm số
thiếu căn cứ.

Bản này dùng extract_structured() của LangChain với schema
LLMEvaluationOutput (không gồm cv_id/jd_id, vì 2 giá trị đó hệ thống
đã biết sẵn, tự ghép vào sau khi nhận kết quả từ LLM).
"""

from app.core.model_config import BaseLLMClient
from app.schemas.models import CandidateEvaluation, LLMEvaluationOutput, MatchResult, ParsedCV, ParsedJD


class ScoringService:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def evaluate(
        self, cv: ParsedCV, jd: ParsedJD, match_result: MatchResult
    ) -> CandidateEvaluation:
        criteria_summary = "\n".join(
            f"- {cs.criterion}: {cs.score:.2f} ({cs.detail})"
            for cs in match_result.criterion_scores
        )
        education_summary = (
            ", ".join(f"{e.degree} ({e.major})" for e in cv.educations)
            if cv.educations
            else "không rõ"
        )

        prompt = f"""
Bạn là chuyên gia tuyển dụng. Dưới đây là thông tin ứng viên, yêu cầu công việc,
và điểm số đã tính toán theo từng tiêu chí (thang 0-1).

Ứng viên:
- Kỹ năng: {", ".join(cv.skills) or "không rõ"}
- Kinh nghiệm: {cv.total_experience_year} năm
- Học vấn: {education_summary}

Yêu cầu công việc: {jd.job_title}
- Kỹ năng bắt buộc: {", ".join(jd.require_skills)}
- Kỹ năng ưu tiên: {", ".join(jd.preferred_skills)}
- Kinh nghiệm tối thiểu: {jd.min_experience_years} năm

Điểm rule-based theo từng tiêu chí:
{criteria_summary}

Điểm tổng hợp rule-based (0-100): {match_result.rule_based_score}

Nhiệm vụ của bạn:
1. Xem xét điểm rule-based có hợp lý không dựa trên thông tin thực tế của ứng viên.
   Bạn có thể điều chỉnh nhẹ (tăng/giảm tối đa 10 điểm) nếu thấy có yếu tố
   rule-based chưa phản ánh đúng (ví dụ: kinh nghiệm thực tế rất liên quan
   dù tên chức danh khác biệt).
2. Liệt kê 2-4 điểm mạnh cụ thể của ứng viên so với yêu cầu.
3. Liệt kê 2-4 khoảng trống (gaps) cụ thể.
4. Viết giải thích ngắn gọn (2-3 câu) cho HR, dễ hiểu, không dùng thuật ngữ kỹ thuật.
""".strip()

        # Không cần EVALUATION_SCHEMA_HINT nữa -> truyền thẳng LLMEvaluationOutput
        llm_output: LLMEvaluationOutput = self.llm.extract_structured(prompt, LLMEvaluationOutput)

        # Ghép cv_id/jd_id (đã biết sẵn từ match_result) với phần LLM vừa sinh ra
        return CandidateEvaluation(
            cv_id=match_result.cv_id,
            jd_id=match_result.jd_id,
            final_score=llm_output.final_score,
            strengths=llm_output.strengths,
            gaps=llm_output.gaps,
            explanation=llm_output.explanation,
        )