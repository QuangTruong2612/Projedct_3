"""
Pipeline tổng hợp toàn bộ quy trình đánh giá 1 ứng viên với 1 JD:

  raw CV text ---\
                   -> parse -> match (rule-based) -> score (LLM) -> kết quả cuối
  raw JD text ---/

Module này là nơi duy nhất các API endpoint cần gọi tới —
không cần biết chi tiết từng service bên trong hoạt động ra sao.
"""

from dataclasses import dataclass

from app.core.embedding_config import get_embedding_client
from app.core.model_config import get_llm_client
from app.core.settings import ModelSettings
from app.schemas.models import CandidateEvaluation
from app.services.matching_service import MatchingService
from app.services.parsing_service import CVParsingService, JDParsingService
from app.services.scoring_service import ScoringService


@dataclass
class PipelineResult:
    evaluation: CandidateEvaluation
    rule_based_score: float  # giữ lại để so sánh/debug, hiển thị cho HR nếu cần


class RecruitmentPipeline:
    def __init__(self, settings: ModelSettings | None = None):
        self.settings = settings or ModelSettings()

        llm_client = get_llm_client(self.settings)
        embedder = get_embedding_client(self.settings)

        self.cv_parser = CVParsingService(llm_client)
        self.jd_parser = JDParsingService(llm_client)
        self.matcher = MatchingService(embedder, self.settings)
        self.scorer = ScoringService(llm_client)

    def run(
        self, cv_raw_text: str, jd_raw_text: str, cv_id: str, jd_id: str
    ) -> PipelineResult:
        cv = self.cv_parser.parse(cv_raw_text)
        jd = self.jd_parser.parse(jd_raw_text)

        match_result = self.matcher.match(cv, jd, cv_id=cv_id, jd_id=jd_id)
        evaluation = self.scorer.evaluate(cv, jd, match_result)

        return PipelineResult(
            evaluation=evaluation,
            rule_based_score=match_result.rule_based_score,
        )

    def rank_candidates(
        self, cv_texts: dict[str, str], jd_raw_text: str, jd_id: str
    ) -> list[PipelineResult]:
        """Đánh giá nhiều CV với cùng 1 JD, trả về danh sách đã xếp hạng
        theo final_score giảm dần -- dùng cho màn hình dashboard chính."""
        jd = self.jd_parser.parse(jd_raw_text)

        results = []
        for cv_id, raw_text in cv_texts.items():
            cv = self.cv_parser.parse(raw_text)
            match_result = self.matcher.match(cv, jd, cv_id=cv_id, jd_id=jd_id)
            evaluation = self.scorer.evaluate(cv, jd, match_result)
            results.append(
                PipelineResult(
                    evaluation=evaluation,
                    rule_based_score=match_result.rule_based_score,
                )
            )

        return sorted(results, key=lambda r: r.evaluation.final_score, reverse=True)