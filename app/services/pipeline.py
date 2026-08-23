"""
Pipeline tổng hợp toàn bộ quy trình đánh giá 1 ứng viên với 1 JD:

  raw CV text ---\
                   -> parse -> match (rule-based) -> score (LLM) -> kết quả cuối
  raw JD text ---/

Module này là nơi duy nhất các API endpoint cần gọi tới —
không cần biết chi tiết từng service bên trong hoạt động ra sao.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.core.embedding_config import get_embedding_client
from app.core.model_config import get_llm_client
from app.core.settings import ModelSettings
from app.schemas.models import CandidateEvaluation, CriterionScore, ParsedCV, ParsedJD
from app.services.matching_service import MatchingService
from app.services.parsing_service import CVParsingService, JDParsingService
from app.services.scoring_service import ScoringService


@dataclass
class PipelineResult:
    evaluation: CandidateEvaluation
    rule_based_score: float  # giữ lại để so sánh/debug, hiển thị cho HR nếu cần
    # Điểm chi tiết từng tiêu chí + dữ liệu đã parse. Trả kèm ra ngoài để bên
    # gọi không phải parse lại bằng LLM lần thứ hai chỉ để lấy breakdown
    # (run_distillation.py trước đây làm đúng như vậy: tốn gấp đôi chi phí API
    # và tệ hơn là hai lần parse có thể ra kết quả khác nhau, khiến feature
    # trong dataset không khớp với rule_based_score của chính dòng đó).
    criterion_scores: list[CriterionScore]
    parsed_cv: ParsedCV
    parsed_jd: ParsedJD


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
            criterion_scores=match_result.criterion_scores,
            parsed_cv=cv,
            parsed_jd=jd,
        )

    def rank_candidates(
        self, cv_texts: dict[str, str], jd_raw_text: str, jd_id: str
    ) -> list[PipelineResult]:
        """Đánh giá nhiều CV với cùng 1 JD, trả về danh sách đã xếp hạng
        theo final_score giảm dần -- dùng cho màn hình dashboard chính.

        JD chỉ parse 1 lần cho cả lô. Mỗi CV cần 2 lượt gọi LLM (parse + score)
        và phần lớn thời gian là chờ mạng, nên xử lý song song có giới hạn
        (rank_max_workers) thay vì tuần tự: 12 CV tuần tự mất vài phút, còn
        chạy song song thì gần bằng thời gian của CV chậm nhất trong mỗi đợt.
        """
        jd = self.jd_parser.parse(jd_raw_text)

        def evaluate_one(item: tuple[str, str]) -> PipelineResult:
            cv_id, raw_text = item
            cv = self.cv_parser.parse(raw_text)
            match_result = self.matcher.match(cv, jd, cv_id=cv_id, jd_id=jd_id)
            evaluation = self.scorer.evaluate(cv, jd, match_result)
            return PipelineResult(
                evaluation=evaluation,
                rule_based_score=match_result.rule_based_score,
                criterion_scores=match_result.criterion_scores,
                parsed_cv=cv,
                parsed_jd=jd,
            )

        items = list(cv_texts.items())
        if len(items) == 1:  # khỏi dựng threadpool cho đúng 1 CV
            results = [evaluate_one(items[0])]
        else:
            workers = min(self.settings.rank_max_workers, len(items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # map() giữ nguyên thứ tự đầu vào và ném lại lỗi đầu tiên
                # gặp phải -> giữ đúng hành vi cũ khi có CV lỗi.
                results = list(pool.map(evaluate_one, items))

        return sorted(results, key=lambda r: r.evaluation.final_score, reverse=True)