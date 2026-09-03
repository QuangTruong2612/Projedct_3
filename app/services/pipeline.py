"""
Pipeline tổng hợp toàn bộ quy trình đánh giá 1 ứng viên với 1 JD:

  raw CV text ---\
                   -> parse -> match (rule-based) -> score (LLM) -> kết quả cuối
  raw JD text ---/

Module này là nơi duy nhất các API endpoint cần gọi tới —
không cần biết chi tiết từng service bên trong hoạt động ra sao.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.core.embedding_config import get_embedding_client
from app.core.model_config import get_llm_client
from app.core.settings import ModelSettings
from app.schemas.models import (
    CandidateEvaluation,
    CriterionScore,
    DataCompleteness,
    InterviewQuestion,
    JDQualityReport,
    ParsedCV,
    ParsedJD,
    SkillImpact,
)
from app.services.interview_service import InterviewService
from app.services.jd_quality_service import JDQualityService
from app.services.matching_service import MatchingService
from app.services.ranker_service import RankerService
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
    # Cảnh báo khi CV parse thiếu -> điểm số có thể không đáng tin
    completeness: DataCompleteness | None = None
    # Bổ sung kỹ năng nào thì điểm tăng nhiều nhất (tính bằng công thức,
    # không tốn thêm lần gọi LLM hay embed nào)
    skill_gaps: list[SkillImpact] = field(default_factory=list)
    # Điểm do mô hình đã train dự đoán (không gọi LLM). None nếu chưa có model.
    # Luôn tính khi có model để HR đối chiếu được với điểm LLM.
    model_score: float | None = None


class RecruitmentPipeline:
    def __init__(self, settings: ModelSettings | None = None):
        self.settings = settings or ModelSettings()

        llm_client = get_llm_client(self.settings)
        embedder = get_embedding_client(self.settings)

        self.cv_parser = CVParsingService(llm_client, self.settings)
        self.jd_parser = JDParsingService(llm_client, self.settings)
        self.matcher = MatchingService(embedder, self.settings)
        self.scorer = ScoringService(llm_client)
        self.interviewer = InterviewService(llm_client)
        self.jd_checker = JDQualityService()
        self.ranker = RankerService(self.settings.ranker_model_path)

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
            completeness=match_result.completeness,
            skill_gaps=self.matcher.skill_gap_impact(cv, jd),
            model_score=self.ranker.predict(match_result, jd),
        )

    def rank_candidates(
        self, cv_texts: dict[str, str], jd_raw_text: str, jd_id: str
    ) -> list[PipelineResult]:
        """Đánh giá nhiều CV với cùng 1 JD, trả về danh sách đã xếp hạng
        theo final_score giảm dần -- dùng cho màn hình dashboard chính.

        JD chỉ parse 1 lần cho cả lô. Mỗi CV cần 2 lượt gọi LLM (parse + score)
        và phần lớn thời gian là chờ mạng, nên xử lý song song có giới hạn
        (rank_max_workers) thay vì tuần tự.

        BỘ LỌC RẺ: nếu có model đã train và bật `ranker_prefilter_top_k`, ta
        chấm toàn bộ bằng model trước (miễn phí), rồi CHỈ gọi LLM chấm điểm cho
        top-K. Các hồ sơ còn lại giữ nguyên điểm mô hình và được đánh dấu
        `scored_by="model"` để HR biết chưa có LLM đọc kỹ.
        Bước parse CV vẫn phải chạy cho mọi hồ sơ vì không có nó thì không có
        feature để mô hình chấm.
        """
        jd = self.jd_parser.parse(jd_raw_text)
        items = list(cv_texts.items())

        def parse_and_match(item: tuple[str, str]):
            cv_id, raw_text = item
            cv = self.cv_parser.parse(raw_text)
            match_result = self.matcher.match(cv, jd, cv_id=cv_id, jd_id=jd_id)
            return cv_id, cv, match_result

        # --- giai đoạn 1: parse + match (LLM parse, chưa chấm điểm) ---
        parsed = self._run_parallel(parse_and_match, items)

        # --- giai đoạn 2: chấm bằng mô hình (miễn phí) ---
        scored = [
            (cv_id, cv, match_result, self.ranker.predict(match_result, jd))
            for cv_id, cv, match_result in parsed
        ]

        top_k = self.settings.ranker_prefilter_top_k
        use_prefilter = (
            self.ranker.is_available
            and top_k > 0
            and len(scored) > top_k
            and all(s is not None for *_, s in scored)
        )

        if use_prefilter:
            # Hồ sơ điểm mô hình cao nhất mới đáng để LLM đọc kỹ
            ordered = sorted(scored, key=lambda t: t[3], reverse=True)
            need_llm, model_only = ordered[:top_k], ordered[top_k:]
        else:
            need_llm, model_only = scored, []

        # --- giai đoạn 3: LLM chấm điểm cho phần cần thiết ---
        def score_with_llm(entry):
            cv_id, cv, match_result, model_score = entry
            evaluation = self.scorer.evaluate(cv, jd, match_result)
            return self._build_result(cv, jd, match_result, evaluation, model_score)

        results = self._run_parallel(score_with_llm, need_llm)

        # --- phần bị lọc: chỉ có điểm mô hình, ghi rõ là chưa qua LLM ---
        for cv_id, cv, match_result, model_score in model_only:
            evaluation = CandidateEvaluation(
                cv_id=cv_id,
                jd_id=jd_id,
                final_score=model_score,
                strengths=[],
                gaps=[],
                explanation=(
                    f"Hồ sơ này được chấm bằng mô hình xếp hạng, chưa qua AI ngôn ngữ "
                    f"vì không nằm trong {top_k} hồ sơ điểm cao nhất. "
                    "Cần nhận xét chi tiết thì chấm lại riêng hồ sơ này."
                ),
                scored_by="model",
            )
            results.append(self._build_result(cv, jd, match_result, evaluation, model_score))

        return sorted(results, key=lambda r: r.evaluation.final_score, reverse=True)

    def _run_parallel(self, fn, items: list):
        """Chạy fn trên từng phần tử, song song có giới hạn. Giữ nguyên thứ tự
        và ném lại lỗi đầu tiên gặp phải, đúng như chạy tuần tự."""
        if not items:
            return []
        if len(items) == 1:  # khỏi dựng threadpool cho đúng 1 phần tử
            return [fn(items[0])]
        workers = min(self.settings.rank_max_workers, len(items))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def _build_result(self, cv, jd, match_result, evaluation, model_score) -> PipelineResult:
        return PipelineResult(
            evaluation=evaluation,
            rule_based_score=match_result.rule_based_score,
            criterion_scores=match_result.criterion_scores,
            parsed_cv=cv,
            parsed_jd=jd,
            completeness=match_result.completeness,
            skill_gaps=self.matcher.skill_gap_impact(cv, jd),
            model_score=model_score,
        )

    def check_jd_quality(self, jd_raw_text: str) -> JDQualityReport:
        """Parse JD rồi soi các vấn đề khiến việc tuyển dụng khó thành công.
        Chỉ tốn đúng 1 lượt gọi LLM (để parse), phần kiểm tra là luật cứng."""
        return self.jd_checker.check(self.jd_parser.parse(jd_raw_text))

    def generate_interview_questions(
        self,
        cv_raw_text: str,
        jd_raw_text: str,
        cv_id: str,
        jd_id: str,
        num_questions: int = 5,
    ) -> tuple[PipelineResult, list[InterviewQuestion]]:
        """Chấm điểm rồi sinh luôn bộ câu hỏi phỏng vấn nhắm vào các gaps.

        Trả về cả kết quả đánh giá để bên gọi không phải chạy pipeline hai lần.
        """
        result = self.run(cv_raw_text, jd_raw_text, cv_id=cv_id, jd_id=jd_id)
        questions = self.interviewer.generate(
            result.evaluation, result.parsed_jd, num_questions=num_questions
        )
        return result, questions