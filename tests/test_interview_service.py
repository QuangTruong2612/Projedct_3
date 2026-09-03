"""Test cho việc sinh câu hỏi phỏng vấn từ các khoảng trống đã phát hiện."""

from app.schemas.models import (
    CandidateEvaluation,
    InterviewGuide,
    InterviewQuestion,
    ParseJD,
)
from app.services.interview_service import InterviewService


class SpyLLM:
    def __init__(self, questions=None):
        self.prompts: list[str] = []
        self.questions = questions if questions is not None else [
            InterviewQuestion(
                question="Kể về lần bạn xử lý hàng đợi tin nhắn?",
                targets_gap="Chưa dùng RabbitMQ",
                what_to_listen_for="Nêu được cơ chế retry và xử lý tin nhắn trùng",
            )
        ]

    def extract_structured(self, prompt: str, schema):
        self.prompts.append(prompt)
        return InterviewGuide(questions=self.questions)


def make_eval(**kw) -> CandidateEvaluation:
    base = dict(
        cv_id="cv.pdf",
        jd_id="JD-01",
        final_score=75.0,
        strengths=["5 năm kinh nghiệm Node.js"],
        gaps=["Chưa dùng RabbitMQ", "Chưa có kinh nghiệm Kubernetes"],
        explanation="ok",
    )
    base.update(kw)
    return CandidateEvaluation(**base)


JD = ParseJD(
    job_title="Senior Backend Developer",
    require_skills=["Node.js", "RabbitMQ"],
    min_experience_years=4,
)


def test_sinh_duoc_cau_hoi():
    llm = SpyLLM()
    questions = InterviewService(llm).generate(make_eval(), JD)

    assert len(questions) == 1
    assert questions[0].targets_gap == "Chưa dùng RabbitMQ"
    assert questions[0].what_to_listen_for


def test_khong_co_gap_thi_khong_goi_llm():
    """Không có khoảng trống nào -> không tốn thêm một lượt gọi LLM."""
    llm = SpyLLM()
    questions = InterviewService(llm).generate(make_eval(gaps=[]), JD)

    assert questions == []
    assert llm.prompts == []


def test_prompt_chua_du_gap_va_thong_tin_vi_tri():
    llm = SpyLLM()
    InterviewService(llm).generate(make_eval(), JD)

    prompt = llm.prompts[0]
    assert "Chưa dùng RabbitMQ" in prompt
    assert "Chưa có kinh nghiệm Kubernetes" in prompt
    assert "Senior Backend Developer" in prompt


def test_prompt_khong_chua_thong_tin_ca_nhan():
    """Chỉ nhận kết quả đánh giá đã tổng hợp, không nhận CV thô -> prompt
    không thể chứa tên/email/số điện thoại ứng viên."""
    llm = SpyLLM()
    InterviewService(llm).generate(make_eval(), JD)

    prompt = llm.prompts[0].lower()
    assert "cv.pdf" not in prompt
    assert "@" not in prompt


def test_truyen_duoc_so_luong_cau_hoi():
    llm = SpyLLM()
    InterviewService(llm).generate(make_eval(), JD, num_questions=8)
    assert "8 câu hỏi" in llm.prompts[0]


def test_strengths_rong_van_chay_duoc():
    llm = SpyLLM()
    questions = InterviewService(llm).generate(make_eval(strengths=[]), JD)
    assert len(questions) == 1
    assert "(chưa ghi nhận)" in llm.prompts[0]


def test_pipeline_sinh_cau_hoi_khong_chay_lai_toan_bo(monkeypatch):
    """generate_interview_questions() phải trả kèm kết quả đánh giá để bên gọi
    không phải chạy pipeline lần thứ hai."""
    from app.services.pipeline import RecruitmentPipeline

    monkeypatch.setattr("app.services.pipeline.get_llm_client", lambda s: object())
    monkeypatch.setattr("app.services.pipeline.get_embedding_client", lambda s: object())
    pipeline = RecruitmentPipeline()

    calls = {"n": 0}

    class FakeCVParser:
        def parse(self, text):
            calls["n"] += 1
            from app.schemas.models import ParseCV
            return ParseCV()

    class FakeJDParser:
        def parse(self, text):
            return JD

    class FakeMatcher:
        def match(self, cv, jd, cv_id, jd_id):
            from app.schemas.models import CriterionScore, MatchResult
            return MatchResult(
                cv_id=cv_id, jd_id=jd_id,
                criterion_scores=[CriterionScore(criterion="skills", score=0.5)],
                rule_based_score=50.0,
            )

        def skill_gap_impact(self, cv, jd, top_n=5):
            return []

    class FakeScorer:
        def evaluate(self, cv, jd, mr):
            return make_eval(cv_id=mr.cv_id)

    pipeline.cv_parser = FakeCVParser()
    pipeline.jd_parser = FakeJDParser()
    pipeline.matcher = FakeMatcher()
    pipeline.scorer = FakeScorer()
    pipeline.interviewer = InterviewService(SpyLLM())

    result, questions = pipeline.generate_interview_questions(
        cv_raw_text="cv", jd_raw_text="jd", cv_id="cv.pdf", jd_id="JD-01"
    )

    assert calls["n"] == 1  # CV chỉ parse đúng 1 lần
    assert result.evaluation.final_score == 75.0
    assert len(questions) == 1
