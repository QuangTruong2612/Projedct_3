"""Test cho RankerService và bộ lọc rẻ trong pipeline.

Không nạp model thật: dùng model giả có `.predict()` để kiểm tra logic.
"""

import pickle

import pytest

from app.core.settings import ModelSettings
from app.schemas.models import CriterionScore, MatchResult, ParseCV, ParseJD
from app.services.pipeline import RecruitmentPipeline
from app.services.ranker_service import RankerService

FEATURES = [
    "skill_score", "experience_score", "education_score",
    "w_skills", "w_experience", "w_education",
    "min_experience_years", "required_degree_rank", "n_require_skills",
]


# Các model giả phải khai báo ở MỨC MODULE thì pickle mới lưu được.
# Cũng vì pickle sao chép object, test không thể đọc thuộc tính của model sau
# khi service nạp lên -> thay vào đó cho model TRẢ VỀ giá trị suy ngược được.


class EchoFirstFeature:
    """Trả về feature đầu tiên nhân 10 -> test suy ra được service đã dựng
    vector theo thứ tự nào."""

    def predict(self, X):
        return [X[0][0] * 10]


class SumFeatures:
    """Trả về tổng các feature -> test kiểm tra được toàn bộ giá trị."""

    def predict(self, X):
        return [sum(X[0])]


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return [self.value]


class ExplodingModel:
    def predict(self, X):
        raise RuntimeError("model hong")


# Dùng cho các test bộ lọc rẻ: điểm = skill_score * 100
FakeModel = EchoFirstFeature


def write_bundle(path, model, features=None, score_range=(0.0, 100.0)):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": features or FEATURES,
                "model_key": "fake",
                "score_range": list(score_range),
            },
            f,
        )
    return path


def make_match(skills=0.8, exp=0.6, edu=1.0) -> MatchResult:
    return MatchResult(
        cv_id="cv.pdf", jd_id="JD",
        criterion_scores=[
            CriterionScore(criterion="skills", score=skills),
            CriterionScore(criterion="experience", score=exp),
            CriterionScore(criterion="education", score=edu),
        ],
        rule_based_score=70.0,
    )


JD = ParseJD(
    job_title="Backend Developer",
    require_skills=["Python", "SQL"],
    min_experience_years=3,
    required_degree="Bachelor",
    weights={"skills": 0.5, "experience": 0.3, "education": 0.2},
)


# ---------------------------------------------------------------- nạp model

def test_khong_co_duong_dan_thi_tat(tmp_path):
    svc = RankerService(None)
    assert svc.is_available is False
    assert svc.predict(make_match(), JD) is None


def test_file_khong_ton_tai_thi_tat(tmp_path):
    svc = RankerService(tmp_path / "khong_co.pkl")
    assert svc.is_available is False


def test_file_hong_khong_lam_sap_app(tmp_path):
    """File hỏng chỉ được làm tắt tính năng, không được ném lỗi lên API."""
    bad = tmp_path / "hong.pkl"
    bad.write_bytes(b"day khong phai pickle")
    svc = RankerService(bad)
    assert svc.is_available is False
    assert svc.predict(make_match(), JD) is None


def test_nap_duoc_model_hop_le(tmp_path):
    svc = RankerService(write_bundle(tmp_path / "m.pkl", EchoFirstFeature()))
    assert svc.is_available is True
    assert svc.model_key == "fake"


# ---------------------------------------------------------------- dự đoán

def test_dung_dung_thu_tu_feature_luu_trong_file(tmp_path):
    """Thứ tự feature phải đọc TỪ FILE. Nếu hardcode mà train_ranker đổi thứ tự
    thì model nhận sai đầu vào và chấm sai một cách âm thầm."""
    thuan = RankerService(write_bundle(tmp_path / "a.pkl", EchoFirstFeature()))
    dao_nguoc = RankerService(
        write_bundle(tmp_path / "b.pkl", EchoFirstFeature(), features=list(reversed(FEATURES)))
    )
    match = make_match(skills=0.8, exp=0.6, edu=1.0)

    # thuận: feature đầu là skill_score = 0.8 -> 8.0
    assert thuan.predict(match, JD) == 8.0
    # đảo: feature đầu là n_require_skills = 2 -> 20.0
    assert dao_nguoc.predict(match, JD) == 20.0


def test_gia_tri_feature_dung(tmp_path):
    """Tổng 9 feature: 0.8+0.6+1.0 + 0.5+0.3+0.2 + 3(năm) + 3(bằng) + 2(kỹ năng)"""
    svc = RankerService(write_bundle(tmp_path / "m.pkl", SumFeatures()))
    assert svc.predict(make_match(skills=0.8, exp=0.6, edu=1.0), JD) == pytest.approx(11.4)


def test_cat_bien_ve_thang_diem(tmp_path):
    for raw, expected in ((150.0, 100.0), (-20.0, 0.0), (73.4, 73.4)):
        svc = RankerService(write_bundle(tmp_path / f"m{raw}.pkl", ConstantModel(raw)))
        assert svc.predict(make_match(), JD) == expected


def test_model_thieu_feature_thi_tra_none(tmp_path):
    svc = RankerService(
        write_bundle(tmp_path / "m.pkl", EchoFirstFeature(), features=["khong_ton_tai"])
    )
    assert svc.predict(make_match(), JD) is None


def test_model_nem_loi_thi_tra_none(tmp_path):
    svc = RankerService(write_bundle(tmp_path / "m.pkl", ExplodingModel()))
    assert svc.predict(make_match(), JD) is None


# ---------------------------------------------------------------- bộ lọc rẻ

class FakeCVParser:
    def __init__(self):
        self.count = 0

    def parse(self, text):
        self.count += 1
        return ParseCV(skills=[text])


class FakeJDParser:
    def parse(self, text):
        return JD


class FakeMatcher:
    def match(self, cv, jd, cv_id, jd_id):
        # điểm kỹ năng suy từ tên file để test kiểm soát được thứ hạng
        score = float(cv_id.split("_")[1]) / 100
        return MatchResult(
            cv_id=cv_id, jd_id=jd_id,
            criterion_scores=[
                CriterionScore(criterion="skills", score=score),
                CriterionScore(criterion="experience", score=0.5),
                CriterionScore(criterion="education", score=1.0),
            ],
            rule_based_score=score * 100,
        )

    def skill_gap_impact(self, cv, jd, top_n=5):
        return []


class CountingScorer:
    def __init__(self):
        self.count = 0

    def evaluate(self, cv, jd, match_result):
        from app.schemas.models import CandidateEvaluation

        self.count += 1
        return CandidateEvaluation(
            cv_id=match_result.cv_id, jd_id=match_result.jd_id,
            final_score=match_result.rule_based_score,
            strengths=["a"], gaps=["b"], explanation="llm",
        )


@pytest.fixture
def pipeline(monkeypatch, tmp_path) -> RecruitmentPipeline:
    monkeypatch.setattr("app.services.pipeline.get_llm_client", lambda s: object())
    monkeypatch.setattr("app.services.pipeline.get_embedding_client", lambda s: object())

    settings = ModelSettings()
    settings.ranker_model_path = str(write_bundle(tmp_path / "m.pkl", FakeModel()))
    p = RecruitmentPipeline(settings)
    p.cv_parser = FakeCVParser()
    p.jd_parser = FakeJDParser()
    p.matcher = FakeMatcher()
    p.scorer = CountingScorer()
    return p


def cv_batch(n: int) -> dict[str, str]:
    # cv_10_.. cv_20_.. -> skill_score 0.10, 0.20, ...
    return {f"cv_{(i + 1) * 10}_.pdf": f"noi dung {i}" for i in range(n)}


def test_tat_bo_loc_thi_llm_cham_het(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 0
    results = pipeline.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")

    assert pipeline.scorer.count == 5
    assert all(r.evaluation.scored_by == "llm" for r in results)


def test_bat_bo_loc_thi_llm_chi_cham_top_k(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 2
    results = pipeline.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")

    assert pipeline.scorer.count == 2, "LLM phải chỉ chấm đúng top-K"
    assert len(results) == 5, "vẫn phải trả về đủ mọi ứng viên"
    assert pipeline.cv_parser.count == 5, "parse vẫn cần cho mọi CV để có feature"


def test_ho_so_bi_loc_duoc_danh_dau_ro(pipeline):
    """Không được để HR tưởng mọi điểm số đều đã qua AI ngôn ngữ."""
    pipeline.settings.ranker_prefilter_top_k = 2
    results = pipeline.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")

    by_source = {r.evaluation.cv_id: r.evaluation.scored_by for r in results}
    assert sum(v == "llm" for v in by_source.values()) == 2
    assert sum(v == "model" for v in by_source.values()) == 3

    filtered = [r for r in results if r.evaluation.scored_by == "model"]
    assert all(r.evaluation.strengths == [] for r in filtered)
    assert all("mô hình" in r.evaluation.explanation for r in filtered)


def test_dung_top_k_diem_cao_nhat_duoc_cham_bang_llm(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 2
    results = pipeline.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")

    llm_scored = {r.evaluation.cv_id for r in results if r.evaluation.scored_by == "llm"}
    assert llm_scored == {"cv_50_.pdf", "cv_40_.pdf"}


def test_it_ho_so_hon_top_k_thi_khong_loc_ai(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 10
    pipeline.rank_candidates(cv_batch(3), jd_raw_text="JD", jd_id="JD")
    assert pipeline.scorer.count == 3


def test_khong_co_model_thi_bo_loc_tu_tat(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.get_llm_client", lambda s: object())
    monkeypatch.setattr("app.services.pipeline.get_embedding_client", lambda s: object())

    settings = ModelSettings()
    settings.ranker_model_path = ""          # không có model
    settings.ranker_prefilter_top_k = 2      # bật lọc nhưng vô hiệu

    p = RecruitmentPipeline(settings)
    p.cv_parser, p.jd_parser = FakeCVParser(), FakeJDParser()
    p.matcher, p.scorer = FakeMatcher(), CountingScorer()

    results = p.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")
    assert p.scorer.count == 5, "thiếu model thì phải quay về chấm LLM cho tất cả"
    assert all(r.model_score is None for r in results)


def test_luon_tra_model_score_de_doi_chieu(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 0
    results = pipeline.rank_candidates(cv_batch(3), jd_raw_text="JD", jd_id="JD")
    assert all(r.model_score is not None for r in results)


def test_ket_qua_van_xep_hang_giam_dan(pipeline):
    pipeline.settings.ranker_prefilter_top_k = 2
    results = pipeline.rank_candidates(cv_batch(5), jd_raw_text="JD", jd_id="JD")
    scores = [r.evaluation.final_score for r in results]
    assert scores == sorted(scores, reverse=True)
