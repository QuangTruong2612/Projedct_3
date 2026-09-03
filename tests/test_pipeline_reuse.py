"""Test hai thứ của Nhóm 1:

1. get_pipeline() chỉ dựng pipeline MỘT lần cho cả tiến trình
   (trước đây mỗi request dựng lại -> nạp lại model embedding ~1.1GB).
2. rank_candidates() chỉ parse JD 1 lần và xử lý các CV song song.

Không gọi LLM/model thật — dùng đối tượng giả có đếm số lần gọi.
"""

import threading
import time

import pytest

from app.api import routers
from app.schemas.models import (
    CandidateEvaluation,
    CriterionScore,
    MatchResult,
    ParseCV,
    ParseJD,
)
from app.services.pipeline import RecruitmentPipeline


# ---------- 1. pipeline dùng chung ----------

def test_get_pipeline_chi_dung_mot_lan(monkeypatch):
    build_count = {"n": 0}

    class FakePipeline:
        def __init__(self, settings=None):
            build_count["n"] += 1

    monkeypatch.setattr(routers, "RecruitmentPipeline", FakePipeline)
    routers.reset_pipeline()

    first = routers.get_pipeline()
    second = routers.get_pipeline()
    third = routers.get_pipeline()

    assert build_count["n"] == 1, "Pipeline bị dựng lại nhiều lần"
    assert first is second is third

    routers.reset_pipeline()


def test_get_pipeline_an_toan_khi_nhieu_luong_goi_cung_luc(monkeypatch):
    """Endpoint chạy trong threadpool nên nhiều request đầu tiên có thể
    cùng vào get_pipeline() -> vẫn chỉ được dựng đúng 1 lần."""
    build_count = {"n": 0}

    class SlowFakePipeline:
        def __init__(self, settings=None):
            time.sleep(0.05)  # mô phỏng thời gian nạp model
            build_count["n"] += 1

    monkeypatch.setattr(routers, "RecruitmentPipeline", SlowFakePipeline)
    routers.reset_pipeline()

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(routers.get_pipeline()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 1
    assert all(r is results[0] for r in results)

    routers.reset_pipeline()


# ---------- 2. rank_candidates ----------

class FakeCVParser:
    def __init__(self):
        self.count = 0

    def parse(self, text: str) -> ParseCV:
        self.count += 1
        time.sleep(0.1)  # mô phỏng độ trễ gọi LLM
        return ParseCV(full_name=text, skills=["Python"], total_experience_year=2.0)


class FakeJDParser:
    def __init__(self):
        self.count = 0

    def parse(self, text: str) -> ParseJD:
        self.count += 1
        return ParseJD(job_title="Dev", require_skills=["Python"])


class FakeMatcher:
    def match(self, cv, jd, cv_id, jd_id) -> MatchResult:
        return MatchResult(
            cv_id=cv_id,
            jd_id=jd_id,
            criterion_scores=[CriterionScore(criterion="skills", score=1.0)],
            rule_based_score=80.0,
        )

    def skill_gap_impact(self, cv, jd, top_n=5):
        return []


class FakeScorer:
    def evaluate(self, cv, jd, match_result) -> CandidateEvaluation:
        time.sleep(0.1)  # mô phỏng độ trễ gọi LLM
        # điểm suy ra từ tên file để test kiểm tra được thứ tự xếp hạng
        return CandidateEvaluation(
            cv_id=match_result.cv_id,
            jd_id=match_result.jd_id,
            final_score=float(len(match_result.cv_id)),
            strengths=["a", "b"],
            gaps=["c", "d"],
            explanation="ok",
        )


@pytest.fixture
def pipeline(monkeypatch) -> RecruitmentPipeline:
    monkeypatch.setattr(
        "app.services.pipeline.get_llm_client", lambda settings: object()
    )
    monkeypatch.setattr(
        "app.services.pipeline.get_embedding_client", lambda settings: object()
    )
    p = RecruitmentPipeline()
    p.cv_parser = FakeCVParser()
    p.jd_parser = FakeJDParser()
    p.matcher = FakeMatcher()
    p.scorer = FakeScorer()
    return p


def test_jd_chi_parse_mot_lan_cho_ca_lo(pipeline):
    cv_texts = {f"cv{i}.pdf": f"noi dung {i}" for i in range(4)}
    pipeline.rank_candidates(cv_texts, jd_raw_text="JD", jd_id="JD-01")

    assert pipeline.jd_parser.count == 1
    assert pipeline.cv_parser.count == 4


def test_cac_cv_duoc_xu_ly_song_song(pipeline):
    """4 CV, mỗi CV tốn ~0.2s (parse 0.1 + score 0.1).
    Tuần tự sẽ mất ~0.8s; song song 4 luồng phải nhanh hơn hẳn."""
    pipeline.settings.rank_max_workers = 4
    cv_texts = {f"cv{i}.pdf": f"noi dung {i}" for i in range(4)}

    start = time.perf_counter()
    results = pipeline.rank_candidates(cv_texts, jd_raw_text="JD", jd_id="JD-01")
    elapsed = time.perf_counter() - start

    assert len(results) == 4
    assert elapsed < 0.5, f"Có vẻ vẫn chạy tuần tự (mất {elapsed:.2f}s)"


def test_ket_qua_duoc_xep_hang_giam_dan(pipeline):
    cv_texts = {
        "ngan.pdf": "a",
        "ten_file_dai_hon.pdf": "b",
        "trung_binh.pdf": "c",
    }
    results = pipeline.rank_candidates(cv_texts, jd_raw_text="JD", jd_id="JD-01")

    scores = [r.evaluation.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_mot_cv_loi_thi_bao_loi_nhu_cu(pipeline):
    """Giữ nguyên hành vi cũ: 1 CV hỏng -> cả request báo lỗi,
    không im lặng bỏ qua ứng viên."""

    class ExplodingScorer(FakeScorer):
        def evaluate(self, cv, jd, match_result):
            if match_result.cv_id == "loi.pdf":
                raise RuntimeError("LLM that bai")
            return super().evaluate(cv, jd, match_result)

    pipeline.scorer = ExplodingScorer()
    cv_texts = {"ok.pdf": "a", "loi.pdf": "b"}

    with pytest.raises(RuntimeError, match="LLM that bai"):
        pipeline.rank_candidates(cv_texts, jd_raw_text="JD", jd_id="JD-01")


def test_mot_cv_duy_nhat_van_chay_dung(pipeline):
    results = pipeline.rank_candidates(
        {"cv.pdf": "a"}, jd_raw_text="JD", jd_id="JD-01"
    )
    assert len(results) == 1
    assert results[0].rule_based_score == 80.0


# ---------- 3. không parse lại lần thứ hai ----------

def test_run_tra_kem_breakdown_va_du_lieu_da_parse(pipeline):
    """Hồi quy cho lỗi cũ: run_distillation.py phải gọi parse LẦN HAI chỉ để
    lấy điểm từng tiêu chí -> tốn gấp đôi chi phí API, và hai lần parse có thể
    ra kết quả khác nhau khiến feature không khớp rule_based_score."""
    result = pipeline.run(
        cv_raw_text="noi dung cv", jd_raw_text="JD", cv_id="cv.pdf", jd_id="JD-01"
    )

    assert [c.criterion for c in result.criterion_scores] == ["skills"]
    assert result.parsed_cv is not None
    assert result.parsed_jd is not None
    # đúng 1 lần parse CV và 1 lần parse JD cho cả lượt chạy
    assert pipeline.cv_parser.count == 1
    assert pipeline.jd_parser.count == 1
