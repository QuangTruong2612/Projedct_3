"""Test cho việc soi chất lượng JD.

Toàn bộ là luật cứng trên ParsedJD nên không cần LLM lẫn embedding.
"""

import pytest

from app.schemas.models import ParseJD
from app.services.jd_quality_service import JDQualityService


@pytest.fixture
def svc() -> JDQualityService:
    return JDQualityService()


def make_jd(**kw) -> ParseJD:
    base = dict(
        job_title="Backend Developer",
        require_skills=["Python", "PostgreSQL", "Docker", "REST API", "Git"],
        preferred_skills=["Kubernetes", "Kafka"],
        min_experience_years=2.0,
        required_degree="Bachelor",
    )
    base.update(kw)
    return ParseJD(**base)


def codes(report) -> set[str]:
    return {w.code for w in report.warnings}


def test_jd_tot_thi_khong_co_canh_bao(svc):
    report = svc.check(make_jd())
    assert report.warnings == []
    assert report.is_healthy is True


def test_qua_nhieu_ky_nang_bat_buoc(svc):
    """Đây là ca quan trọng nhất: JD 19 kỹ năng bắt buộc khiến MỌI ứng viên
    đều độ phủ thấp -> mọi hồ sơ đều điểm thấp, lỗi ở JD chứ không ở ứng viên."""
    jd = make_jd(require_skills=[f"Kỹ năng {i}" for i in range(19)])
    report = svc.check(jd)

    assert "too_many_required_skills" in codes(report)
    assert report.is_healthy is False
    warning = next(w for w in report.warnings if w.code == "too_many_required_skills")
    assert "19" in warning.message
    assert warning.suggestion


def test_khong_co_ky_nang_bat_buoc(svc):
    report = svc.check(make_jd(require_skills=[], preferred_skills=[]))
    assert "no_required_skills" in codes(report)
    assert report.is_healthy is False


def test_qua_it_ky_nang_bat_buoc(svc):
    report = svc.check(make_jd(require_skills=["Python", "SQL"]))
    assert "too_few_required_skills" in codes(report)


def test_khong_tach_ky_nang_uu_tien(svc):
    report = svc.check(make_jd(preferred_skills=[]))
    assert "no_preferred_skills" in codes(report)
    assert report.is_healthy is True  # chỉ là cảnh báo nhẹ


@pytest.mark.parametrize("title,years", [
    ("Junior Backend Developer", 6.0),
    ("Fresher Data Analyst", 4.0),
    ("Intern Mobile Developer", 3.0),
    ("Senior AI Engineer", 0.0),
])
def test_cap_bac_khong_khop_so_nam(svc, title, years):
    report = svc.check(make_jd(job_title=title, min_experience_years=years))
    assert "seniority_mismatch" in codes(report)


@pytest.mark.parametrize("title,years", [
    ("Junior Backend Developer", 1.0),
    ("Senior AI Engineer", 5.0),
    ("Fresher Data Analyst", 0.0),
    ("Intern Mobile Developer", 0.0),
])
def test_cap_bac_khop_so_nam_thi_khong_canh_bao(svc, title, years):
    report = svc.check(make_jd(job_title=title, min_experience_years=years))
    assert "seniority_mismatch" not in codes(report)


def test_so_nam_bat_thuong(svc):
    report = svc.check(make_jd(job_title="Backend Developer", min_experience_years=25.0))
    assert "unrealistic_experience" in codes(report)


def test_thieu_chuc_danh(svc):
    report = svc.check(make_jd(job_title=None))
    assert "no_job_title" in codes(report)
    assert report.is_healthy is False


def test_trong_so_khong_bang_1(svc):
    jd = make_jd(weights={"skills": 0.5, "experience": 0.3, "education": 0.5})
    assert "weights_not_normalized" in codes(svc.check(jd))


def test_ky_nang_trung_o_ca_hai_muc(svc):
    jd = make_jd(
        require_skills=["Python", "Docker", "SQL", "Git"],
        preferred_skills=["docker", "Kafka"],  # khác hoa thường vẫn tính là trùng
    )
    report = svc.check(jd)
    assert "duplicated_skills" in codes(report)


def test_bao_cao_kem_so_lieu_tong_quan(svc):
    report = svc.check(make_jd())
    assert report.n_require_skills == 5
    assert report.n_preferred_skills == 2
    assert report.min_experience_years == 2.0
    assert report.job_title == "Backend Developer"


def test_moi_canh_bao_deu_co_de_xuat_sua(svc):
    """Cảnh báo mà không nói cách sửa thì HR không dùng được."""
    jd = make_jd(job_title=None, require_skills=[], preferred_skills=[], min_experience_years=99.0)
    for w in svc.check(jd).warnings:
        assert w.suggestion, f"Cảnh báo {w.code} thiếu đề xuất sửa"
        assert w.severity in {"high", "medium", "low"}
