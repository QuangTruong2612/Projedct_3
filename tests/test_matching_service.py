"""Test cho MatchingService sau khi hiệu chỉnh công thức chấm điểm.

Dùng embedder giả để kiểm soát hoàn toàn similarity -> test được đúng phần
LOGIC chấm điểm mà không cần nạp model thật.
"""

import pytest

from app.core.settings import ModelSettings
from app.schemas.models import Education, ExperienceWork, ParseCV, ParseJD, Project
from app.services.matching_service import MatchingService, _lexical_match, _normalize


class DictEmbedder:
    """Embedder giả: mỗi văn bản là một vector one-hot theo bảng chỉ định,
    nhờ đó cosine giữa 2 văn bản chính là giá trị ta đặt trong `sims`."""

    def __init__(self, sims: dict[tuple[str, str], float] | None = None):
        self.sims = sims or {}
        self._ids: dict[str, int] = {}

    def _id(self, text: str) -> int:
        if text not in self._ids:
            self._ids[text] = len(self._ids)
        return self._ids[text]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Vector = [id, marker]; cosine thật sẽ được ghi đè bởi monkeypatch bên dưới
        return [[float(self._id(t)), 1.0] for t in texts]


@pytest.fixture
def svc(monkeypatch):
    """MatchingService với cosine_similarity được thay bằng bảng tra cứu."""
    embedder = DictEmbedder()
    settings = ModelSettings()
    settings.similarity_floor = 0.80
    settings.experience_relevance_base = 0.3
    settings.skill_match_threshold = 0.85

    table: dict[tuple[float, float], float] = {}

    def fake_cosine(a, b):
        key = (a[0], b[0])
        if a[0] == b[0]:
            return 1.0
        return table.get(key, table.get((b[0], a[0]), 0.0))

    monkeypatch.setattr("app.services.matching_service.cosine_similarity", fake_cosine)

    service = MatchingService(embedder, settings)
    service._table = table  # type: ignore[attr-defined]
    service._embedder_ids = embedder._ids  # type: ignore[attr-defined]

    def set_sim(a: str, b: str, value: float) -> None:
        embedder.embed([a, b])  # đảm bảo đã có id
        table[(float(embedder._id(a)), float(embedder._id(b)))] = value

    service.set_sim = set_sim  # type: ignore[attr-defined]
    return service


def make_jd(**kw) -> ParseJD:
    base = dict(job_title="Backend Developer", require_skills=[], preferred_skills=[],
                min_experience_years=0, required_degree=None)
    base.update(kw)
    return ParseJD(**base)


def make_cv(**kw) -> ParseCV:
    base = dict(skills=[], total_experience_year=0.0, work_experiences=[], projects=[], educations=[])
    base.update(kw)
    return ParseCV(**base)


# ---------------------------------------------------------------- chuẩn hoá chữ

@pytest.mark.parametrize("raw,expected", [
    ("React.js", "reactjs"),
    ("Node JS", "nodejs"),
    ("PostgreSQL", "postgresql"),
    ("Kỹ thuật phần mềm", "kythuatphanmem"),
    ("CI/CD", "cicd"),
])
def test_normalize(raw, expected):
    assert _normalize(raw) == expected


@pytest.mark.parametrize("a,b", [
    ("React.js", "ReactJS"),
    ("Postgres", "PostgreSQL"),
    ("Node JS", "Node.js"),
    ("Docker", "docker "),
])
def test_lexical_match_bat_duoc_bien_the_viet(a, b):
    assert _lexical_match(a, b)


@pytest.mark.parametrize("a,b", [
    ("Python", "Java"),
    ("R", "React"),        # chuỗi quá ngắn -> không cho khớp kiểu chứa
    ("Go", "MongoDB"),     # tránh khớp bừa
    ("", "Python"),
])
def test_lexical_match_khong_khop_bua(a, b):
    assert not _lexical_match(a, b)


# ---------------------------------------------------------------- kỹ năng

def test_khop_chu_duoc_tinh_tron_diem(svc):
    cv = make_cv(skills=["React.js"])
    jd = make_jd(require_skills=["ReactJS"])
    cs = svc._score_skills(cv, jd)
    # 0.8 * 1.0 (bắt buộc) + 0.2 * 1.0 (không có ưu tiên -> mặc định 1.0)
    assert cs.score == pytest.approx(1.0)
    assert "1/1" in cs.detail


def test_similarity_duoi_san_nhieu_bi_dua_ve_0(svc):
    cv = make_cv(skills=["Photoshop"])
    jd = make_jd(require_skills=["Kubernetes"])
    svc.set_sim("Kubernetes", "Photoshop", 0.79)  # dưới sàn 0.80
    cs = svc._score_skills(cv, jd)
    assert cs.score == pytest.approx(0.2)  # chỉ còn phần preferred rỗng = 1.0 * 0.2


def test_similarity_tren_san_duoc_keo_gian(svc):
    cv = make_cv(skills=["Postgres DB"])
    jd = make_jd(require_skills=["Kubernetes"])
    svc.set_sim("Kubernetes", "Postgres DB", 0.90)  # (0.90-0.80)/0.20 = 0.5
    cs = svc._score_skills(cv, jd)
    assert cs.score == pytest.approx(0.8 * 0.5 + 0.2)


def test_diem_lien_tuc_chu_khong_nhi_phan(svc):
    """Điểm phải phân biệt được 0.84 và 0.20 -- cách đếm theo ngưỡng cũ thì không."""
    jd = make_jd(require_skills=["Kubernetes"])

    cv_gan = make_cv(skills=["Container orchestration"])
    svc.set_sim("Kubernetes", "Container orchestration", 0.84)
    diem_gan = svc._score_skills(cv_gan, jd).score

    cv_xa = make_cv(skills=["Photoshop"])
    svc.set_sim("Kubernetes", "Photoshop", 0.20)
    diem_xa = svc._score_skills(cv_xa, jd).score

    assert diem_gan > diem_xa


def test_ky_nang_trong_project_duoc_tinh(svc):
    cv = make_cv(
        skills=[],
        projects=[Project(project_title="P", tech_stack=["Docker"], is_academic=False)],
    )
    jd = make_jd(require_skills=["Docker"])
    assert svc._score_skills(cv, jd).score == pytest.approx(1.0)


def test_jd_khong_yeu_cau_ky_nang_thi_du_diem(svc):
    assert svc._score_skills(make_cv(skills=["X"]), make_jd()).score == pytest.approx(1.0)


def test_cv_khong_co_ky_nang_thi_diem_0_phan_bat_buoc(svc):
    cs = svc._score_skills(make_cv(), make_jd(require_skills=["Docker"]))
    assert cs.score == pytest.approx(0.2)


# ---------------------------------------------------------------- kinh nghiệm

def test_kinh_nghiem_trai_nganh_bi_phat_nang(svc):
    """Ca thực tế: kế toán 2.2 năm ứng tuyển QA (yêu cầu 0.5 năm).
    Công thức cũ cho 0.93; công thức nhân phải cho thấp hơn hẳn."""
    cv = make_cv(
        total_experience_year=2.2,
        work_experiences=[ExperienceWork(job_title="Kế toán viên", duration_work=2.2)],
    )
    jd = make_jd(job_title="QA Engineer", min_experience_years=0.5)
    svc.set_sim("QA Engineer", "Kế toán viên", 0.79)  # trái ngành -> dưới sàn

    cs = svc._score_experience(cv, jd)
    assert cs.score == pytest.approx(0.3)  # = years_score(1.0) * base(0.3)


def test_kinh_nghiem_dung_nganh_duoc_diem_cao(svc):
    cv = make_cv(
        total_experience_year=3.0,
        work_experiences=[ExperienceWork(job_title="QA Engineer", duration_work=3.0)],
    )
    jd = make_jd(job_title="QA Engineer", min_experience_years=2.0)
    cs = svc._score_experience(cv, jd)
    assert cs.score == pytest.approx(1.0)  # cùng chức danh -> cosine 1.0 -> gate 1.0


def test_thieu_nam_kinh_nghiem_van_bi_tru(svc):
    cv = make_cv(
        total_experience_year=1.0,
        work_experiences=[ExperienceWork(job_title="QA Engineer", duration_work=1.0)],
    )
    jd = make_jd(job_title="QA Engineer", min_experience_years=4.0)
    assert svc._score_experience(cv, jd).score == pytest.approx(0.25)  # 1/4 * gate 1.0


def test_ung_vien_moi_ra_truong_duoc_project_ganh(svc):
    """Dưới 1 năm kinh nghiệm -> project được dùng thay cho chức danh."""
    cv = make_cv(
        total_experience_year=0.5,
        work_experiences=[ExperienceWork(job_title="Thực tập sinh", duration_work=0.5)],
        projects=[Project(project_title="API service", description="REST API", tech_stack=["Go"])],
    )
    jd = make_jd(job_title="Backend Developer", require_skills=["Go"], min_experience_years=0)
    svc.set_sim("Backend Developer", "Thực tập sinh", 0.80)  # chức danh: không liên quan
    svc.set_sim(
        "Backend Developer. Yêu cầu: Go",
        "API service: REST API (Công nghệ: Go)",
        1.0,
    )
    cs = svc._score_experience(cv, jd)
    assert cs.score == pytest.approx(1.0)
    assert "project" in cs.detail


def test_jd_khong_yeu_cau_so_nam_thi_chi_xet_do_lien_quan(svc):
    cv = make_cv(
        total_experience_year=10.0,
        work_experiences=[ExperienceWork(job_title="Đầu bếp", duration_work=10.0)],
    )
    jd = make_jd(job_title="Backend Developer", min_experience_years=0)
    svc.set_sim("Backend Developer", "Đầu bếp", 0.75)
    assert svc._score_experience(cv, jd).score == pytest.approx(0.3)


# ---------------------------------------------------------------- học vấn

def test_hoc_van_du_bang_thi_tron_diem(svc):
    cv = make_cv(educations=[Education(degree="Bachelor", major="CNTT")])
    assert svc._score_education(cv, make_jd(required_degree="Bachelor")).score == 1.0


def test_hoc_van_khong_xet_chuyen_nganh(svc):
    """Quyết định CÓ CHỦ ĐÍCH (xem docstring _score_education): ngành khác
    nhau nhưng cùng cấp bằng thì điểm bằng nhau."""
    cntt = make_cv(educations=[Education(degree="Bachelor", major="Công nghệ Thông tin")])
    do_hoa = make_cv(educations=[Education(degree="Bachelor", major="Thiết kế Đồ họa")])
    jd = make_jd(required_degree="Bachelor")
    assert svc._score_education(cntt, jd).score == svc._score_education(do_hoa, jd).score


def test_hoc_van_thap_hon_yeu_cau_bi_tru(svc):
    cv = make_cv(educations=[Education(degree="Associate", major="CNTT")])
    # điểm được làm tròn 3 chữ số trong service
    assert svc._score_education(cv, make_jd(required_degree="Bachelor")).score == pytest.approx(2 / 3, abs=1e-3)


def test_lay_bang_cao_nhat(svc):
    cv = make_cv(educations=[
        Education(degree="Bachelor", major="A"),
        Education(degree="Master", major="B"),
    ])
    assert svc._score_education(cv, make_jd(required_degree="Master")).score == 1.0


def test_jd_khong_yeu_cau_bang_cap(svc):
    assert svc._score_education(make_cv(), make_jd()).score == 1.0


def test_cv_khong_co_hoc_van(svc):
    assert svc._score_education(make_cv(), make_jd(required_degree="Bachelor")).score == 0.0


# ---------------------------------------------------------------- tổng hợp

def test_match_tong_hop_theo_trong_so_cua_jd(svc):
    cv = make_cv(
        skills=["Docker"],
        total_experience_year=2.0,
        work_experiences=[ExperienceWork(job_title="Backend Developer", duration_work=2.0)],
        educations=[Education(degree="Bachelor", major="CNTT")],
    )
    jd = make_jd(
        require_skills=["Docker"],
        min_experience_years=2.0,
        required_degree="Bachelor",
        weights={"skills": 0.5, "experience": 0.3, "education": 0.2},
    )
    result = svc.match(cv, jd, cv_id="cv1", jd_id="jd1")

    assert result.rule_based_score == pytest.approx(100.0)
    assert [c.criterion for c in result.criterion_scores] == ["skills", "experience", "education"]


# ---------------------------------------------------------------- thiếu dữ liệu

def test_cv_khong_co_hoc_van_duoc_danh_dau_thieu_du_lieu(svc):
    """Điểm 0 vì THIẾU DỮ LIỆU phải phân biệt được với điểm 0 vì KHÔNG ĐẠT.
    Trước đây cả hai đều trả 0.0 và HR không có cách nào biết."""
    cs = svc._score_education(make_cv(), make_jd(required_degree="Bachelor"))
    assert cs.score == 0.0
    assert cs.missing_data is True


def test_bang_cap_thap_hon_yeu_cau_khong_phai_thieu_du_lieu(svc):
    cv = make_cv(educations=[Education(degree="Associate", major="CNTT")])
    cs = svc._score_education(cv, make_jd(required_degree="Bachelor"))
    assert cs.missing_data is False  # có dữ liệu, chỉ là chưa đạt


def test_cv_khong_trich_duoc_ky_nang_bi_danh_dau(svc):
    cs = svc._score_skills(make_cv(), make_jd(require_skills=["Docker"]))
    assert cs.missing_data is True
    assert "scan ảnh" in cs.detail


def test_cv_khong_co_kinh_nghiem_lan_project_bi_danh_dau(svc):
    cs = svc._score_experience(make_cv(), make_jd(min_experience_years=2))
    assert cs.missing_data is True


def test_cv_day_du_thi_khong_co_canh_bao(svc):
    cv = make_cv(
        skills=["Docker"],
        total_experience_year=2.0,
        work_experiences=[ExperienceWork(job_title="Backend Developer", duration_work=2.0)],
        educations=[Education(degree="Bachelor", major="CNTT")],
    )
    c = svc._completeness(cv)
    assert c.score == 1.0
    assert c.warnings == []
    assert c.is_reliable is True


def test_cv_rong_hoan_toan_bi_canh_bao_du_ba_muc(svc):
    c = svc._completeness(make_cv())
    assert c.score == 0.0
    assert len(c.warnings) == 3
    assert c.is_reliable is False


def test_ky_nang_tu_project_van_tinh_la_co_du_lieu(svc):
    cv = make_cv(projects=[Project(project_title="P", tech_stack=["Docker"])])
    c = svc._completeness(cv)
    assert c.has_skills is True
    assert c.has_experience is True   # project cũng là bằng chứng kinh nghiệm
    assert c.has_education is False


def test_match_tra_kem_completeness(svc):
    result = svc.match(make_cv(), make_jd(require_skills=["Docker"]), cv_id="c", jd_id="j")
    assert result.completeness is not None
    assert result.completeness.score == 0.0
    assert any(c.missing_data for c in result.criterion_scores)


# ---------------------------------------------------------------- phân tích phản thực

def test_ky_nang_hut_nhieu_nhat_duoc_xep_dau(svc):
    """Kỹ năng nào bù vào làm điểm tăng nhiều nhất phải đứng đầu danh sách."""
    jd = make_jd(
        require_skills=["Docker", "Kubernetes", "Kafka"],
        weights={"skills": 1.0, "experience": 0.0, "education": 0.0},
    )
    svc.set_sim("Kubernetes", "Docker", 0.90)  # hụt một nửa: (0.90-0.80)/0.20 = 0.5
    svc.set_sim("Kafka", "Docker", 0.80)       # hụt hoàn toàn

    gaps = svc.skill_gap_impact(make_cv(skills=["Docker"]), jd)
    assert [g.skill for g in gaps] == ["Kafka", "Kubernetes"]
    assert gaps[0].score_gain > gaps[1].score_gain


def test_ky_nang_da_dap_ung_khong_xuat_hien(svc):
    cv = make_cv(skills=["Docker"])
    jd = make_jd(require_skills=["Docker", "Kafka"])
    svc.set_sim("Kafka", "Docker", 0.80)

    assert "Docker" not in [g.skill for g in svc.skill_gap_impact(cv, jd)]


def test_muc_tang_diem_tinh_dung_cong_thuc(svc):
    """1 trong 2 kỹ năng bắt buộc hụt hoàn toàn, trọng số skills = 0.5.
    Bù vào -> điểm tổng tăng 0.5 x 0.8 x 1.0 / 2 x 100 = 20 điểm."""
    cv = make_cv(skills=["Docker"])
    jd = make_jd(
        require_skills=["Docker", "Kafka"],
        weights={"skills": 0.5, "experience": 0.3, "education": 0.2},
    )
    svc.set_sim("Kafka", "Docker", 0.80)

    gaps = svc.skill_gap_impact(cv, jd)
    assert gaps[0].skill == "Kafka"
    assert gaps[0].score_gain == pytest.approx(20.0)


def test_muc_tang_khop_voi_diem_thuc_te_khi_bo_sung(svc):
    """Kiểm chứng chéo: con số dự đoán phải khớp với việc thật sự thêm kỹ năng
    vào CV rồi chấm lại."""
    cv = make_cv(skills=["Docker"])
    jd = make_jd(
        require_skills=["Docker", "Kafka"],
        weights={"skills": 0.5, "experience": 0.3, "education": 0.2},
    )
    svc.set_sim("Kafka", "Docker", 0.80)

    truoc = svc.match(cv, jd, cv_id="c", jd_id="j").rule_based_score
    du_doan = svc.skill_gap_impact(cv, jd)[0].score_gain

    cv_bo_sung = make_cv(skills=["Docker", "Kafka"])
    sau = svc.match(cv_bo_sung, jd, cv_id="c", jd_id="j").rule_based_score

    assert sau - truoc == pytest.approx(du_doan, abs=0.2)


def test_ky_nang_uu_tien_cung_duoc_tinh_nhung_trong_so_thap_hon(svc):
    cv = make_cv(skills=["Docker"])
    jd = make_jd(
        require_skills=["Docker"],
        preferred_skills=["Kafka"],
        weights={"skills": 1.0, "experience": 0.0, "education": 0.0},
    )
    svc.set_sim("Kafka", "Docker", 0.80)

    gaps = svc.skill_gap_impact(cv, jd)
    assert len(gaps) == 1
    assert gaps[0].skill == "Kafka"
    assert gaps[0].is_required is False


def test_gioi_han_so_luong_tra_ve(svc):
    cv = make_cv(skills=[])
    jd = make_jd(require_skills=[f"KN{i}" for i in range(10)])
    assert len(svc.skill_gap_impact(cv, jd, top_n=3)) == 3


def test_jd_khong_yeu_cau_ky_nang_thi_khong_co_gap(svc):
    assert svc.skill_gap_impact(make_cv(skills=["X"]), make_jd()) == []


# ---------------------------------------------------------------- kỹ năng trùng trong JD

def test_jd_viet_lap_ky_nang_khong_lam_thay_doi_diem(svc):
    """JD ghi "Python" hai lần là lỗi soạn thảo, không được phép trừ điểm
    ứng viên. Trước khi sửa: per_skill gộp còn 2 mục nhưng mẫu số vẫn đếm 3
    -> ratio 0.333 thay vì 0.5."""
    cv = make_cv(skills=["Python"])
    khong_lap = make_jd(require_skills=["Python", "SQL"])
    co_lap = make_jd(require_skills=["Python", "Python", "SQL"])

    assert svc._score_skills(cv, khong_lap).score == svc._score_skills(cv, co_lap).score


def test_khu_trung_lap_khong_phan_biet_hoa_thuong(svc):
    cov = svc._skill_coverage(["Python"], ["Python", "python", "PYTHON", "SQL"])
    assert len(cov.per_skill) == 2
    assert cov.ratio == pytest.approx(0.5)
    assert cov.clear_matches == 1


def test_khong_gop_nham_cpp_voi_c(svc):
    """Khử trùng lặp dùng .lower() chứ không dùng _normalize(), nếu không
    "C++" và "C" sẽ bị gộp làm một."""
    cov = svc._skill_coverage(["Java"], ["C++", "C"])
    assert len(cov.per_skill) == 2


def test_muc_tang_diem_van_dung_khi_jd_co_ky_nang_lap(svc):
    """Mẫu số của skill_gap_impact phải khớp với mẫu số của _skill_coverage."""
    cv = make_cv(skills=["Docker"])
    jd = make_jd(
        require_skills=["Docker", "Docker", "Kafka"],
        weights={"skills": 0.5, "experience": 0.3, "education": 0.2},
    )
    svc.set_sim("Kafka", "Docker", 0.80)

    truoc = svc.match(cv, jd, cv_id="c", jd_id="j").rule_based_score
    du_doan = svc.skill_gap_impact(cv, jd)[0].score_gain
    sau = svc.match(make_cv(skills=["Docker", "Kafka"]), jd, cv_id="c", jd_id="j").rule_based_score

    assert sau - truoc == pytest.approx(du_doan, abs=0.2)
