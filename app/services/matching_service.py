"""
Matching service: so khớp CV với JD theo từng tiêu chí riêng biệt
(skills, experience, education), sau đó tổng hợp theo trọng số của JD.

Đây là bước "rule-based" chạy trước LLM scoring, giúp:
- Có căn cứ định lượng rõ ràng (không phụ thuộc hoàn toàn vào LLM)
- Giảm chi phí gọi LLM (LLM chỉ cần tinh chỉnh + giải thích, không tự tính từ đầu)

Các hằng số trong file này được HIỆU CHỈNH TỪ DỮ LIỆU chứ không chọn cảm tính
-- xem phần "Calibrating the scoring formulas" trong README.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from app.core.embedding_config import BaseEmbeddingClient, cosine_similarity
from app.core.settings import ModelSettings
from app.schemas.models import (
    DEGREE_RANK_FOR_RANKER,
    CriterionScore,
    DataCompleteness,
    MatchResult,
    ParsedCV,
    ParsedJD,
    SkillImpact,
)

# Alias giữ nguyên tên cũ để run_distillation.py và test không phải sửa
DEGREE_RANK = DEGREE_RANK_FOR_RANKER

# Chuỗi ngắn hơn ngưỡng này không được phép khớp kiểu "nằm trong chuỗi kia",
# nếu không "R" sẽ khớp vào mọi kỹ năng có chữ r, "Go" khớp vào "MongoDB"...
MIN_LEXICAL_LEN = 3

# Trọng số giữa kỹ năng bắt buộc và kỹ năng ưu tiên trong điểm skills.
# Tách thành hằng số vì phần phân tích phản thực cần dùng lại đúng con số này.
REQUIRED_WEIGHT = 0.8
PREFERRED_WEIGHT = 0.2


@dataclass
class SkillCoverage:
    """Kết quả so khớp một danh sách kỹ năng JD với pool bằng chứng của CV."""

    ratio: float                                   # độ phủ trung bình 0-1
    clear_matches: int                             # số kỹ năng khớp rõ ràng
    per_skill: dict[str, float] = field(default_factory=dict)  # kỹ năng -> độ phủ


def _normalize(text: str) -> str:
    """Bỏ dấu tiếng Việt, viết thường, bỏ mọi ký tự không phải chữ/số.

    "React.js" -> "reactjs", "Node JS" -> "nodejs", "PostgreSQL" -> "postgresql"
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", without_marks.lower())


def _lexical_match(a: str, b: str) -> bool:
    """So khớp theo mặt chữ: bằng nhau, hoặc chuỗi ngắn nằm trong chuỗi dài.

    Bắt được phần lớn biến thể viết ("ReactJS"/"React.js", "PostgreSQL"/"Postgres")
    một cách CHẮC CHẮN, không phụ thuộc vào chất lượng embedding.
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(short) >= MIN_LEXICAL_LEN and short in long


class MatchingService:
    def __init__(self, embedder: BaseEmbeddingClient, settings: ModelSettings):
        self.embedder = embedder
        self.settings = settings

    # ------------------------------------------------------------------ tiện ích

    def _rescale(self, similarity: float) -> float:
        """Kéo giãn similarity từ [sàn nhiễu, 1] về [0, 1].

        LÝ DO: cosine similarity của embedding trên các chuỗi kỹ năng ngắn KHÔNG
        trải đều trên [0, 1]. Đo trên ~4.000 cặp kỹ năng ghép ngẫu nhiên (hoàn
        toàn không liên quan) cho thấy trung vị đã là 0.79 và 5% số cặp vượt
        0.844 -- tức ngưỡng 0.85 cũ chỉ nằm nhỉnh hơn nhiễu thuần tuý.
        Kéo giãn lại giúp "không liên quan" thực sự về 0.
        """
        floor = self.settings.similarity_floor
        if floor >= 1.0:
            return 0.0
        return min(1.0, max(0.0, (similarity - floor) / (1.0 - floor)))

    def _best_similarity(self, target: str, pool_vectors: list[list[float]], target_vector) -> float:
        if not pool_vectors:
            return 0.0
        return max(cosine_similarity(target_vector, v) for v in pool_vectors)

    # ------------------------------------------------------------------ skills

    def _skill_coverage(
        self, cv_skills: list[str], target_skills: list[str]
    ) -> SkillCoverage:
        """Đo mức độ CV đáp ứng từng kỹ năng trong danh sách JD yêu cầu.

        Với mỗi kỹ năng JD yêu cầu:
        - khớp được về mặt chữ  -> 1.0 (chắc chắn đúng)
        - còn lại               -> similarity cao nhất, đã kéo giãn về [0, 1]

        Dùng điểm liên tục thay vì đếm nhị phân qua một ngưỡng: cách cũ coi
        0.849 và 0.20 đều là 0 điểm, làm mất gần hết thông tin.

        Giữ lại độ phủ CỦA TỪNG kỹ năng (`per_skill`) để phần phân tích phản
        thực biết chính xác ứng viên đang hụt ở đâu.
        """
        # Khử trùng lặp TRƯỚC khi tính: JD viết lặp một kỹ năng ("Python" xuất
        # hiện 2 lần) không được phép làm thay đổi điểm của ứng viên. Nếu không
        # khử, `per_skill` (dict) gộp lại còn 1 mục trong khi mẫu số vẫn đếm 2
        # -> ứng viên bị trừ điểm oan.
        # Chỉ so theo chữ thường đã trim, KHÔNG dùng _normalize() vì nó sẽ gộp
        # nhầm "C++" với "C".
        targets: list[str] = []
        seen: set[str] = set()
        for t in target_skills:
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                targets.append(t)

        if not targets:
            return SkillCoverage(ratio=1.0, clear_matches=0)
        if not cv_skills:
            return SkillCoverage(
                ratio=0.0, clear_matches=0, per_skill={t: 0.0 for t in targets}
            )

        cv_vectors = self.embedder.embed(cv_skills)
        target_vectors = self.embedder.embed(targets)

        per_skill: dict[str, float] = {}
        clear_matches = 0
        for target, t_vec in zip(targets, target_vectors):
            if any(_lexical_match(target, c) for c in cv_skills):
                per_skill[target] = 1.0
                clear_matches += 1
                continue
            best = self._best_similarity(target, cv_vectors, t_vec)
            if best >= self.settings.skill_match_threshold:
                clear_matches += 1
            per_skill[target] = self._rescale(best)

        return SkillCoverage(
            ratio=sum(per_skill.values()) / len(targets),
            clear_matches=clear_matches,
            per_skill=per_skill,
        )

    def _score_skills(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        # Gộp skill khai báo trực tiếp + skill dùng trong project làm 1 "pool"
        # bằng chứng kỹ năng -> vì tech_stack trong project chứng minh ứng viên
        # đã THỰC SỰ áp dụng, không chỉ liệt kê suông ở mục Skills
        project_skills = [s for p in cv.projects for s in p.tech_stack]
        all_skill_evidence = list(set(cv.skills + project_skills))

        # ParseJD dùng 'require_skills' (không có 'd')
        required_skills = jd.require_skills
        preferred_skills = jd.preferred_skills

        required = self._skill_coverage(all_skill_evidence, required_skills)
        preferred = self._skill_coverage(all_skill_evidence, preferred_skills)

        score = REQUIRED_WEIGHT * required.ratio + PREFERRED_WEIGHT * preferred.ratio

        # Không trích được kỹ năng nào -> điểm 0 là do THIẾU DỮ LIỆU, không phải
        # do ứng viên kém. Đánh dấu để HR biết đừng tin con số này.
        if not all_skill_evidence:
            return CriterionScore(
                criterion="skills",
                score=round(score, 3),
                detail="Không trích được kỹ năng nào từ CV (có thể do CV scan ảnh hoặc định dạng lạ)",
                missing_data=True,
            )

        detail = (
            f"Khớp rõ {required.clear_matches}/{len(required_skills)} kỹ năng bắt buộc, "
            f"{preferred.clear_matches}/{len(preferred_skills)} kỹ năng ưu tiên; "
            f"độ phủ bắt buộc {required.ratio:.2f} "
            f"(tính cả kỹ năng thể hiện qua {len(cv.projects)} project)"
        )
        return CriterionScore(criterion="skills", score=round(score, 3), detail=detail)

    def skill_gap_impact(
        self, cv: ParsedCV, jd: ParsedJD, top_n: int = 5
    ) -> list[SkillImpact]:
        """Phân tích phản thực: bổ sung kỹ năng nào thì điểm tổng tăng nhiều nhất.

        Tính THẲNG BẰNG CÔNG THỨC chứ không chạy lại pipeline cho từng kỹ năng.
        Điểm skills là trung bình có trọng số của độ phủ từng kỹ năng, nên nếu
        một kỹ năng đi từ độ phủ `c` lên 1.0 thì:

            điểm tổng tăng = w_skills x REQUIRED_WEIGHT x (1 - c) / n x 100

        Cách này vừa chính xác tuyệt đối vừa không tốn thêm lần embed nào.

        Trả về danh sách đã sắp theo mức tăng giảm dần; kỹ năng đã đáp ứng
        đầy đủ thì không xuất hiện.
        """
        project_skills = [sk for pr in cv.projects for sk in pr.tech_stack]
        evidence = list(set(cv.skills + project_skills))

        w_skills = jd.weights.get("skills", 0.5)
        impacts: list[SkillImpact] = []

        for targets, group_weight, is_required in (
            (jd.require_skills, REQUIRED_WEIGHT, True),
            (jd.preferred_skills, PREFERRED_WEIGHT, False),
        ):
            if not targets:
                continue
            coverage = self._skill_coverage(evidence, targets)
            # Mẫu số phải là số kỹ năng SAU khi khử trùng lặp, đúng bằng mẫu số
            # mà _skill_coverage dùng để tính ratio -- nếu lấy len(targets) thô
            # thì mức tăng dự đoán sẽ sai khi JD có kỹ năng viết lặp.
            n_targets = len(coverage.per_skill)
            for skill, current in coverage.per_skill.items():
                if current >= 0.999:  # đã đáp ứng, bổ sung cũng không tăng thêm
                    continue
                gain = w_skills * group_weight * (1.0 - current) / n_targets * 100
                impacts.append(
                    SkillImpact(
                        skill=skill,
                        current_coverage=round(current, 3),
                        score_gain=round(gain, 2),
                        is_required=is_required,
                    )
                )

        impacts.sort(key=lambda i: i.score_gain, reverse=True)
        return impacts[:top_n]

    # ------------------------------------------------------------------ experience

    def _project_relevance(self, cv: ParsedCV, jd: ParsedJD) -> float:
        """Độ liên quan giữa project trong CV với yêu cầu JD, đã kéo giãn.
        Trả về 0.0 nếu CV không có project nào."""
        if not cv.projects:
            return 0.0

        project_texts = [
            f"{p.project_title or 'Project'}: {p.description or ''} (Công nghệ: {', '.join(p.tech_stack)})"
            for p in cv.projects
        ]
        jd_text = f"{jd.job_title}. Yêu cầu: {', '.join(jd.require_skills)}"

        project_vectors = self.embedder.embed(project_texts)
        jd_vector = self.embedder.embed([jd_text])[0]

        best = max(cosine_similarity(jd_vector, p_vec) for p_vec in project_vectors)
        return self._rescale(best)

    def _title_relevance(self, cv: ParsedCV, jd: ParsedJD) -> float:
        """Độ liên quan giữa chức danh JD và các chức danh từng làm, đã kéo giãn."""
        # ParseCV dùng 'work_experiences' (có 's')
        cv_titles = [w.job_title for w in cv.work_experiences if w.job_title]
        if not cv_titles or not jd.job_title:
            return 0.0

        title_vectors = self.embedder.embed(cv_titles)
        jd_vector = self.embedder.embed([jd.job_title])[0]
        best = max(cosine_similarity(jd_vector, t_vec) for t_vec in title_vectors)
        return self._rescale(best)

    def _score_experience(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        # ParseCV dùng 'total_experience_year' (không có 's')
        total_years = cv.total_experience_year

        if jd.min_experience_years <= 0:
            years_score = 1.0
        else:
            years_score = min(total_years / jd.min_experience_years, 1.0)

        title_relevance = self._title_relevance(cv, jd)

        # Ứng viên ít/chưa có kinh nghiệm làm việc chính thức (sinh viên, junior)
        # -> cho project "gánh" phần độ liên quan thay cho chức danh, vì project
        # là bằng chứng gần nhất cho thấy khả năng áp dụng kỹ năng
        LOW_EXPERIENCE_THRESHOLD = 1.0  # năm
        used_projects = total_years < LOW_EXPERIENCE_THRESHOLD and bool(cv.projects)
        if used_projects:
            relevance = max(title_relevance, self._project_relevance(cv, jd))
        else:
            relevance = title_relevance

        # NHÂN chứ không CỘNG: số năm chỉ có giá trị khi công việc có liên quan.
        # Cách cũ (0.6*năm + 0.4*liên_quan) cho một kế toán 2.2 năm ứng tuyển QA
        # tới 0.93 điểm kinh nghiệm, vì số năm được cộng thẳng bất kể trái ngành.
        # `base` là phần điểm giữ lại kể cả khi hoàn toàn không liên quan.
        base = self.settings.experience_relevance_base
        gate = base + (1.0 - base) * relevance
        score = years_score * gate

        source = "project" if used_projects else "chức danh"
        detail = (
            f"{total_years} năm kinh nghiệm "
            f"(yêu cầu tối thiểu {jd.min_experience_years} năm), "
            f"độ liên quan theo {source}: {relevance:.2f}"
        )
        # Không có mục kinh nghiệm lẫn project -> không đủ căn cứ để chấm
        missing = not cv.work_experiences and not cv.projects
        if missing:
            detail = "CV không có mục kinh nghiệm làm việc lẫn dự án -- không đủ căn cứ để đánh giá"
        return CriterionScore(
            criterion="experience", score=round(score, 3), detail=detail, missing_data=missing
        )

    # ------------------------------------------------------------------ education

    def _score_education(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        """CỐ Ý chỉ xét cấp bằng, không xét chuyên ngành.

        Đã thử nhân thêm độ liên quan giữa chuyên ngành và JD (so với chức danh,
        so với chức danh + kỹ năng, nhiều mức sàn khác nhau). Đo trên 200 cặp
        CV-JD: AUC good/poor chỉ nhích từ 0.520 lên 0.564 và mọi biến thể đều
        làm tiêu chí này PHẲNG HƠN (độ lệch chuẩn 0.047 -> 0.025), tức càng ít
        thông tin hơn cho mô hình học ở bước sau. Nên giữ nguyên công thức đơn
        giản. Xem README mục "Calibrating the scoring formulas".
        """
        if not jd.required_degree:
            return CriterionScore(criterion="education", score=1.0, detail="Không yêu cầu")

        # ParseCV dùng 'educations' (có 's')
        if not cv.educations:
            return CriterionScore(
                criterion="education",
                score=0.0,
                detail="CV không có thông tin học vấn -- điểm 0 này là do thiếu dữ liệu, không phải do không đạt yêu cầu",
                missing_data=True,
            )

        # Ứng viên có thể có nhiều bằng (Bachelor, Master...) -> lấy bằng cao nhất
        # để so sánh, vì đó là trình độ học vấn thực tế cao nhất của ứng viên
        highest = max(cv.educations, key=lambda e: DEGREE_RANK.get(e.degree.lower(), 0))
        cv_degree = highest.degree.lower()
        cv_rank = DEGREE_RANK.get(cv_degree, 0)
        required_rank = DEGREE_RANK.get(jd.required_degree.lower(), 3)

        score = 1.0 if cv_rank >= required_rank else cv_rank / max(required_rank, 1)
        detail = f"CV: {cv_degree or 'không rõ'} (bằng cao nhất), yêu cầu: {jd.required_degree}"
        return CriterionScore(criterion="education", score=round(score, 3), detail=detail)

    # ------------------------------------------------------------------ tổng hợp

    def _completeness(self, cv: ParsedCV) -> DataCompleteness:
        """Đo xem trích được bao nhiêu phần của CV, để cảnh báo khi điểm số
        có thể không đáng tin."""
        has_skills = bool(cv.skills or any(p.tech_stack for p in cv.projects))
        has_experience = bool(cv.work_experiences or cv.projects)
        has_education = bool(cv.educations)

        warnings: list[str] = []
        if not has_skills:
            warnings.append("Không trích được kỹ năng nào")
        if not has_experience:
            warnings.append("Không trích được kinh nghiệm làm việc hay dự án nào")
        if not has_education:
            warnings.append("Không trích được thông tin học vấn")

        found = sum([has_skills, has_experience, has_education])
        return DataCompleteness(
            has_skills=has_skills,
            has_experience=has_experience,
            has_education=has_education,
            score=round(found / 3, 3),
            warnings=warnings,
        )

    def match(self, cv: ParsedCV, jd: ParsedJD, cv_id: str, jd_id: str) -> MatchResult:
        skill_cs = self._score_skills(cv, jd)
        exp_cs = self._score_experience(cv, jd)
        edu_cs = self._score_education(cv, jd)

        weighted_sum = (
            jd.weights.get("skills", 0.5) * skill_cs.score
            + jd.weights.get("experience", 0.3) * exp_cs.score
            + jd.weights.get("education", 0.2) * edu_cs.score
        )

        return MatchResult(
            cv_id=cv_id,
            jd_id=jd_id,
            criterion_scores=[skill_cs, exp_cs, edu_cs],
            rule_based_score=round(weighted_sum * 100, 1),
            completeness=self._completeness(cv),
        )
