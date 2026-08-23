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

from app.core.embedding_config import BaseEmbeddingClient, cosine_similarity
from app.core.settings import ModelSettings
from app.schemas.models import CriterionScore, MatchResult, ParsedCV, ParsedJD

DEGREE_RANK = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}

# Chuỗi ngắn hơn ngưỡng này không được phép khớp kiểu "nằm trong chuỗi kia",
# nếu không "R" sẽ khớp vào mọi kỹ năng có chữ r, "Go" khớp vào "MongoDB"...
MIN_LEXICAL_LEN = 3


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
    ) -> tuple[float, int]:
        """Trả về (điểm phủ 0-1, số kỹ năng khớp rõ).

        Với mỗi kỹ năng JD yêu cầu:
        - khớp được về mặt chữ  -> 1.0 (chắc chắn đúng)
        - còn lại               -> similarity cao nhất, đã kéo giãn về [0, 1]

        Dùng điểm liên tục thay vì đếm nhị phân qua một ngưỡng: cách cũ coi
        0.849 và 0.20 đều là 0 điểm, làm mất gần hết thông tin.
        """
        if not target_skills:
            return 1.0, 0
        if not cv_skills:
            return 0.0, 0

        cv_vectors = self.embedder.embed(cv_skills)
        target_vectors = self.embedder.embed(target_skills)

        total = 0.0
        clear_matches = 0
        for target, t_vec in zip(target_skills, target_vectors):
            if any(_lexical_match(target, c) for c in cv_skills):
                total += 1.0
                clear_matches += 1
                continue
            best = self._best_similarity(target, cv_vectors, t_vec)
            if best >= self.settings.skill_match_threshold:
                clear_matches += 1
            total += self._rescale(best)

        return total / len(target_skills), clear_matches

    def _score_skills(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        # Gộp skill khai báo trực tiếp + skill dùng trong project làm 1 "pool"
        # bằng chứng kỹ năng -> vì tech_stack trong project chứng minh ứng viên
        # đã THỰC SỰ áp dụng, không chỉ liệt kê suông ở mục Skills
        project_skills = [s for p in cv.projects for s in p.tech_stack]
        all_skill_evidence = list(set(cv.skills + project_skills))

        # ParseJD dùng 'require_skills' (không có 'd')
        required_skills = jd.require_skills
        preferred_skills = jd.preferred_skills

        required_ratio, required_hits = self._skill_coverage(all_skill_evidence, required_skills)
        preferred_ratio, preferred_hits = self._skill_coverage(all_skill_evidence, preferred_skills)

        score = 0.8 * required_ratio + 0.2 * preferred_ratio
        detail = (
            f"Khớp rõ {required_hits}/{len(required_skills)} kỹ năng bắt buộc, "
            f"{preferred_hits}/{len(preferred_skills)} kỹ năng ưu tiên; "
            f"độ phủ bắt buộc {required_ratio:.2f} "
            f"(tính cả kỹ năng thể hiện qua {len(cv.projects)} project)"
        )
        return CriterionScore(criterion="skills", score=round(score, 3), detail=detail)

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
        return CriterionScore(criterion="experience", score=round(score, 3), detail=detail)

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
                criterion="education", score=0.0, detail="CV không có thông tin học vấn"
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
        )
