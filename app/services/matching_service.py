"""
Matching service: so khớp CV với JD theo từng tiêu chí riêng biệt
(skills, experience, education), sau đó tổng hợp theo trọng số của JD.

Đây là bước "rule-based" chạy trước LLM scoring, giúp:
- Có căn cứ định lượng rõ ràng (không phụ thuộc hoàn toàn vào LLM)
- Giảm chi phí gọi LLM (LLM chỉ cần tinh chỉnh + giải thích, không tự tính từ đầu)
"""

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


class MatchingService:
    def __init__(self, embedder: BaseEmbeddingClient, settings: ModelSettings):
        self.embedder = embedder
        self.settings = settings

    def _skill_matches(self, cv_skills: list[str], target_skills: list[str]) -> int:
        """Đếm số skill trong target_skills có ít nhất 1 skill tương đồng
        trong CV, dựa trên embedding similarity thay vì so chuỗi trực tiếp
        (để "Postgres" và "PostgreSQL" vẫn được coi là khớp)."""
        if not target_skills or not cv_skills:
            return 0

        cv_vectors = self.embedder.embed(cv_skills)
        target_vectors = self.embedder.embed(target_skills)

        matched = 0
        for t_vec in target_vectors:
            best = max(cosine_similarity(t_vec, c_vec) for c_vec in cv_vectors)
            if best >= self.settings.skill_match_threshold:
                matched += 1
        return matched

    def _score_skills(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        # Gộp skill khai báo trực tiếp + skill dùng trong project làm 1 "pool"
        # bằng chứng kỹ năng -> vì tech_stack trong project chứng minh ứng viên
        # đã THỰC SỰ áp dụng, không chỉ liệt kê suông ở mục Skills
        project_skills = [s for p in cv.projects for s in p.tech_stack]
        all_skill_evidence = list(set(cv.skills + project_skills))

        # ParseJD dùng 'require_skills' (không có 'd')
        required_skills = jd.require_skills
        preferred_skills = jd.preferred_skills

        matched_required = self._skill_matches(all_skill_evidence, required_skills)
        matched_preferred = self._skill_matches(all_skill_evidence, preferred_skills)

        required_ratio = (
            matched_required / len(required_skills) if required_skills else 1.0
        )
        preferred_ratio = (
            matched_preferred / len(preferred_skills) if preferred_skills else 0.0
        )

        score = 0.8 * required_ratio + 0.2 * preferred_ratio
        detail = (
            f"Khớp {matched_required}/{len(required_skills)} kỹ năng bắt buộc, "
            f"{matched_preferred}/{len(preferred_skills)} kỹ năng ưu tiên "
            f"(tính cả kỹ năng thể hiện qua {len(cv.projects)} project)"
        )
        return CriterionScore(criterion="skills", score=round(score, 3), detail=detail)

    def _project_relevance_score(self, cv: ParsedCV, jd: ParsedJD) -> float:
        """Đo mức độ liên quan giữa các project trong CV với yêu cầu JD,
        dùng embedding similarity giữa mô tả project và (job_title + require_skills).
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

        return max(cosine_similarity(jd_vector, p_vec) for p_vec in project_vectors)

    def _score_experience(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
        # ParseCV dùng 'total_experience_year' (không có 's')
        total_years = cv.total_experience_year

        if jd.min_experience_years <= 0:
            years_score = 1.0
        else:
            years_score = min(total_years / jd.min_experience_years, 1.0)

        # Độ liên quan chức danh: so job_title trong JD với các job_title trong CV
        # ParseCV dùng 'work_experiences' (có 's')
        title_score = 0.0
        if cv.work_experiences:
            cv_titles = [w.job_title for w in cv.work_experiences if w.job_title]
            if cv_titles:
                title_vectors = self.embedder.embed(cv_titles)
                jd_vector = self.embedder.embed([jd.job_title or ""])[0]
                title_score = max(
                    cosine_similarity(jd_vector, t_vec) for t_vec in title_vectors
                )

        # Ứng viên ít/chưa có kinh nghiệm làm việc chính thức (sinh viên, junior)
        # -> cho project "gánh" một phần trọng số thay cho kinh nghiệm thực tế,
        # vì project là bằng chứng gần nhất cho thấy khả năng áp dụng kỹ năng
        LOW_EXPERIENCE_THRESHOLD = 1.0  # năm
        if total_years < LOW_EXPERIENCE_THRESHOLD and cv.projects:
            project_score = self._project_relevance_score(cv, jd)
            score = 0.35 * years_score + 0.30 * title_score + 0.35 * project_score
            detail = (
                f"{total_years} năm kinh nghiệm chính thức (thấp), "
                f"nên tính thêm độ liên quan từ {len(cv.projects)} project "
                f"({project_score:.2f}) để bù đắp"
            )
        else:
            score = 0.6 * years_score + 0.4 * title_score
            detail = (
                f"{total_years} năm kinh nghiệm "
                f"(yêu cầu tối thiểu {jd.min_experience_years} năm), "
                f"độ liên quan chức danh: {title_score:.2f}"
            )

        return CriterionScore(criterion="experience", score=round(score, 3), detail=detail)

    def _score_education(self, cv: ParsedCV, jd: ParsedJD) -> CriterionScore:
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