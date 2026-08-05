from pydantic import BaseModel, Field

class Education(BaseModel):
    degree: str = Field(description="Ví dụ: Bachelor, Master, PhD")
    major: str | None = None
    school: str | None = None
    graduation_year: int | None = None

class ExperienceWork(BaseModel):
    job_title: str | None = None
    duration_work: float | None = None
    company: str | None = None
    description: str | None = None

class Project(BaseModel):
    project_title: str | None = None
    duration_complete: float | None = None
    description: str | None = None
    tech_stack: list[str] = []
    role: str | None = None
    is_academic: bool = False


class ParseCV(BaseModel):
    full_name: str | None = None
    email: str | None = None
    skills: list[str] = []
    total_experience_year: float = 0
    work_experiences: list[ExperienceWork] = []
    projects: list[Project] = []
    educations: list[Education] = []

class ParseJD(BaseModel):
    job_title: str | None = None
    require_skills: list[str] = []
    preferred_skills: list[str] = []
    min_experience_years: float = 0
    required_degree: str | None = None
    # Trọng số cho từng tiêu chí khi tổng hợp điểm, tổng nên = 1.0
    weights: dict[str, float] = {
        "skills": 0.5,
        "experience": 0.3,
        "education": 0.2,
    }


class CriterionScore(BaseModel):
    criterion: str
    score: float  # 0-1
    detail: str | None = None
 
 
class MatchResult(BaseModel):
    cv_id: str
    jd_id: str
    criterion_scores: list[CriterionScore]
    rule_based_score: float  # 0-100, tổng hợp từ criterion_scores theo weights
 
 
class CandidateEvaluation(BaseModel):
    cv_id: str
    jd_id: str
    final_score: float  # 0-100, sau khi LLM tinh chỉnh
    strengths: list[str]
    gaps: list[str]
    explanation: str

# Aliases cho đồng bộ với các service khác
ParsedCV = ParseCV
ParsedJD = ParseJD

class LLMEvaluationOutput(BaseModel):
    final_score: float
    strengths: list[str]
    gaps: list[str]
    explanation: str