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


# Thang bậc học vấn, dùng chung cho MatchingService và RankerService.
# Đặt ở đây thay vì trong matching_service để ranker không phải import ngược
# vào service khác chỉ vì một cái dict.
DEGREE_RANK_FOR_RANKER = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}


class CriterionScore(BaseModel):
    criterion: str
    score: float  # 0-1
    detail: str | None = None
    # True khi điểm thấp vì CV THIẾU DỮ LIỆU (parse hỏng, CV scan ảnh, thiếu
    # hẳn một mục) chứ không phải vì ứng viên thực sự không đạt. Hai trường hợp
    # này trước đây đều ra 0.0 và HR không có cách nào phân biệt.
    missing_data: bool = False


class DataCompleteness(BaseModel):
    """Mức độ đầy đủ của dữ liệu trích được từ CV.

    Dùng để cảnh báo HR rằng điểm số có thể không đáng tin -- một CV scan ảnh
    parse hỏng sẽ bị chấm y như một ứng viên kém, nếu không có cờ này.
    """

    has_skills: bool
    has_experience: bool
    has_education: bool
    score: float = Field(description="Tỷ lệ mục trích được, 0-1")
    warnings: list[str] = []

    @property
    def is_reliable(self) -> bool:
        return not self.warnings
 
 
class MatchResult(BaseModel):
    cv_id: str
    jd_id: str
    criterion_scores: list[CriterionScore]
    rule_based_score: float  # 0-100, tổng hợp từ criterion_scores theo weights
    completeness: DataCompleteness | None = None
 
 
class CandidateEvaluation(BaseModel):
    cv_id: str
    jd_id: str
    final_score: float  # 0-100, sau khi LLM tinh chỉnh
    strengths: list[str]
    gaps: list[str]
    explanation: str
    # "llm"   -> LLM đã đọc và giải thích hồ sơ này
    # "model" -> chỉ mô hình chấm, KHÔNG có LLM đọc (bị lọc khỏi top-K)
    # Bắt buộc phải phân biệt: nếu không, HR sẽ tưởng mọi điểm số đều được
    # AI ngôn ngữ xem xét kỹ như nhau.
    scored_by: str = "llm"

# Aliases cho đồng bộ với các service khác
ParsedCV = ParseCV
ParsedJD = ParseJD

class LLMEvaluationOutput(BaseModel):
    final_score: float
    strengths: list[str]
    gaps: list[str]
    explanation: str


class JDQualityWarning(BaseModel):
    code: str
    severity: str = Field(description="high | medium | low")
    message: str
    suggestion: str


class JDQualityReport(BaseModel):
    """Kết quả soi chất lượng một JD.

    Chất lượng chấm điểm phụ thuộc trực tiếp vào chất lượng JD: một JD liệt kê
    19 kỹ năng bắt buộc sẽ khiến mọi ứng viên đều điểm thấp -- lỗi ở JD chứ
    không phải ở ứng viên.
    """

    job_title: str | None
    n_require_skills: int
    n_preferred_skills: int
    min_experience_years: float
    warnings: list[JDQualityWarning] = []

    @property
    def is_healthy(self) -> bool:
        return not any(w.severity == "high" for w in self.warnings)


class SkillImpact(BaseModel):
    """Một kỹ năng còn hụt và việc bổ sung nó ảnh hưởng thế nào tới điểm.

    Trả lời câu hỏi mà cả HR lẫn ứng viên đều muốn biết: "thiếu gì, và bù vào
    thì điểm lên bao nhiêu" -- thay vì chỉ đưa ra một con số không giải thích.
    """

    skill: str
    current_coverage: float = Field(description="Mức đáp ứng hiện tại, 0-1")
    score_gain: float = Field(description="Điểm tổng (0-100) sẽ tăng thêm nếu bổ sung kỹ năng này")
    is_required: bool = True


class InterviewQuestion(BaseModel):
    """Một câu hỏi phỏng vấn nhắm vào đúng một khoảng trống đã phát hiện."""

    question: str
    targets_gap: str = Field(description="Khoảng trống mà câu hỏi này đang kiểm chứng")
    what_to_listen_for: str = Field(
        description="Dấu hiệu nào trong câu trả lời cho thấy ứng viên thực sự có năng lực đó"
    )


class InterviewGuide(BaseModel):
    questions: list[InterviewQuestion]
