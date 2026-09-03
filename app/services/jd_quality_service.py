"""
JD quality service: soi lại một JD đã parse và cảnh báo những vấn đề khiến
việc tuyển dụng khó thành công.

Toàn bộ là luật cứng trên `ParsedJD`, KHÔNG gọi thêm LLM -- JD đã được parse
ở bước trước rồi.

Vì sao cần: chất lượng chấm điểm phụ thuộc trực tiếp vào chất lượng JD. Một JD
liệt kê 19 kỹ năng "bắt buộc" sẽ khiến mọi ứng viên đều có độ phủ thấp và điểm
thấp -- lỗi nằm ở JD chứ không phải ở ứng viên. Phát hiện sớm giúp HR sửa tin
tuyển dụng trước khi đăng.
"""

import re

from app.schemas.models import JDQualityReport, JDQualityWarning, ParsedJD

# Trên mức này thì gần như không ứng viên nào đáp ứng nổi toàn bộ
MANY_REQUIRED_SKILLS = 15
# Số kỹ năng bắt buộc tối thiểu để JD đủ cụ thể mà chấm điểm
FEW_REQUIRED_SKILLS = 3
# Số năm kinh nghiệm cao bất thường
UNREALISTIC_YEARS = 15

# Từ khoá cấp bậc trong chức danh -> khoảng số năm hợp lý
SENIORITY_HINTS: list[tuple[tuple[str, ...], float, float]] = [
    (("intern", "thực tập"), 0.0, 1.0),
    (("fresher", "mới ra trường"), 0.0, 1.5),
    (("junior",), 0.0, 3.0),
    (("senior", "lead", "principal", "staff"), 3.0, 100.0),
    (("manager", "head of", "director"), 4.0, 100.0),
]


def _title_contains(title: str, keywords: tuple[str, ...]) -> bool:
    lowered = title.lower()
    return any(re.search(rf"\b{re.escape(k)}", lowered) for k in keywords)


class JDQualityService:
    def check(self, jd: ParsedJD) -> JDQualityReport:
        warnings: list[JDQualityWarning] = []
        title = jd.job_title or ""
        n_required = len(jd.require_skills)

        # --- số lượng kỹ năng bắt buộc ---
        if n_required == 0:
            warnings.append(
                JDQualityWarning(
                    code="no_required_skills",
                    severity="high",
                    message="JD không nêu kỹ năng bắt buộc nào.",
                    suggestion="Liệt kê 5-10 kỹ năng bắt buộc để hệ thống có căn cứ chấm điểm.",
                )
            )
        elif n_required < FEW_REQUIRED_SKILLS:
            warnings.append(
                JDQualityWarning(
                    code="too_few_required_skills",
                    severity="medium",
                    message=f"Chỉ có {n_required} kỹ năng bắt buộc — quá ít để phân biệt ứng viên.",
                    suggestion="Bổ sung thêm kỹ năng cốt lõi của vị trí.",
                )
            )
        elif n_required > MANY_REQUIRED_SKILLS:
            warnings.append(
                JDQualityWarning(
                    code="too_many_required_skills",
                    severity="high",
                    message=(
                        f"Có tới {n_required} kỹ năng bắt buộc. Gần như không ứng viên nào "
                        "đáp ứng đủ, nên mọi hồ sơ đều sẽ bị điểm thấp."
                    ),
                    suggestion="Giữ lại 8-12 kỹ năng thực sự bắt buộc, chuyển phần còn lại sang mục ưu tiên.",
                )
            )

        # --- không phân tách bắt buộc / ưu tiên ---
        if n_required > 0 and not jd.preferred_skills:
            warnings.append(
                JDQualityWarning(
                    code="no_preferred_skills",
                    severity="low",
                    message="JD không có kỹ năng ưu tiên nào, mọi yêu cầu đều là bắt buộc.",
                    suggestion="Tách các kỹ năng 'có thì tốt' sang mục ưu tiên để không loại nhầm ứng viên tốt.",
                )
            )

        # --- số năm kinh nghiệm ---
        if jd.min_experience_years > UNREALISTIC_YEARS:
            warnings.append(
                JDQualityWarning(
                    code="unrealistic_experience",
                    severity="high",
                    message=f"Yêu cầu tối thiểu {jd.min_experience_years} năm kinh nghiệm là bất thường.",
                    suggestion="Kiểm tra lại — con số này có thể do trích xuất nhầm từ JD.",
                )
            )

        # --- cấp bậc trong chức danh vs số năm ---
        if title:
            for keywords, lo, hi in SENIORITY_HINTS:
                if not _title_contains(title, keywords):
                    continue
                years = jd.min_experience_years
                if years < lo or years > hi:
                    warnings.append(
                        JDQualityWarning(
                            code="seniority_mismatch",
                            severity="medium",
                            message=(
                                f"Chức danh '{title}' nhưng yêu cầu {years} năm kinh nghiệm "
                                f"— khoảng hợp lý thường là {lo:g}-{hi:g} năm."
                            ),
                            suggestion="Chỉnh lại chức danh hoặc số năm cho khớp nhau.",
                        )
                    )
                break

        # --- thiếu thông tin cơ bản ---
        if not title:
            warnings.append(
                JDQualityWarning(
                    code="no_job_title",
                    severity="high",
                    message="Không xác định được chức danh từ JD.",
                    suggestion="Ghi rõ tên vị trí ở đầu tin tuyển dụng.",
                )
            )

        # --- trọng số ---
        total_weight = sum(jd.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(
                JDQualityWarning(
                    code="weights_not_normalized",
                    severity="low",
                    message=f"Tổng trọng số các tiêu chí là {total_weight:.2f}, không bằng 1.0.",
                    suggestion="Chuẩn hoá lại trọng số để điểm tổng nằm đúng thang 0-100.",
                )
            )

        # --- kỹ năng trùng giữa hai nhóm ---
        required_lower = {s.strip().lower() for s in jd.require_skills}
        duplicated = sorted(
            s for s in jd.preferred_skills if s.strip().lower() in required_lower
        )
        if duplicated:
            warnings.append(
                JDQualityWarning(
                    code="duplicated_skills",
                    severity="low",
                    message=f"{len(duplicated)} kỹ năng xuất hiện ở cả mục bắt buộc lẫn ưu tiên: {', '.join(duplicated[:3])}.",
                    suggestion="Bỏ khỏi một trong hai mục để tránh tính điểm hai lần.",
                )
            )

        return JDQualityReport(
            job_title=title or None,
            n_require_skills=n_required,
            n_preferred_skills=len(jd.preferred_skills),
            min_experience_years=jd.min_experience_years,
            warnings=warnings,
        )
