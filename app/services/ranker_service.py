"""
Ranker service: dùng mô hình đã huấn luyện (train_ranker.py) để dự đoán
final_score mà KHÔNG cần gọi LLM.

Công dụng chính là làm BỘ LỌC RẺ: với một đợt 100 CV, thay vì gọi LLM chấm
điểm cho cả 100, ta chấm bằng mô hình trước (miễn phí, vài mili giây) rồi chỉ
gọi LLM cho top-K hồ sơ đáng đọc kỹ.

Toàn bộ service này là TUỲ CHỌN: thiếu file model, thiếu scikit-learn, hay file
hỏng thì `is_available` trả False và pipeline chạy y như trước.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from app.schemas.models import DEGREE_RANK_FOR_RANKER, MatchResult, ParsedJD

logger = logging.getLogger(__name__)


class RankerService:
    def __init__(self, model_path: str | Path | None):
        self.model = None
        self.features: list[str] = []
        self.model_key: str | None = None
        self.score_range = (0.0, 100.0)

        if not model_path:
            return

        path = Path(model_path)
        if not path.exists():
            logger.info("Không tìm thấy model xếp hạng ở %s -- bỏ qua bộ lọc rẻ.", path)
            return

        try:
            # pickle.load thực thi được mã tuỳ ý -> chỉ nạp file do CHÍNH dự án
            # sinh ra bằng train_ranker.py, đừng bao giờ trỏ vào file lạ.
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.features = list(bundle["features"])
            self.model_key = bundle.get("model_key")
            lo, hi = bundle.get("score_range", [0.0, 100.0])
            self.score_range = (float(lo), float(hi))
            logger.info("Đã nạp model xếp hạng '%s' từ %s", self.model_key, path)
        except Exception as e:  # noqa: BLE001 - file hỏng/thiếu sklearn đều không được làm sập app
            logger.warning("Không nạp được model xếp hạng từ %s: %s", path, e)
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _feature_vector(self, match_result: MatchResult, jd: ParsedJD) -> list[float]:
        """Dựng vector feature theo ĐÚNG thứ tự đã lưu trong file model.

        Thứ tự đọc từ file chứ không hardcode: nếu train_ranker.py đổi danh sách
        feature mà đây vẫn giữ thứ tự cũ thì mô hình sẽ nhận sai đầu vào và
        chấm sai một cách âm thầm.
        """
        by_criterion = {c.criterion: c.score for c in match_result.criterion_scores}
        weights = jd.weights

        values = {
            "skill_score": by_criterion.get("skills", 0.0),
            "experience_score": by_criterion.get("experience", 0.0),
            "education_score": by_criterion.get("education", 0.0),
            "w_skills": weights.get("skills", 0.5),
            "w_experience": weights.get("experience", 0.3),
            "w_education": weights.get("education", 0.2),
            "min_experience_years": jd.min_experience_years,
            "required_degree_rank": DEGREE_RANK_FOR_RANKER.get(
                (jd.required_degree or "").lower(), 0
            ),
            "n_require_skills": len(jd.require_skills),
        }

        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(
                f"Model cần feature chưa dựng được: {missing}. "
                "Chạy lại train_ranker.py hoặc cập nhật _feature_vector()."
            )
        return [float(values[f]) for f in self.features]

    def predict(self, match_result: MatchResult, jd: ParsedJD) -> float | None:
        """Dự đoán final_score (0-100). Trả None nếu model không dùng được."""
        if not self.is_available:
            return None

        try:
            raw = float(self.model.predict([self._feature_vector(match_result, jd)])[0])
        except Exception as e:  # noqa: BLE001
            logger.warning("Model xếp hạng lỗi khi dự đoán: %s", e)
            return None

        lo, hi = self.score_range
        return round(min(max(raw, lo), hi), 1)
