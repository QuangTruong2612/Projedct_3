"""Huấn luyện và đánh giá mô hình xếp hạng ứng viên từ training_data.csv.

Mục tiêu: kiểm tra xem một mô hình học từ dữ liệu có vượt được công thức
rule-based cứng hay không, ở CHÍNH bài toán mà hệ thống cần giải -- xếp hạng
các ứng viên trong cùng một JD.

Cách đánh giá:
- Chia fold theo `jd_id` (GroupKFold): các CV của cùng một JD không bao giờ
  nằm cả ở tập train lẫn tập test, tránh mô hình "học thuộc" JD đó.
- Hai thước đo:
    * MAE / R² khi dự đoán `final_score`
    * Tỷ lệ xếp đúng thứ tự từng cặp CV trong cùng 1 JD  <-- quan trọng nhất
- Luôn so với 2 mốc: đoán bừa (luôn trả trung bình) và dùng thẳng
  `rule_based_score` (tức công thức hiện có, không học gì cả).

Chạy:
    python train_ranker.py                 # đánh giá + in bảng so sánh
    python train_ranker.py --save model.pkl  # huấn luyện trên toàn bộ và lưu lại
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_data.csv"

# Feature chỉ mô tả cặp (CV, JD) -- KHÔNG dùng rule_based_score, vì mục đích
# là xem mô hình có tự học được cách tổng hợp tốt hơn công thức hay không.
BASE_FEATURES = ["skill_score", "experience_score", "education_score"]
# Ngữ cảnh JD: cùng một bộ điểm tiêu chí nhưng JD khác nhau thì kết quả khác
# nhau, vì mỗi JD có trọng số và yêu cầu riêng. Thiếu nhóm này, mô hình bị buộc
# phải đoán một biến mà nó không nhìn thấy.
JD_FEATURES = [
    "w_skills",
    "w_experience",
    "w_education",
    "min_experience_years",
    "required_degree_rank",
    "n_require_skills",
]

FIT_ORDER = {"poor_fit": 0, "partial_fit": 1, "good_fit": 2}


def load_rows() -> list[dict]:
    if not DATA.exists():
        sys.exit(f"Không tìm thấy {DATA}. Chạy run_distillation.py để sinh dữ liệu trước.")
    with open(DATA, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.array([[float(r[c]) for c in features] for r in rows])


def pairwise_ranking_accuracy(rows: list[dict], scores: np.ndarray) -> tuple[int, int]:
    """Với mỗi cặp CV trong cùng 1 JD có mức phù hợp khác nhau, kiểm tra xem
    ứng viên phù hợp hơn có được chấm cao hơn không."""
    per_jd: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r, s in zip(rows, scores):
        per_jd[r["jd_id"]].append((r["fit_level_intended"], float(s)))

    ok = total = 0
    for items in per_jd.values():
        for (f1, s1), (f2, s2) in combinations(items, 2):
            if FIT_ORDER[f1] == FIT_ORDER[f2]:
                continue
            total += 1
            better, worse = (s1, s2) if FIT_ORDER[f1] > FIT_ORDER[f2] else (s2, s1)
            if better > worse:
                ok += 1
    return ok, total


def cross_val_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_fn, n_splits=5):
    from sklearn.model_selection import GroupKFold

    predictions = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions


def report(name: str, rows: list[dict], y: np.ndarray, pred: np.ndarray) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score

    ok, total = pairwise_ranking_accuracy(rows, pred)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"  {name:38s} MAE={mae:6.2f}  R2={r2:+.3f}  xếp đúng {ok:3d}/{total} = {100 * ok / total:5.1f}%")


def evaluate() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    rows = load_rows()
    y = np.array([float(r["final_score"]) for r in rows])
    groups = np.array([r["jd_id"] for r in rows])

    n_jd = len(set(groups))
    print(f"[i] {len(rows)} dòng, {n_jd} JD, {len(rows) / n_jd:.1f} CV mỗi JD\n")

    print("=== MỐC SO SÁNH (không học gì) ===")
    report("đoán bừa (luôn trả trung bình)", rows, y, np.full_like(y, y.mean()))
    report("dùng thẳng rule_based_score", rows, y, np.array([float(r["rule_based_score"]) for r in rows]))

    print("\n=== MÔ HÌNH HỌC TỪ DỮ LIỆU (5-fold, tách theo JD) ===")
    configs = [
        ("3 tiêu chí - tuyến tính", BASE_FEATURES, lambda: LinearRegression()),
        ("3 tiêu chí - GBoost", BASE_FEATURES, lambda: GradientBoostingRegressor(random_state=0)),
        ("3 tiêu chí + ngữ cảnh JD - tuyến tính", BASE_FEATURES + JD_FEATURES, lambda: LinearRegression()),
        ("3 tiêu chí + ngữ cảnh JD - GBoost", BASE_FEATURES + JD_FEATURES,
         lambda: GradientBoostingRegressor(random_state=0)),
    ]
    for name, features, model_fn in configs:
        X = build_matrix(rows, features)
        report(name, rows, y, cross_val_predict(X, y, groups, model_fn))

    # Mức độ đóng góp của từng feature
    X = build_matrix(rows, BASE_FEATURES + JD_FEATURES)
    model = GradientBoostingRegressor(random_state=0).fit(X, y)
    print("\n=== ĐỘ QUAN TRỌNG CỦA FEATURE (GBoost, huấn luyện trên toàn bộ) ===")
    for name, imp in sorted(
        zip(BASE_FEATURES + JD_FEATURES, model.feature_importances_),
        key=lambda kv: -kv[1],
    ):
        bar = "#" * int(round(imp * 50))
        print(f"  {name:22s} {imp:5.3f} {bar}")


def save_model(path: Path) -> None:
    import pickle

    from sklearn.ensemble import GradientBoostingRegressor

    rows = load_rows()
    features = BASE_FEATURES + JD_FEATURES
    X = build_matrix(rows, features)
    y = np.array([float(r["final_score"]) for r in rows])

    model = GradientBoostingRegressor(random_state=0).fit(X, y)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"[done] đã lưu mô hình ({len(rows)} dòng huấn luyện) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="huấn luyện trên toàn bộ dữ liệu và lưu mô hình")
    args = parser.parse_args()

    if args.save:
        save_model(Path(args.save))
    else:
        evaluate()


if __name__ == "__main__":
    sys.exit(main())
