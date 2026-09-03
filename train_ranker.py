"""Huấn luyện và đánh giá các mô hình xếp hạng ứng viên từ training_data.csv.

Mục tiêu: kiểm tra xem một mô hình học từ dữ liệu có vượt được công thức
rule-based cứng hay không, ở CHÍNH bài toán mà hệ thống cần giải -- xếp hạng
các ứng viên trong cùng một JD.

Phạm vi: chỉ so sánh các model thuộc nhánh CÂY (bagging và boosting).

Cách đánh giá:
- Chia fold theo `jd_id` (GroupKFold): các CV của cùng một JD không bao giờ
  nằm cả ở tập train lẫn tập test, tránh mô hình "học thuộc" JD đó.
- Hai thước đo:
    * MAE / R² khi dự đoán `final_score`
    * Tỷ lệ xếp đúng thứ tự từng cặp CV trong cùng 1 JD  <-- quan trọng nhất
- Luôn so với 2 mốc: đoán bừa (luôn trả trung bình) và dùng thẳng
  `rule_based_score` (tức công thức hiện có, không học gì cả).

Chạy (không cần tham số gì):

    python train_ranker.py

Kết quả ghi vào thư mục `train/`:
    comparison.csv   bảng so sánh dạng máy đọc được
    comparison.md    bảng so sánh để dán thẳng vào báo cáo
    <tên model>.pkl  mỗi model huấn luyện trên toàn bộ dữ liệu, lưu riêng
"""

from __future__ import annotations

import csv
import pickle
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_data.csv"
OUT_DIR = ROOT / "train"

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
FULL_FEATURES = BASE_FEATURES + JD_FEATURES

FIT_ORDER = {"poor_fit": 0, "partial_fit": 1, "good_fit": 2}

# final_score theo định nghĩa nằm trong [0, 100]. Model cây không ngoại suy ra
# ngoài khoảng giá trị đã thấy lúc train nên thực tế không bao giờ vượt biên --
# giữ lại như lưới an toàn, miễn phí, và cần thiết nếu sau này thêm model khác.
SCORE_MIN, SCORE_MAX = 0.0, 100.0


def model_registry() -> dict[str, tuple[str, object]]:
    """Các model dạng CÂY được so sánh.

    xgboost/lightgbm tự động thêm vào nếu đã cài; chúng không nằm trong
    requirements.txt để người chỉ chạy API không phải tải thêm.
    """
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )

    registry: dict[str, tuple[str, object]] = {
        # --- bagging ---
        "rf": ("Random Forest", lambda: RandomForestRegressor(n_estimators=300, random_state=0)),
        "extratrees": ("Extra Trees", lambda: ExtraTreesRegressor(n_estimators=300, random_state=0)),
        # --- boosting ---
        "gboost": ("Gradient Boosting", lambda: GradientBoostingRegressor(random_state=0)),
        "histgb": ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=0)),
    }

    try:
        from xgboost import XGBRegressor

        registry["xgboost"] = ("XGBoost", lambda: XGBRegressor(random_state=0, verbosity=0))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        registry["lightgbm"] = ("LightGBM", lambda: LGBMRegressor(random_state=0, verbose=-1))
    except ImportError:
        pass

    return registry


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
    return np.clip(predictions, SCORE_MIN, SCORE_MAX)


def score_row(name: str, feature_set: str, rows: list[dict], y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    ok, total = pairwise_ranking_accuracy(rows, pred)
    return {
        "model": name,
        "features": feature_set,
        "mae": round(float(mean_absolute_error(y, pred)), 2),
        "r2": round(float(r2_score(y, pred)), 3),
        "ranked_ok": ok,
        "ranked_total": total,
        "ranked_pct": round(100 * ok / total, 1),
    }


def print_row(r: dict) -> None:
    label = f"{r['model']} — {r['features']}" if r["features"] else r["model"]
    print(
        f"  {label:44s} MAE={r['mae']:6.2f}  R2={r['r2']:+.3f}  "
        f"xếp đúng {r['ranked_ok']:3d}/{r['ranked_total']} = {r['ranked_pct']:5.1f}%"
    )


def write_reports(results: list[dict], best: dict, saved: list[str], n_rows: int, n_jd: int) -> None:
    """Ghi bảng so sánh ra 2 định dạng: CSV để xử lý tiếp, Markdown để dán vào báo cáo."""
    csv_path = OUT_DIR / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# So sánh mô hình xếp hạng ứng viên",
        "",
        f"Dữ liệu: **{n_rows} dòng, {n_jd} JD** ({n_rows / n_jd:.1f} CV mỗi JD). "
        "Đánh giá bằng 5-fold GroupKFold tách theo `jd_id`, dự đoán được cắt về [0, 100].",
        "",
        "| Mô hình | Feature | MAE | R² | Xếp đúng |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in results:
        star = " **←**" if r is best else ""
        lines.append(
            f"| {r['model']} | {r['features'] or '—'} | {r['mae']:.2f} | "
            f"{r['r2']:+.3f} | {r['ranked_ok']}/{r['ranked_total']} = {r['ranked_pct']:.1f}%{star} |"
        )
    lines += [
        "",
        f"Tốt nhất: **{best['model']}** ({best['features']}) — MAE {best['mae']:.2f}, "
        f"xếp đúng {best['ranked_pct']:.1f}%.",
        "",
        "Model đã lưu: " + ", ".join(f"`train/{k}.pkl`" for k in saved) + ".",
        "",
        "> Mỗi file `.pkl` chứa `{model, features, model_key, score_range}`. "
        "Nhớ cắt kết quả `predict()` về `[0, 100]` khi dùng.",
    ]
    md_path = OUT_DIR / "comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[done] Bảng so sánh -> {csv_path.relative_to(ROOT)} và {md_path.relative_to(ROOT)}")


def main() -> None:
    rows = load_rows()
    y = np.array([float(r["final_score"]) for r in rows])
    groups = np.array([r["jd_id"] for r in rows])
    n_jd = len(set(groups))

    OUT_DIR.mkdir(exist_ok=True)
    print(f"[i] {len(rows)} dòng, {n_jd} JD, {len(rows) / n_jd:.1f} CV mỗi JD\n")

    results: list[dict] = []

    # --- mốc so sánh: không học gì cả ---
    print("=== MỐC SO SÁNH (không học gì) ===")
    for name, pred in (
        ("Đoán bừa (luôn trả trung bình)", np.full_like(y, y.mean())),
        ("Dùng thẳng rule_based_score", np.array([float(r["rule_based_score"]) for r in rows])),
    ):
        row = score_row(name, "", rows, y, pred)
        results.append(row)
        print_row(row)

    # --- các model cây ---
    registry = model_registry()
    print("\n=== MÔ HÌNH CÂY (5-fold, tách theo JD) ===")
    model_rows: list[dict] = []
    for label, model_fn in registry.values():
        for feature_set, features in (
            ("3 tiêu chí", BASE_FEATURES),
            ("+ ngữ cảnh JD", FULL_FEATURES),
        ):
            X = build_matrix(rows, features)
            row = score_row(label, feature_set, rows, y, cross_val_predict(X, y, groups, model_fn))
            results.append(row)
            model_rows.append(row)
            print_row(row)

    # Tốt nhất = MAE thấp nhất; hoà thì lấy tỷ lệ xếp đúng cao hơn
    best = min(model_rows, key=lambda r: (r["mae"], -r["ranked_pct"]))

    # --- huấn luyện trên toàn bộ dữ liệu và lưu từng model ---
    print(f"\n=== LƯU MODEL (huấn luyện trên toàn bộ {len(rows)} dòng, feature đầy đủ) ===")
    X_full = build_matrix(rows, FULL_FEATURES)
    saved: list[str] = []
    for key, (label, model_fn) in registry.items():
        model = model_fn().fit(X_full, y)
        path = OUT_DIR / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "features": FULL_FEATURES,
                    "model_key": key,
                    # Bên dùng model phải cắt kết quả về khoảng này.
                    "score_range": [SCORE_MIN, SCORE_MAX],
                },
                f,
            )
        saved.append(key)
        print(f"  {label:22s} -> {path.relative_to(ROOT)}")

    # --- độ quan trọng feature ---
    from sklearn.ensemble import GradientBoostingRegressor

    gb = GradientBoostingRegressor(random_state=0).fit(X_full, y)
    print("\n=== ĐỘ QUAN TRỌNG CỦA FEATURE (Gradient Boosting) ===")
    for name, imp in sorted(zip(FULL_FEATURES, gb.feature_importances_), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {imp:5.3f} {'#' * int(round(imp * 50))}")

    write_reports(results, best, saved, len(rows), n_jd)
    print(
        f"[i] Tốt nhất: {best['model']} ({best['features']}) "
        f"— MAE {best['mae']:.2f}, xếp đúng {best['ranked_pct']:.1f}%"
    )


if __name__ == "__main__":
    sys.exit(main())
