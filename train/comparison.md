# So sánh mô hình xếp hạng ứng viên

Dữ liệu: **200 dòng, 100 JD** (2.0 CV mỗi JD). Đánh giá bằng 5-fold GroupKFold tách theo `jd_id`, dự đoán được cắt về [0, 100].

| Mô hình | Feature | MAE | R² | Xếp đúng |
| --- | --- | ---: | ---: | ---: |
| Đoán bừa (luôn trả trung bình) | — | 17.55 | +0.000 | 0/100 = 0.0% |
| Dùng thẳng rule_based_score | — | 10.65 | +0.639 | 99/100 = 99.0% |
| Random Forest | 3 tiêu chí | 6.03 | +0.846 | 98/100 = 98.0% |
| Random Forest | + ngữ cảnh JD | 5.47 | +0.874 | 96/100 = 96.0% |
| Extra Trees | 3 tiêu chí | 6.18 | +0.839 | 97/100 = 97.0% |
| Extra Trees | + ngữ cảnh JD | 5.45 | +0.872 | 98/100 = 98.0% **←** |
| Gradient Boosting | 3 tiêu chí | 5.98 | +0.853 | 94/100 = 94.0% |
| Gradient Boosting | + ngữ cảnh JD | 5.64 | +0.864 | 96/100 = 96.0% |
| HistGradientBoosting | 3 tiêu chí | 5.82 | +0.861 | 98/100 = 98.0% |
| HistGradientBoosting | + ngữ cảnh JD | 5.56 | +0.872 | 98/100 = 98.0% |

Tốt nhất: **Extra Trees** (+ ngữ cảnh JD) — MAE 5.45, xếp đúng 98.0%.

Model đã lưu: `train/rf.pkl`, `train/extratrees.pkl`, `train/gboost.pkl`, `train/histgb.pkl`.

> Mỗi file `.pkl` chứa `{model, features, model_key, score_range}`. Nhớ cắt kết quả `predict()` về `[0, 100]` khi dùng.
