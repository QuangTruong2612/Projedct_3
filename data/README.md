# Dữ liệu trung gian của bước distillation

Ba file này là **kết quả gọi LLM đã lưu lại**, giữ để không phải trả tiền API
lần nữa mỗi khi cần tính lại feature.

| File | Nội dung |
| --- | --- |
| `parsed_cv.jsonl` | 200 CV đã parse thành `ParseCV` (`{"id": ..., "parsed": {...}}`) |
| `parsed_jd.jsonl` | 100 JD đã parse thành `ParseJD` |
| `evaluations.jsonl` | 200 đánh giá của LLM: `final_score`, `strengths`, `gaps`, `explanation` |

Văn bản gốc chưa parse nằm ở `cv_synthetic.jsonl` và `jd_synthetic.jsonl`
tại thư mục gốc project.

## Khi nào cần dùng

- **Sửa công thức trong `MatchingService`** → chỉ cần tính lại feature từ
  `parsed_cv.jsonl` + `parsed_jd.jsonl`, **không cần gọi LLM**. Ghép với nhãn
  trong `evaluations.jsonl` là ra `training_data.csv` mới.
- **Sinh lại toàn bộ từ đầu** (cần API key) → chạy `run_distillation.py`.

## Lưu ý về nhãn

`final_score` trong `evaluations.jsonl` được sinh khi công thức rule-based còn
là bản cũ. Nhãn vẫn dùng được sau khi sửa công thức, vì nó là đánh giá về **mức
độ phù hợp của ứng viên với vị trí** — đại lượng này không đổi khi ta thay cách
tính điểm trung gian. Sinh lại nhãn dựa trên điểm rule-based mới còn khiến nhãn
bám theo chính công thức, làm mất ý nghĩa của phép kiểm tra "mô hình học có
vượt được công thức không".

Toàn bộ dữ liệu là **hư cấu** (tên/email/số điện thoại bịa), an toàn để lưu trữ.
