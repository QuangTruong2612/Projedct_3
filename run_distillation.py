"""
Bước "distillation": đưa từng cặp (CV, JD) synthetic qua CHÍNH pipeline hiện
có của project (parse -> match rule-based -> LLM scoring) để lấy:
  - final_score (LLM đã tinh chỉnh) -> dùng làm LABEL cho model ranker
  - criterion_scores (skills/experience/education) -> dùng làm FEATURE
  - ngữ cảnh JD (trọng số, số năm yêu cầu...) -> FEATURE bổ sung

Kết quả lưu ra 1 file CSV phẳng, sẵn sàng đưa vào train_ranker.py.

Script chạy được nhiều lần: CV nào đã có trong file CSV sẽ được bỏ qua, nên
nếu đang chạy dở mà lỗi mạng thì chỉ cần chạy lại.

Cách dùng (từ thư mục gốc project, nơi có folder `app/` và file `.env`):
    python run_distillation.py --jd_file jd_synthetic.jsonl \
        --cv_file cv_synthetic.jsonl --out training_data.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # đọc .env của project (API key, LLM_PROVIDER...) TRƯỚC khi import app.*

from app.core.settings import ModelSettings  # noqa: E402
from app.services.matching_service import DEGREE_RANK  # noqa: E402
from app.services.pipeline import RecruitmentPipeline  # noqa: E402

FIELDNAMES = [
    "cv_id", "jd_id", "jd_role", "fit_level_intended",
    # feature mô tả mức độ khớp giữa CV và JD
    "skill_score", "experience_score", "education_score", "rule_based_score",
    # feature mô tả bản thân JD -- thiếu nhóm này thì cùng một bộ điểm tiêu chí
    # ở hai JD khác nhau lại ứng với hai final_score rất khác nhau, và mô hình
    # bị buộc phải đoán một biến nó không nhìn thấy
    "w_skills", "w_experience", "w_education",
    "min_experience_years", "required_degree_rank", "n_require_skills",
    # nhãn
    "final_score", "strengths", "gaps", "explanation",
]


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def already_done_cv_ids(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["cv_id"])
    return done


def build_row(cv: dict, jd: dict, result) -> dict:
    """Gom mọi thứ cần cho 1 dòng dataset từ DUY NHẤT một lần chạy pipeline."""
    crit = {cs.criterion: cs.score for cs in result.criterion_scores}
    parsed_jd = result.parsed_jd
    weights = parsed_jd.weights

    return {
        "cv_id": cv["id"],
        "jd_id": jd["id"],
        "jd_role": jd.get("role"),
        "fit_level_intended": cv.get("fit_level"),
        "skill_score": crit.get("skills"),
        "experience_score": crit.get("experience"),
        "education_score": crit.get("education"),
        "rule_based_score": result.rule_based_score,
        "w_skills": round(weights.get("skills", 0.5), 3),
        "w_experience": round(weights.get("experience", 0.3), 3),
        "w_education": round(weights.get("education", 0.2), 3),
        "min_experience_years": parsed_jd.min_experience_years,
        "required_degree_rank": DEGREE_RANK.get((parsed_jd.required_degree or "").lower(), 0),
        "n_require_skills": len(parsed_jd.require_skills),
        "final_score": result.evaluation.final_score,
        "strengths": " | ".join(result.evaluation.strengths),
        "gaps": " | ".join(result.evaluation.gaps),
        "explanation": result.evaluation.explanation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jd_file", default="jd_synthetic.jsonl")
    parser.add_argument("--cv_file", default="cv_synthetic.jsonl")
    parser.add_argument("--out", default="training_data.csv")
    parser.add_argument("--delay", type=float, default=0.5, help="nghỉ giữa 2 lần gọi LLM (giây)")
    args = parser.parse_args()

    jds = {jd["id"]: jd for jd in load_jsonl(Path(args.jd_file))}
    cvs = load_jsonl(Path(args.cv_file))
    print(f"[i] Đọc {len(jds)} JD, {len(cvs)} CV từ file input.")

    out_path = Path(args.out)
    done_ids = already_done_cv_ids(out_path)
    if done_ids:
        print(f"[i] Đã có {len(done_ids)} CV trong {out_path}, sẽ bỏ qua.")

    pipeline = RecruitmentPipeline(ModelSettings())

    write_header = not out_path.exists()
    failed = 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        processed = 0
        for cv in cvs:
            cv_id = cv["id"]
            if cv_id in done_ids:
                continue

            jd = jds.get(cv["jd_id"])
            if jd is None:
                print(f"[!] Bỏ qua {cv_id}: không tìm thấy jd_id={cv['jd_id']}")
                continue

            try:
                # MỘT lần chạy duy nhất là đủ: PipelineResult trả kèm
                # criterion_scores và dữ liệu đã parse. (Bản trước gọi parse
                # thêm lần nữa để lấy breakdown -> tốn gấp đôi chi phí API và
                # hai lần parse có thể ra kết quả khác nhau, khiến feature
                # không khớp với rule_based_score của chính dòng đó.)
                result = pipeline.run(
                    cv_raw_text=cv["cv_text"],
                    jd_raw_text=jd["jd_text"],
                    cv_id=cv_id,
                    jd_id=jd["id"],
                )
            except Exception as e:
                failed += 1
                print(f"[!] Lỗi ở {cv_id}: {e} -> nghỉ 5s, bỏ qua (chạy lại script sau để thử lại)")
                time.sleep(5)
                continue

            writer.writerow(build_row(cv, jd, result))
            f.flush()
            processed += 1
            print(
                f"[{processed}] {cv_id} -> final_score={result.evaluation.final_score} "
                f"(rule_based={result.rule_based_score}, intended={cv.get('fit_level')})"
            )

            time.sleep(args.delay)

    print(f"\nXong. Đã ghi thêm {processed} dòng vào {out_path}.")
    if failed:
        print(f"[!] {failed} CV bị lỗi -- chạy lại script để thử lại đúng những CV đó.")


if __name__ == "__main__":
    sys.exit(main())
