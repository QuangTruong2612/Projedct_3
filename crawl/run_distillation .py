"""
Bước "distillation": đưa từng cặp (CV, JD) synthetic qua CHÍNH pipeline hiện
có của project (parse -> match rule-based -> LLM scoring) để lấy:
  - final_score (LLM đã tinh chỉnh) -> dùng làm LABEL cho model ranker
  - criterion_scores (skills/experience/education) -> dùng làm FEATURE

Kết quả lưu ra 1 file CSV phẳng, sẵn sàng đưa vào bước train XGBoost/LightGBM.

QUAN TRỌNG: script này phải chạy được từ THƯ MỤC GỐC của project (nơi có
folder `app/`), vì nó import trực tiếp app.services.pipeline. Copy file này
vào thư mục gốc project (ngang hàng với folder app/) trước khi chạy.

Cách dùng:
    cd thực_tập_tốt_nghiệp        (thư mục gốc project, có sẵn .env)
    python run_distillation.py --jd_file jd_synthetic.jsonl \
        --cv_file cv_synthetic.jsonl --out training_data.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # đọc .env của project (API key, LLM_PROVIDER...) TRƯỚC khi import app.*

from app.core.settings import ModelSettings          # noqa: E402
from app.services.pipeline import RecruitmentPipeline  # noqa: E402

FIELDNAMES = [
    "cv_id", "jd_id", "jd_role", "fit_level_intended",
    "skill_score", "experience_score", "education_score",
    "rule_based_score", "final_score",
    "strengths", "gaps", "explanation",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd_file", default="jd_synthetic.jsonl")
    parser.add_argument("--cv_file", default="cv_synthetic.jsonl")
    parser.add_argument("--out", default="training_data.csv")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    jds = {jd["id"]: jd for jd in load_jsonl(Path(args.jd_file))}
    cvs = load_jsonl(Path(args.cv_file))
    print(f"[i] Đọc {len(jds)} JD, {len(cvs)} CV từ file input.")

    out_path = Path(args.out)
    done_ids = already_done_cv_ids(out_path)
    if done_ids:
        print(f"[i] Đã có {len(done_ids)} CV được xử lý trong {out_path}, sẽ bỏ qua.")

    pipeline = RecruitmentPipeline(ModelSettings())

    write_header = not out_path.exists()
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
                print(f"[!] Bỏ qua {cv_id}: không tìm thấy jd_id={cv['jd_id']} trong jd_file")
                continue

            try:
                result = pipeline.run(
                    cv_raw_text=cv["cv_text"],
                    jd_raw_text=jd["jd_text"],
                    cv_id=cv_id,
                    jd_id=jd["id"],
                )
            except Exception as e:
                print(f"[!] Lỗi ở {cv_id}: {e} -> nghỉ 5s, bỏ qua (chạy lại script sau để retry)")
                time.sleep(5)
                continue

            # PipelineResult chỉ trả (evaluation, rule_based_score), không có breakdown
            # theo từng tiêu chí -> gọi lại parser/matcher riêng để lấy skill/experience/
            # education score làm feature cho model ranker. Có gọi trùng LLM parse 2 lần
            # (1 lần trong pipeline.run(), 1 lần ở đây) -> tốn thêm chi phí API, chấp nhận
            # được vì đây là bước sinh dataset một lần, không phải chạy production.
            cv_parsed = pipeline.cv_parser.parse(cv["cv_text"])
            jd_parsed = pipeline.jd_parser.parse(jd["jd_text"])
            match_result = pipeline.matcher.match(cv_parsed, jd_parsed, cv_id=cv_id, jd_id=jd["id"])
            crit = {cs.criterion: cs.score for cs in match_result.criterion_scores}

            row = {
                "cv_id": cv_id,
                "jd_id": jd["id"],
                "jd_role": jd.get("role"),
                "fit_level_intended": cv.get("fit_level"),
                "skill_score": crit.get("skills"),
                "experience_score": crit.get("experience"),
                "education_score": crit.get("education"),
                "rule_based_score": result.rule_based_score,
                "final_score": result.evaluation.final_score,
                "strengths": " | ".join(result.evaluation.strengths),
                "gaps": " | ".join(result.evaluation.gaps),
                "explanation": result.evaluation.explanation,
            }
            writer.writerow(row)
            f.flush()
            processed += 1
            print(f"[{processed}] {cv_id} -> final_score={result.evaluation.final_score} "
                  f"(rule_based={result.rule_based_score}, intended={cv.get('fit_level')})")

            time.sleep(args.delay)

    print(f"\nXong. Dữ liệu training đã lưu ở {out_path}")


if __name__ == "__main__":
    main()