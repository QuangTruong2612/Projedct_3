"""
Sinh CV (Curriculum Vitae) tiếng Việt tổng hợp bằng Claude API, mỗi CV được
viết CÓ CHỦ ĐÍCH khớp với 1 JD cụ thể (từ jd_synthetic.jsonl) ở 1 trong 3 mức
độ: tốt (good_fit) / một phần (partial_fit) / kém (poor_fit).

Mục đích: có input đa dạng độ khớp để sau này chạy qua pipeline hiện có
(CVParsingService + MatchingService + ScoringService) sinh label (final_score)
dùng train model ranker nhẹ.

Cách dùng:
    pip install anthropic --break-system-packages
    set ANTHROPIC_API_KEY=sk-ant-...

    python generate_synthetic_cv.py --jd_file jd_synthetic.jsonl --per_jd 2 \
        --out cv_synthetic.jsonl

Với --per_jd 2: mỗi JD sẽ được ghép với 2 CV, mỗi CV random 1 trong 3 mức độ
khớp, chọn ngẫu nhiên 2 mức khác nhau trong {good_fit, partial_fit, poor_fit}
-> đảm bảo toàn bộ dataset có đủ 3 loại, không lệch hẳn về 1 loại.
"""

import argparse
import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic

MODEL = "claude-sonnet-5"
FIT_LEVELS = ["good_fit", "partial_fit", "poor_fit"]

FIT_INSTRUCTIONS = {
    "good_fit": (
        "CV này PHẢI khớp TỐT với JD: ứng viên đáp ứng khoảng 80-100% kỹ năng bắt buộc, "
        "có số năm kinh nghiệm bằng hoặc nhỉnh hơn yêu cầu, học vấn phù hợp hoặc cao hơn, "
        "và lý tưởng là có thêm 1-2 kỹ năng ưu tiên/cộng điểm trong JD. Đây là ứng viên "
        "mà nhà tuyển dụng sẽ muốn phỏng vấn ngay."
    ),
    "partial_fit": (
        "CV này PHẢI chỉ khớp MỘT PHẦN với JD: ứng viên có nền tảng cùng lĩnh vực nhưng "
        "thiếu hụt rõ ràng ở 1-2 điểm quan trọng — ví dụ: chỉ đáp ứng ~40-60% kỹ năng bắt buộc, "
        "HOẶC số năm kinh nghiệm thấp hơn yêu cầu 1-2 năm, HOẶC học vấn không đúng chuyên ngành, "
        "HOẶC kỹ năng đúng nhưng chưa từng làm đúng loại hình dự án JD yêu cầu. "
        "Chỉ chọn 1-2 điểm thiếu hụt, không thiếu toàn bộ."
    ),
    "poor_fit": (
        "CV này PHẢI khớp KÉM với JD: ứng viên đến từ chuyên môn/lĩnh vực khá khác biệt, "
        "chỉ trùng một vài kỹ năng nền tảng chung chung (ví dụ: đều biết Git, đều làm việc "
        "nhóm) nhưng KHÔNG có kinh nghiệm hoặc kỹ năng cốt lõi JD yêu cầu, số năm kinh nghiệm "
        "hoặc học vấn không phù hợp với cấp bậc JD đăng tuyển. Đây là ứng viên nhà tuyển dụng "
        "sẽ loại ở vòng sàng lọc CV."
    ),
}

SYSTEM_PROMPT = """Bạn là một sinh viên/người đi làm đang viết CV xin việc thật, không phải nhà tuyển dụng.
Nhiệm vụ: viết 1 CV tiếng Việt HOÀN CHỈNH, thực tế, tự nhiên như CV thật ứng viên nộp.

Yêu cầu bắt buộc:
- Tên ứng viên, email, số điện thoại PHẢI là thông tin HƯ CẤU, không dùng người/công ty thật.
  Email nên dùng dạng như "tenrieng.hoten@gmail.com" (tên hư cấu), số điện thoại dùng đầu số
  Việt Nam hợp lệ nhưng là số bịa (ví dụ 09xxxxxxxx với các chữ số ngẫu nhiên).
- CV cần có đầy đủ các mục: Thông tin cá nhân, Mục tiêu nghề nghiệp (ngắn), Kỹ năng,
  Kinh nghiệm làm việc (nếu có), Dự án cá nhân/học tập (Projects), Học vấn.
- Nếu ứng viên có nhiều bằng cấp, liệt kê đủ. Nếu có dự án, ghi rõ tech stack dùng và
  đây là dự án học tập/cá nhân hay dự án công việc chính thức.
- Độ dài vừa phải (300-500 từ), viết trọn vẹn, không bỏ dở câu.
- CHỈ trả về nội dung CV, không thêm lời dẫn, không markdown code fence.
"""


def build_user_prompt(jd_text: str, fit_level: str) -> str:
    jd_excerpt = jd_text[:2000]  # đủ context, tránh prompt quá dài
    return f"""Dưới đây là JD mà ứng viên đang ứng tuyển:

---
{jd_excerpt}
---

{FIT_INSTRUCTIONS[fit_level]}

Hãy viết CV của ứng viên này."""


def generate_one(client: Anthropic, jd_text: str, fit_level: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(jd_text, fit_level)}],
        # Không truyền temperature -> Claude Sonnet 5 không hỗ trợ tham số này.
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("CV bị cắt cụt (max_tokens) — cần tăng max_tokens.")
    return text


def load_jds(jd_file: Path) -> list[dict]:
    jds = []
    with open(jd_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jds.append(json.loads(line))
    return jds


def already_done_pairs(out_path: Path) -> set[tuple[str, str]]:
    """Trả về set (jd_id, fit_level) đã sinh, để chạy tiếp không bị trùng khi resume."""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    done.add((d["jd_id"], d["fit_level"]))
    return done


JD_FILE = Path("jd_synthetic.jsonl")
OUT_FILE = Path("cv_synthetic.jsonl")
# Nghỉ giữa 2 lượt gọi API để không chạm rate limit
DELAY_SECONDS = 0.5
# Cố định seed để việc chọn fit_level lặp lại được khi chạy tiếp
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    # Chỉ giữ tham số THỰC SỰ thay đổi giữa các lần chạy. Số CV mỗi JD là thứ
    # đáng chỉnh nhất: tăng từ 2 lên 5 sẽ cho gấp 10 số cặp để đo xếp hạng.
    parser.add_argument("--per_jd", type=int, default=2, help="số CV sinh cho mỗi JD")
    args = parser.parse_args()

    jds = load_jds(JD_FILE)
    print(f"[i] Đã đọc {len(jds)} JD từ {JD_FILE}")

    out_path = OUT_FILE
    done_pairs = already_done_pairs(out_path)
    if done_pairs:
        print(f"[i] Đã có {len(done_pairs)} cặp (jd_id, fit_level) trong {out_path}, sẽ bỏ qua các cặp này.")

    client = Anthropic()

    rng = random.Random(RANDOM_SEED)

    total_generated = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for jd in jds:
            jd_id = jd["id"]
            jd_text = jd["jd_text"]

            # Chọn ngẫu nhiên (nhưng lặp lại được nhờ seed) per_jd mức fit khác nhau
            chosen_levels = rng.sample(FIT_LEVELS, k=min(args.per_jd, len(FIT_LEVELS)))

            for fit_level in chosen_levels:
                if (jd_id, fit_level) in done_pairs:
                    continue

                try:
                    cv_text = generate_one(client, jd_text, fit_level)
                except Exception as e:
                    print(f"[!] Lỗi ở JD {jd_id} / {fit_level}: {e} -> nghỉ 5s rồi bỏ qua, chạy lại script sau để retry")
                    time.sleep(5)
                    continue

                record = {
                    "id": f"cv_{jd_id}_{fit_level}",
                    "jd_id": jd_id,
                    "jd_role": jd.get("role"),
                    "fit_level": fit_level,  # nhãn CHỦ Ý khi sinh -> dùng để kiểm tra chéo, KHÔNG dùng trực tiếp làm final_score
                    "cv_text": cv_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                total_generated += 1
                print(f"[{total_generated}] {jd_id} | {fit_level} | {jd.get('role')}")

                time.sleep(DELAY_SECONDS)

    print(f"\nXong. Tổng số CV mới sinh: {total_generated}")
    print(f"Tổng số dòng hiện có trong {out_path}: {len(already_done_pairs(out_path))}")


if __name__ == "__main__":
    main()