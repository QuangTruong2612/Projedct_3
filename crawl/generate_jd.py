"""
Sinh JD (Job Description) tiếng Việt tổng hợp (synthetic) bằng Claude API,
dùng để bổ sung dataset training cho bài toán CV-JD matching khi không
crawl được JD thật từ ITviec/TopCV (bị chặn ở robots.txt).

Cách dùng:
    pip install anthropic --break-system-packages
    set ANTHROPIC_API_KEY=sk-ant-...      (Windows CMD)
    $env:ANTHROPIC_API_KEY="sk-ant-..."    (PowerShell)

    python generate_synthetic_jd.py --count 200 --out jd_synthetic.jsonl

Ghi chú:
- Đa dạng hoá bằng cách random hoá (vị trí, cấp bậc, ngành, loại công ty,
  văn phong) TRƯỚC khi gọi model, thay vì chỉ dựa vào temperature — vì
  Claude Sonnet 5 trở lên không nhận tham số temperature (xem
  app/core/model_config.py trong project gốc, hàm _build_chat_model).
- Lưu theo dạng .jsonl, ghi từng dòng ngay sau khi sinh xong -> nếu script
  bị dừng giữa chừng (rate limit, mất mạng...) không mất dữ liệu đã có.
- Công ty trong JD là hư cấu (system prompt yêu cầu rõ) để tránh dùng tên
  công ty thật ngoài đời.
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

ROLES = [
    "Backend Developer", "Frontend Developer", "Fullstack Developer",
    "Mobile Developer (Android)", "Mobile Developer (iOS)",
    "Data Engineer", "Data Analyst", "Data Scientist",
    "AI Engineer", "Machine Learning Engineer", "MLOps Engineer",
    "DevOps Engineer", "Cloud Engineer", "System Administrator",
    "QA Engineer / Tester", "QA Automation Engineer",
    "Business Analyst (IT)", "Product Manager (Tech)",
    "Security Engineer", "Database Administrator",
    "Embedded Software Engineer", "Game Developer",
]

SENIORITY = [
    "Intern/Thực tập sinh", "Fresher", "Junior (0-2 năm)",
    "Mid-level (2-4 năm)", "Senior (4-7 năm)", "Senior/Lead (7+ năm)",
]

DOMAINS = [
    "Fintech", "E-commerce", "Ngân hàng", "Bảo hiểm", "Giáo dục (Edtech)",
    "Y tế (Healthtech)", "Logistics", "Gaming", "SaaS B2B",
    "Outsourcing/IT dịch vụ", "Startup công nghệ giai đoạn đầu",
    "Bất động sản (Proptech)", "Sản xuất/Manufacturing", "AI/Data platform",
]

COMPANY_TYPES = [
    "công ty outsourcing quy mô 500+ nhân viên",
    "startup nhỏ dưới 30 người",
    "công ty product quy mô vừa (100-300 người)",
    "tập đoàn lớn có nhiều chi nhánh",
    "công ty nước ngoài (Nhật/Hàn/Mỹ) đặt văn phòng tại Việt Nam",
]

STYLES = [
    "liệt kê bằng bullet point rõ ràng, ngắn gọn",
    "viết dạng đoạn văn liền mạch, ít bullet",
    "chia rõ 2 phần: kỹ năng bắt buộc và kỹ năng ưu tiên/cộng điểm",
    "có kèm mức lương và quyền lợi (benefits)",
    "không đề cập mức lương, tập trung mô tả công việc và yêu cầu",
]

SYSTEM_PROMPT = """Bạn là chuyên gia tuyển dụng IT tại Việt Nam, có nhiệm vụ soạn Job Description (JD)
thật tự nhiên và thực tế, giống như JD thật được đăng trên ITviec/TopCV/VietnamWorks.

Yêu cầu:
- Viết hoàn toàn bằng tiếng Việt (có thể chèn thuật ngữ kỹ thuật tiếng Anh khi tự nhiên).
- Tên công ty PHẢI là hư cấu, không dùng tên công ty thật ngoài đời.
- Nội dung phải thực tế, chi tiết, có yêu cầu kỹ năng cụ thể (framework, ngôn ngữ,
  công cụ...) phù hợp với vị trí và cấp bậc được yêu cầu, không viết chung chung.
- Chỉ trả về nội dung JD, không thêm lời dẫn, không thêm markdown code fence.
- Độ dài vừa phải, khoảng 400-600 từ (giống JD thật, không lan man), và LUÔN
  viết trọn vẹn hết phần yêu cầu kỹ năng + kết thúc rõ ràng, không bỏ dở câu.
"""


def build_user_prompt(role: str, seniority: str, domain: str, company_type: str, style: str) -> str:
    return f"""Hãy viết 1 JD tuyển dụng cho vị trí: {role}
Cấp bậc: {seniority}
Ngành/lĩnh vực công ty: {domain}
Loại công ty: {company_type}
Văn phong trình bày: {style}

JD cần có đầy đủ: tên vị trí, mô tả công việc, yêu cầu kỹ năng bắt buộc,
kỹ năng ưu tiên (nếu phù hợp với cấp bậc), số năm kinh nghiệm tối thiểu,
yêu cầu bằng cấp (nếu có), địa điểm làm việc (một thành phố lớn ở Việt Nam)."""


def generate_one(client: Anthropic, role, seniority, domain, company_type, style) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": build_user_prompt(role, seniority, domain, company_type, style),
        }],
        # Lưu ý: KHÔNG truyền temperature -> Claude Sonnet 5 không hỗ trợ tham số này.
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    if resp.stop_reason == "max_tokens":
        # JD bị cắt cụt giữa chừng vì hết token budget -> không nên giữ lại,
        # báo lỗi để hàm gọi retry hoặc bỏ qua record này.
        raise RuntimeError("Output bị cắt cụt (max_tokens) — cần tăng max_tokens hoặc rút gọn yêu cầu.")
    return text


def already_done(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    with open(out_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


OUT_FILE = Path("jd_synthetic.jsonl")
# Nghỉ giữa 2 lượt gọi API để không chạm rate limit
DELAY_SECONDS = 0.5


def main():
    parser = argparse.ArgumentParser()
    # Chỉ giữ tham số THỰC SỰ thay đổi giữa các lần chạy; đường dẫn và độ trễ
    # là hằng số của dự án (xem OUT_FILE / DELAY_SECONDS).
    parser.add_argument("--count", type=int, default=200, help="số JD cần sinh")
    args = parser.parse_args()

    out_path = OUT_FILE
    start_idx = already_done(out_path)  # cho phép chạy tiếp nếu bị dừng giữa chừng
    if start_idx:
        print(f"[i] Đã có {start_idx} JD trong {out_path}, tiếp tục sinh thêm...")

    client = Anthropic()  # đọc ANTHROPIC_API_KEY từ biến môi trường

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(start_idx, args.count):
            role = random.choice(ROLES)
            seniority = random.choice(SENIORITY)
            domain = random.choice(DOMAINS)
            company_type = random.choice(COMPANY_TYPES)
            style = random.choice(STYLES)

            try:
                jd_text = generate_one(client, role, seniority, domain, company_type, style)
            except Exception as e:
                print(f"[!] Lỗi ở JD #{i+1}: {e} -> nghỉ 5s rồi thử tiếp")
                time.sleep(5)
                continue

            record = {
                "id": f"synthetic_jd_{i+1:04d}",
                "role": role,
                "seniority": seniority,
                "domain": domain,
                "company_type": company_type,
                "style": style,
                "jd_text": jd_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{args.count}] {role} | {seniority} | {domain}")

            time.sleep(DELAY_SECONDS)

    print(f"\nXong. Tổng số JD trong {out_path}: {already_done(out_path)}")


if __name__ == "__main__":
    main()