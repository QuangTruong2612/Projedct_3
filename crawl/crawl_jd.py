"""
Script kiểm tra robots.txt và (nếu được phép) crawl JD từ ITviec / TopCV
để phục vụ xây dựng dataset training cho đồ án Đánh giá CV.

Cách dùng:
    pip install requests beautifulsoup4 --break-system-packages
    python crawl_jd.py --site itviec --pages 5
    python crawl_jd.py --site topcv --pages 5

Lưu ý:
- Script LUÔN kiểm tra robots.txt trước khi crawl bất kỳ trang nào.
- Có delay giữa các request để không gây tải cho server.
- Chỉ dùng cho mục đích nghiên cứu/đồ án cá nhân, không redistribute dữ liệu.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "ThesisResearchBot/1.0 (+educational CV-JD matching research)"
HEADERS = {"User-Agent": USER_AGENT}
DELAY_SECONDS = 2.0  # khoảng nghỉ giữa các request, tránh spam server

SITE_CONFIG = {
    "itviec": {
        "base_url": "https://itviec.com",
        "list_path": "/it-jobs",
        # selector có thể lệch nếu ITviec đổi giao diện -> cần tự kiểm tra lại
        "job_link_selector": "a.job_link, a[data-search-job-link]",
        "title_selector": "h1, h2.title",
        "desc_selector": "div.job-description, div[class*='job-description']",
    },
    "topcv": {
        "base_url": "https://www.topcv.vn",
        "list_path": "/tim-viec-lam-it",
        "job_link_selector": "a.job-title, h3.title a",
        "title_selector": "h1",
        "desc_selector": "div.job-description, div[class*='job-description']",
    },
}


def check_robots(base_url: str, path: str) -> bool:
    """Đọc robots.txt thật của site và kiểm tra path có được phép crawl không."""
    rp = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as e:
        print(f"[!] Không đọc được {robots_url}: {e}")
        print("    -> Mặc định coi là KHÔNG được phép, dừng lại để an toàn.")
        return False

    allowed = rp.can_fetch(USER_AGENT, urljoin(base_url, path))
    print(f"[robots.txt] {robots_url}")
    print(f"    Path '{path}' -> {'CHO PHÉP' if allowed else 'KHÔNG CHO PHÉP'} crawl")
    return allowed


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[!] Lỗi khi tải {url}: {e}")
        return None


def crawl_site(site: str, max_pages: int) -> list[dict]:
    cfg = SITE_CONFIG[site]
    base_url = cfg["base_url"]

    # Bước 1: kiểm tra robots.txt cho trang danh sách job
    if not check_robots(base_url, cfg["list_path"]):
        print(f"[X] robots.txt của {site} không cho phép crawl '{cfg['list_path']}'.")
        print("    Dừng lại. Hãy thu thập thủ công (copy tay) thay vì crawl.")
        return []

    results = []
    for page in range(1, max_pages + 1):
        list_url = urljoin(base_url, f"{cfg['list_path']}?page={page}")
        html = fetch(list_url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        job_links = soup.select(cfg["job_link_selector"])
        if not job_links:
            print(f"[i] Không tìm thấy job link ở trang {page} — có thể selector đã lỗi thời "
                  f"(site đổi giao diện) hoặc trang cần JS render. Kiểm tra lại bằng tay.")
            break

        for a in job_links:
            href = a.get("href")
            if not href:
                continue
            job_url = urljoin(base_url, href)
            job_path = job_url.replace(base_url, "")

            # Bước 2: kiểm tra robots.txt cho TỪNG trang JD cụ thể trước khi vào
            if not check_robots(base_url, job_path):
                continue

            time.sleep(DELAY_SECONDS)
            job_html = fetch(job_url)
            if not job_html:
                continue

            job_soup = BeautifulSoup(job_html, "html.parser")
            title_el = job_soup.select_one(cfg["title_selector"])
            desc_el = job_soup.select_one(cfg["desc_selector"])

            results.append({
                "source": site,
                "url": job_url,
                "title": title_el.get_text(strip=True) if title_el else None,
                "description": desc_el.get_text("\n", strip=True) if desc_el else None,
            })
            print(f"[+] Đã lấy: {job_url}")

        time.sleep(DELAY_SECONDS)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["itviec", "topcv"], required=True)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = crawl_site(args.site, args.pages)

    out_path = args.out or f"jd_{args.site}.json"
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {len(data)} JD vào {out_path}")


if __name__ == "__main__":
    main()