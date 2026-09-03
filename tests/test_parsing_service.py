"""Test cho việc cắt bớt văn bản đầu vào trước khi gửi cho LLM.

`MAX_INPUT_TOKENS` trước đây là setting chết: có trong settings và README
nhưng không nơi nào dùng, nên CV dài bao nhiêu cũng bị nhồi hết vào prompt.
"""

import pytest

from app.core.settings import ModelSettings
from app.schemas.models import ParsedCV, ParsedJD
from app.services.parsing_service import (
    CHARS_PER_TOKEN,
    TRUNCATION_NOTICE,
    CVParsingService,
    JDParsingService,
    truncate_for_llm,
)


class SpyLLM:
    """Ghi lại prompt được gửi đi để test kiểm tra độ dài."""

    def __init__(self):
        self.prompts: list[str] = []

    def extract_structured(self, prompt: str, schema):
        self.prompts.append(prompt)
        return schema()


# ---------------------------------------------------------------- truncate

def test_van_ban_ngan_khong_bi_cat():
    text = "CV ngắn"
    out, cut = truncate_for_llm(text, max_tokens=1000)
    assert out == text
    assert cut is False


def test_van_ban_dai_bi_cat():
    text = "x" * 10_000
    out, cut = truncate_for_llm(text, max_tokens=100)  # ngân sách 300 ký tự
    assert cut is True
    assert len(out) <= 100 * CHARS_PER_TOKEN
    assert TRUNCATION_NOTICE in out


def test_giu_lai_ca_dau_va_cuoi():
    """Đầu CV có thông tin cá nhân/kỹ năng, cuối CV có học vấn -> phải giữ cả hai."""
    text = "BẮT_ĐẦU" + ("x" * 5000) + "KẾT_THÚC"
    out, cut = truncate_for_llm(text, max_tokens=200)
    assert cut is True
    assert out.startswith("BẮT_ĐẦU")
    assert out.endswith("KẾT_THÚC")


def test_max_tokens_bang_0_thi_khong_gioi_han():
    text = "x" * 10_000
    out, cut = truncate_for_llm(text, max_tokens=0)
    assert out == text
    assert cut is False


def test_do_dai_dung_bang_ngan_sach_thi_khong_cat():
    budget = 50 * CHARS_PER_TOKEN
    text = "x" * budget
    out, cut = truncate_for_llm(text, max_tokens=50)
    assert cut is False
    assert out == text


# ---------------------------------------------------------------- service

@pytest.mark.parametrize("service_cls", [CVParsingService, JDParsingService])
def test_service_that_su_cat_bot_truoc_khi_goi_llm(service_cls):
    llm = SpyLLM()
    settings = ModelSettings()
    settings.max_input_tokens = 100  # ngân sách 300 ký tự

    service = service_cls(llm, settings)
    service.parse("y" * 50_000)

    assert len(llm.prompts) == 1
    # prompt = phần hướng dẫn + nội dung đã cắt; nội dung không được vượt ngân sách
    assert llm.prompts[0].count("y") <= 100 * CHARS_PER_TOKEN


def test_cv_ngan_duoc_gui_nguyen_ven():
    llm = SpyLLM()
    settings = ModelSettings()
    settings.max_input_tokens = 4000

    CVParsingService(llm, settings).parse("Nguyễn Văn A - Python Developer")

    assert "Nguyễn Văn A - Python Developer" in llm.prompts[0]
    assert TRUNCATION_NOTICE not in llm.prompts[0]


def test_service_tra_ve_dung_schema():
    llm = SpyLLM()
    assert isinstance(CVParsingService(llm).parse("abc"), ParsedCV)
    assert isinstance(JDParsingService(llm).parse("abc"), ParsedJD)


def test_ngan_sach_nho_hon_dong_ghi_chu_van_khong_vuot_gioi_han():
    """Nếu ngân sách nhỏ hơn độ dài TRUNCATION_NOTICE, việc chèn ghi chú sẽ
    khiến kết quả dài HƠN giới hạn -- đúng thứ mà hàm này phải bảo đảm không
    xảy ra."""
    for max_tokens in (1, 2, 5, 10, 15):
        out, cut = truncate_for_llm("x" * 10_000, max_tokens)
        assert cut is True
        assert len(out) <= max_tokens * CHARS_PER_TOKEN, (
            f"max_tokens={max_tokens}: trả về {len(out)} ký tự, "
            f"vượt ngân sách {max_tokens * CHARS_PER_TOKEN}"
        )


def test_moi_muc_ngan_sach_deu_ton_trong_gioi_han():
    text = "y" * 50_000
    for max_tokens in (20, 50, 100, 500, 4000):
        out, _ = truncate_for_llm(text, max_tokens)
        assert len(out) <= max_tokens * CHARS_PER_TOKEN
