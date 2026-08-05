"""
Trích xuất text thô từ file CV upload (PDF/DOCX), phục vụ cho
CVParsingService xử lý ở bước tiếp theo.

Đây là bước tiền xử lý layout — KHÔNG dùng LLM ở đây để tiết kiệm
chi phí, chỉ đơn thuần lấy text ra rồi đưa qua bước parse bằng LLM.

Dùng LangChain Document Loaders (PyPDFLoader, Docx2txtLoader) thay vì
gọi trực tiếp pdfplumber/python-docx, để đồng bộ 1 hệ sinh thái LangChain
với model_config.py và embedding_client.py (đỡ phải quản lý nhiều bộ
thư viện khác mục đích riêng lẻ). Loader trả về list[Document] (mỗi
Document ứng với 1 trang/1 phần nội dung, kèm metadata) -> ta chỉ cần
nối page_content lại thành 1 chuỗi.
"""

import tempfile
from pathlib import Path

from fastapi import UploadFile


def extract_text_from_file(file: UploadFile) -> str:
    filename = (file.filename or "").lower()
    content = file.file.read()

    if filename.endswith(".pdf"):
        return _extract_with_loader(content, suffix=".pdf", loader_name="pdf")
    elif filename.endswith(".docx"):
        return _extract_with_loader(content, suffix=".docx", loader_name="docx")
    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Định dạng file '{filename}' chưa được hỗ trợ. "
            "Chỉ hỗ trợ .pdf, .docx, .txt"
        )


def _extract_with_loader(content: bytes, suffix: str, loader_name: str) -> str:
    """LangChain Document Loaders yêu cầu đường dẫn file trên đĩa, không
    nhận trực tiếp bytes trong bộ nhớ -> cần ghi ra file tạm trước."""

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if loader_name == "pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(tmp_path)
        else:  # docx
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(tmp_path)

        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)

        if not text.strip():
            raise ValueError(
                f"Không trích xuất được text từ file {suffix}. Có thể đây là "
                "file scan ảnh (PDF), cần xử lý qua OCR (chưa hỗ trợ trong bản này)."
            )

        return text
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # luôn dọn file tạm dù thành công hay lỗi