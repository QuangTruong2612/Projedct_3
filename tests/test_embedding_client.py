"""Test cho lớp embedding client: prefix của model e5 + cache theo nội dung.

Không gọi model thật — thay LangChain Embeddings bằng một đối tượng giả
đếm số văn bản đã embed, nhờ đó kiểm chứng được cache có hoạt động không.
"""

import pytest

from app.core.embedding_config import (
    LangChainEmbeddingClient,
    cosine_similarity,
    resolve_embedding_prefix,
)
from app.core.settings import ModelSettings


class FakeEmbeddings:
    """Embeddings giả: mỗi văn bản thành 1 vector suy ra từ độ dài chuỗi.
    Quan trọng nhất là nó GHI LẠI mọi văn bản từng được yêu cầu embed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    @property
    def embedded_texts(self) -> list[str]:
        return [t for call in self.calls for t in call]


@pytest.fixture
def client(monkeypatch) -> LangChainEmbeddingClient:
    """Client dùng embeddings giả và TẮT prefix, để nhóm test cache kiểm tra
    đúng một thứ là hành vi cache (prefix có nhóm test riêng ở dưới)."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(
        "app.core.embedding_config._build_langchain_embeddings",
        lambda settings: fake,
    )
    monkeypatch.setenv("EMBEDDING_PREFIX", "none")
    c = LangChainEmbeddingClient(ModelSettings())
    c.fake = fake  # type: ignore[attr-defined]  # để test đọc được số liệu
    return c


# ---------- prefix ----------

def test_mac_dinh_khong_them_prefix(monkeypatch):
    """Mặc định là "none" để không làm lệch điểm số so với training_data.csv
    (đo được: prefix không cải thiện độ tách biệt -- xem docstring của
    resolve_embedding_prefix)."""
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
    monkeypatch.delenv("EMBEDDING_PREFIX", raising=False)
    assert resolve_embedding_prefix(ModelSettings()) == ""


def test_auto_them_prefix_query_cho_model_e5(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
    monkeypatch.setenv("EMBEDDING_PREFIX", "auto")
    assert resolve_embedding_prefix(ModelSettings()) == "query: "


def test_auto_khong_them_gi_cho_model_khong_phai_e5(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_PREFIX", "auto")
    assert resolve_embedding_prefix(ModelSettings()) == ""


def test_co_the_dat_prefix_tuy_y(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PREFIX", "passage: ")
    assert resolve_embedding_prefix(ModelSettings()) == "passage: "


def test_prefix_thuc_su_duoc_gan_vao_van_ban(client, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
    client.prefix = "query: "
    client.embed(["Python"])
    assert client.fake.embedded_texts == ["query: Python"]


# ---------- cache ----------

def test_goi_lai_cung_van_ban_khong_embed_lai(client):
    client.embed(["Python", "Docker"])
    client.embed(["Python", "Docker"])
    assert client.fake.embedded_texts == ["Python", "Docker"]


def test_van_ban_trung_trong_cung_mot_lo_chi_embed_mot_lan(client):
    client.embed(["Python", "Python", "SQL"])
    assert sorted(client.fake.embedded_texts) == ["Python", "SQL"]


def test_ket_qua_dung_thu_tu_va_dung_gia_tri_du_co_cache(client):
    first = client.embed(["Python", "Docker"])
    second = client.embed(["Docker", "Kubernetes", "Python"])
    assert second[0] == first[1]  # Docker
    assert second[2] == first[0]  # Python
    assert client.fake.embedded_texts == ["Python", "Docker", "Kubernetes"]


def test_lo_lon_hon_suc_chua_cache_van_tra_du_ket_qua(monkeypatch):
    """Cache LRU nhỏ có thể đẩy mục vừa ghi ra ngay -> phải không mất dữ liệu."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(
        "app.core.embedding_config._build_langchain_embeddings",
        lambda settings: fake,
    )
    c = LangChainEmbeddingClient(ModelSettings(), cache_size=2)
    texts = [f"skill-{i}" for i in range(10)]
    result = c.embed(texts)
    assert len(result) == 10
    assert all(v is not None for v in result)


def test_cache_stats_dem_dung_hit_va_miss(client):
    client.embed(["Python"])
    client.embed(["Python"])
    stats = client.cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_embed_danh_sach_rong(client):
    assert client.embed([]) == []
    assert client.fake.calls == []


# ---------- cosine ----------

def test_cosine_vector_khong_tra_ve_0_thay_vi_nan():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_vector_giong_het_nhau_bang_1():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
