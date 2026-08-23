import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.settings import EmbeddingProvider, ModelSettings


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Nhận list văn bản, trả về list vector tương ứng."""
        raise NotImplementedError


def _build_langchain_embeddings(settings: ModelSettings) -> Embeddings:
    """Trả về đối tượng Embeddings chuẩn của LangChain tương ứng với provider.

    Giống _build_chat_model() trong model_config.py: mọi Embeddings đều
    implement chung interface, nên phần gọi ở dưới không cần biết provider.
    """
    if settings.embedding_provider == EmbeddingProvider.VOYAGE:
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=settings.embedding_model_name)

    elif settings.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model_name)

    elif settings.embedding_provider == EmbeddingProvider.OPEN_SOURCE:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = settings.embedding_model_name or "intfloat/multilingual-e5-large"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise NotImplementedError(
            f"Embedding provider '{settings.embedding_provider}' chưa được hỗ trợ."
        )


def resolve_embedding_prefix(settings: ModelSettings) -> str:
    """Xác định prefix cần thêm vào trước mỗi văn bản khi embed.

    Họ model e5 (intfloat/multilingual-e5-*) được HUẤN LUYỆN kèm prefix
    "query: " / "passage: ". Bỏ prefix đi thì vector lệch khỏi phân phối
    lúc train và điểm similarity kém chính xác hơn.

    Toàn bộ so sánh trong MatchingService là ĐỐI XỨNG (skill với skill,
    chức danh với chức danh, mô tả project với yêu cầu JD) chứ không phải
    truy hồi tài liệu, nên theo khuyến nghị của tác giả e5 ta dùng
    "query: " cho CẢ HAI vế, không dùng "passage: ".

    MẶC ĐỊNH LÀ "none" — có lý do. Đo trên 20 cặp kỹ năng (10 cặp đồng nghĩa
    nên khớp, 10 cặp trái ngành không nên khớp) cho thấy prefix chỉ dịch toàn
    bộ điểm similarity lên khoảng +0.015 mà KHÔNG cải thiện độ tách biệt giữa
    hai nhóm (0.081 -> 0.078). Ở ngưỡng 0.85 hiện tại, kết quả khớp y hệt nhau.
    Vì đổi prefix làm lệch mọi điểm số (kéo theo training_data.csv không còn
    khớp với hệ thống), ta giữ mặc định "none" và để "auto" như một tuỳ chọn
    để thí nghiệm tiếp.

    Điều khiển qua EMBEDDING_PREFIX:
      - "none" (mặc định): không thêm gì
      - "auto": tự thêm "query: " nếu model thuộc họ e5
      - giá trị khác: dùng đúng chuỗi đó làm prefix
    """
    raw = settings.embedding_prefix or "none"
    keyword = raw.strip().lower()

    if keyword == "none":
        return ""
    if keyword != "auto":
        return raw  # trả nguyên văn, giữ cả khoảng trắng cuối

    # Chỉ áp dụng cho e5 — các họ model khác (bge, gte...) dùng quy ước
    # instruction riêng, thêm nhầm prefix sẽ phản tác dụng.
    if "e5" in (settings.embedding_model_name or "").lower():
        return "query: "
    return ""


class _BoundedCache:
    """Cache text -> vector, giới hạn kích thước để server chạy dài không phình
    bộ nhớ (LRU: hết chỗ thì bỏ mục lâu chưa dùng nhất).

    Có khoá vì endpoint chạy trong threadpool của FastAPI -> nhiều request
    có thể đọc/ghi cache cùng lúc.
    """

    def __init__(self, max_size: int = 8192):
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)  # vừa dùng -> đẩy về cuối hàng đợi LRU
            self.hits += 1
            return value

    def set(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)  # bỏ mục lâu chưa dùng nhất

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self.hits,
                "misses": self.misses,
            }


class LangChainEmbeddingClient(BaseEmbeddingClient):
    """Một class DUY NHẤT dùng chung cho mọi provider — giống
    LangChainLLMClient trong model_config.py.

    Bổ sung 2 thứ so với bản gọi thẳng LangChain:
    - prefix theo yêu cầu của model (xem resolve_embedding_prefix)
    - cache theo nội dung văn bản: cùng một kỹ năng xuất hiện ở nhiều CV/JD
      chỉ phải embed đúng 1 lần. Đặc biệt quan trọng với /rank, nơi danh
      sách kỹ năng của JD trước đây bị embed lại cho TỪNG CV.
    """

    def __init__(self, settings: ModelSettings, cache_size: int = 8192):
        self.embeddings: Embeddings = _build_langchain_embeddings(settings)
        self.prefix = resolve_embedding_prefix(settings)
        self._cache = _BoundedCache(max_size=cache_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self.prefix + t for t in texts]

        # Chỉ gọi model cho phần chưa có trong cache, đồng thời loại trùng
        # NGAY TRONG cùng một lượt gọi (một CV có thể liệt kê trùng kỹ năng).
        pending: list[str] = []
        pending_seen: set[str] = set()
        for key in keys:
            if key in pending_seen:
                continue
            if self._cache.get(key) is None:
                pending.append(key)
                pending_seen.add(key)

        fresh: dict[str, list[float]] = {}
        if pending:
            # embed_documents(): API chuẩn của LangChain cho việc embed nhiều
            # văn bản cùng lúc (khác với embed_query() dùng cho 1 câu truy vấn)
            vectors = self.embeddings.embed_documents(pending)
            fresh = dict(zip(pending, vectors))
            for key, vector in fresh.items():
                self._cache.set(key, vector)

        # Đọc lại từ `fresh` trước rồi mới tới cache: nếu batch lớn hơn sức
        # chứa cache, mục vừa ghi có thể đã bị LRU đẩy ra ngay lập tức.
        result: list[list[float]] = []
        for key in keys:
            vector = fresh.get(key)
            if vector is None:
                vector = self._cache.get(key)
            if vector is None:  # cực hiếm, chỉ xảy ra nếu cache quá nhỏ
                vector = self.embeddings.embed_documents([key])[0]
                self._cache.set(key, vector)
            result.append(vector)
        return result

    def cache_stats(self) -> dict[str, int]:
        """Dùng để đo hiệu quả cache (hits/misses) khi viết báo cáo."""
        return self._cache.stats()


def get_embedding_client(settings: ModelSettings) -> BaseEmbeddingClient:
    """Các module khác (matching_service.py) chỉ cần gọi hàm này,
    không cần biết chi tiết provider hay LangChain Embeddings nào
    đang chạy bên dưới."""
    return LangChainEmbeddingClient(settings)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:  # vector rỗng/toàn 0 -> tránh chia cho 0 (nan)
        return 0.0
    return float(np.dot(a, b) / denom)
