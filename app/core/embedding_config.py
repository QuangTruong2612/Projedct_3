from abc import ABC, abstractmethod
 
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
 
        # multilingual-e5 yêu cầu prefix "query: "/"passage: " để đạt hiệu quả
        # tốt nhất theo khuyến nghị của tác giả model -> truyền qua encode_kwargs
        model_name = settings.embedding_model_name or "intfloat/multilingual-e5-large"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
 
    else:
        raise NotImplementedError(
            f"Embedding provider '{settings.embedding_provider}' chưa được hỗ trợ."
        )
 
 
class LangChainEmbeddingClient(BaseEmbeddingClient):
    """Một class DUY NHẤT dùng chung cho mọi provider — giống
    LangChainLLMClient trong model_config.py."""
 
    def __init__(self, settings: ModelSettings):
        self.embeddings: Embeddings = _build_langchain_embeddings(settings)
 
    def embed(self, texts: list[str]) -> list[list[float]]:
        # embed_documents(): API chuẩn của LangChain cho việc embed nhiều
        # văn bản cùng lúc (khác với embed_query() dùng cho 1 câu truy vấn)
        return self.embeddings.embed_documents(texts)
 
 
def get_embedding_client(settings: ModelSettings) -> BaseEmbeddingClient:
    """Các module khác (matching_service.py) chỉ cần gọi hàm này,
    không cần biết chi tiết provider hay LangChain Embeddings nào
    đang chạy bên dưới."""
    return LangChainEmbeddingClient(settings)
 
 
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))