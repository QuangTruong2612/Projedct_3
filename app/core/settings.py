import os
from dataclasses import dataclass
from enum import Enum


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPEN_SOURCE = "open_source"  # ví dụ: tự host qua Ollama
 
 
class EmbeddingProvider(str, Enum):
    VOYAGE = "voyage"
    OPENAI = "openai"
    OPEN_SOURCE = "open_source"  # ví dụ: bge-m3, multilingual-e5
 
 
@dataclass
class ModelSettings:
    """Cấu hình model, đọc từ biến môi trường để dễ đổi mà không sửa code.

    Dùng chung cho cả model_config.py (phần llm_*) và embedding_client.py
    (phần embedding_*) -> chỉ cần sửa 1 chỗ khi thêm setting mới.
    """

    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model_name: str = "claude-3-5-sonnet-latest"
    llm_temperature: float = 0.0

    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPEN_SOURCE
    embedding_model_name: str = "intfloat/multilingual-e5-large"

    skill_match_threshold: float = 0.85
    max_input_tokens: int = 4000

    def __post_init__(self):
        # Đọc llm_provider an toàn
        llm_val = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
        try:
            self.llm_provider = LLMProvider(llm_val)
        except ValueError:
            self.llm_provider = LLMProvider.ANTHROPIC

        # Đọc llm_model_name
        self.llm_model_name = (os.getenv("LLM_MODEL_NAME") or "claude-3-5-sonnet-latest").strip()

        # Đọc llm_temperature
        try:
            temp_val = os.getenv("LLM_TEMPERATURE")
            self.llm_temperature = float(temp_val) if temp_val and temp_val.strip() else 0.0
        except ValueError:
            self.llm_temperature = 0.0

        # Đọc embedding_provider an toàn
        emb_val = (os.getenv("EMBEDDING_PROVIDER") or "open_source").strip().lower()
        try:
            self.embedding_provider = EmbeddingProvider(emb_val)
        except ValueError:
            self.embedding_provider = EmbeddingProvider.OPEN_SOURCE

        # Đọc embedding_model_name
        self.embedding_model_name = (os.getenv("EMBEDDING_MODEL_NAME") or "intfloat/multilingual-e5-large").strip()

        # Đọc threshold & tokens
        try:
            thresh_val = os.getenv("SKILL_MATCH_THRESHOLD")
            self.skill_match_threshold = float(thresh_val) if thresh_val and thresh_val.strip() else 0.85
        except ValueError:
            self.skill_match_threshold = 0.85

        try:
            tok_val = os.getenv("MAX_INPUT_TOKENS")
            self.max_input_tokens = int(tok_val) if tok_val and tok_val.strip() else 4000
        except ValueError:
            self.max_input_tokens = 4000