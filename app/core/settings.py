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
    # "auto" | "none" | chuỗi prefix tuỳ ý — xem resolve_embedding_prefix()
    embedding_prefix: str = "none"

    # Chỉ dùng để hiển thị "khớp rõ X/Y kỹ năng" cho HR, KHÔNG dùng để tính điểm
    skill_match_threshold: float = 0.85
    # Sàn nhiễu của embedding: similarity dưới mức này coi như không liên quan.
    # Hiệu chỉnh từ ~4.000 cặp kỹ năng ghép ngẫu nhiên (trung vị 0.79).
    similarity_floor: float = 0.80
    # Phần điểm kinh nghiệm giữ lại khi công việc hoàn toàn trái ngành
    experience_relevance_base: float = 0.3
    max_input_tokens: int = 4000

    # Số CV được xử lý song song trong /rank. Mỗi luồng gọi LLM riêng nên
    # đặt quá cao dễ chạm rate limit của provider.
    rank_max_workers: int = 4

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

        # Đọc embedding_prefix. KHÔNG .strip() và KHÔNG .lower(): prefix thật
        # gần như luôn kết thúc bằng khoảng trắng ("query: "), strip đi là hỏng.
        prefix_val = os.getenv("EMBEDDING_PREFIX")
        self.embedding_prefix = prefix_val if prefix_val and prefix_val.strip() else "none"

        # Đọc threshold & tokens
        try:
            thresh_val = os.getenv("SKILL_MATCH_THRESHOLD")
            self.skill_match_threshold = float(thresh_val) if thresh_val and thresh_val.strip() else 0.85
        except ValueError:
            self.skill_match_threshold = 0.85

        try:
            floor_val = os.getenv("SIMILARITY_FLOOR")
            self.similarity_floor = float(floor_val) if floor_val and floor_val.strip() else 0.80
        except ValueError:
            self.similarity_floor = 0.80

        try:
            base_val = os.getenv("EXPERIENCE_RELEVANCE_BASE")
            self.experience_relevance_base = (
                float(base_val) if base_val and base_val.strip() else 0.3
            )
        except ValueError:
            self.experience_relevance_base = 0.3

        try:
            tok_val = os.getenv("MAX_INPUT_TOKENS")
            self.max_input_tokens = int(tok_val) if tok_val and tok_val.strip() else 4000
        except ValueError:
            self.max_input_tokens = 4000

        try:
            workers_val = os.getenv("RANK_MAX_WORKERS")
            self.rank_max_workers = int(workers_val) if workers_val and workers_val.strip() else 4
        except ValueError:
            self.rank_max_workers = 4
        self.rank_max_workers = max(1, self.rank_max_workers)