from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.core.settings import LLMProvider, ModelSettings

import os

def _force_load_env():
    """Đọc file .env và ghi đè os.environ (bao gồm cả Machine-level system env vars)."""
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k.strip()] = v.strip()


def _build_chat_model(settings: ModelSettings) -> BaseModel:
    _force_load_env()  # Luôn override system env vars bằng giá trị trong .env

    if settings.llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_API_URL")

        # Claude 5+ (claude-sonnet-5, claude-opus-5) không hỗ trợ temperature
        model_name = settings.llm_model_name or ""
        is_claude5 = any(x in model_name for x in ["claude-sonnet-5", "claude-opus-5", "claude-fable-5"])

        kwargs = {
            "model": model_name,
            "max_tokens": 8000,
        }
        if not is_claude5:
            kwargs["temperature"] = settings.llm_temperature

        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["anthropic_api_url"] = base_url

        return ChatAnthropic(**kwargs)
    elif settings.llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        kwargs = {
            "model": settings.llm_model_name,
            "temperature": settings.llm_temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)
    elif settings.llm_provider == LLMProvider.OPEN_SOURCE:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.llm_model_name, temperature=settings.llm_temperature)
    else:
        raise NotImplementedError(f"Provider '{settings.llm_provider}' chưa được hỗ trợ.")


class BaseLLMClient(ABC):
    @abstractmethod
    def extract_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Truyền thẳng Pydantic schema (ví dụ ParsedCV) thay vì mô tả
        schema bằng chuỗi. LangChain tự sinh JSON Schema, ép model tuân
        theo, và validate luôn bằng Pydantic."""
        raise NotImplementedError
 
 
class LangChainLLMClient(BaseLLMClient):
    """Một class DUY NHẤT dùng chung cho mọi provider, vì LangChain đã
    chuẩn hoá interface gọi model."""
 
    def __init__(self, settings: ModelSettings):
        self.chat_model = _build_chat_model(settings)
 
    def extract_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        structured_llm = self.chat_model.with_structured_output(schema)
        return structured_llm.invoke(prompt)
 
 
def get_llm_client(settings: ModelSettings) -> BaseLLMClient:
    """Các module khác (parsing_service, scoring_service) chỉ cần gọi
    hàm này, không cần biết chi tiết provider hay LangChain ChatModel
    nào đang chạy bên dưới."""
    return LangChainLLMClient(settings)
 