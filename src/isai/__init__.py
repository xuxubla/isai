"""isai — лёгкий клиент для OpenAI-совместимого LLM API (/chat/completions).

Быстрый старт:

    from isai import LLMClient

    client = LLMClient(api_key="...", model="gpt-4o-mini")
    answer = client.complete("Извлеки email из текста: ...")
    print(answer)
"""

from .client import AsyncLLMClient, LLMClient
from .exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ConfigurationError,
    LLMError,
    RateLimitError,
)
from .images import build_user_content, image_part
from .models import ChatCompletion, Choice, Message, StreamChunk, Usage

__version__ = "0.1.0"

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "Message",
    "ChatCompletion",
    "Choice",
    "Usage",
    "StreamChunk",
    "image_part",
    "build_user_content",
    "LLMError",
    "ConfigurationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "APIConnectionError",
    "__version__",
]
