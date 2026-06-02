"""Исключения клиента isai."""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Базовое исключение для всех ошибок клиента."""


class ConfigurationError(LLMError):
    """Некорректная конфигурация клиента (например, не задан API-ключ)."""


class APIError(LLMError):
    """Ошибка, возвращённая API.

    Attributes:
        status_code: HTTP-статус ответа (если есть).
        body: тело ответа сервера (распарсенное или сырой текст).
        request_id: идентификатор запроса из заголовков, если сервер его прислал.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: object = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


class AuthenticationError(APIError):
    """Неверный или отсутствующий API-ключ (HTTP 401/403)."""


class RateLimitError(APIError):
    """Превышен лимит запросов (HTTP 429)."""


class APIConnectionError(LLMError):
    """Не удалось соединиться с сервером (сетевая ошибка/таймаут)."""
