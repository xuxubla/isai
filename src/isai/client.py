"""Синхронный и асинхронный клиенты для OpenAI-совместимого /chat/completions."""

from __future__ import annotations

import json
import os
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

import httpx

from .exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ConfigurationError,
    RateLimitError,
)
from .models import ChatCompletion, Message, StreamChunk

__all__ = ["LLMClient", "AsyncLLMClient"]

# Сообщения можно передавать как объекты Message или как обычные dict.
MessageInput = Union[Message, Mapping[str, Any]]

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
# Статусы, при которых имеет смысл повторить запрос.
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


def _normalize_messages(
    messages: Iterable[MessageInput],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, Message):
            result.append(m.to_dict())
        elif isinstance(m, Mapping):
            if "role" not in m or "content" not in m:
                raise ValueError(
                    "Каждое сообщение должно содержать ключи 'role' и 'content'"
                )
            result.append(dict(m))
        else:
            raise TypeError(
                f"Сообщение должно быть Message или dict, получено: {type(m)!r}"
            )
    return result


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    request_id = response.headers.get("x-request-id")
    try:
        body: Any = response.json()
        message = body.get("error", {}).get("message") if isinstance(body, dict) else None
    except (json.JSONDecodeError, ValueError):
        body = response.text
        message = None

    message = message or f"HTTP {response.status_code}: {response.reason_phrase}"
    kwargs = dict(status_code=response.status_code, body=body, request_id=request_id)

    if response.status_code in (401, 403):
        raise AuthenticationError(message, **kwargs)
    if response.status_code == 429:
        raise RateLimitError(message, **kwargs)
    raise APIError(message, **kwargs)


class _BaseClient:
    """Общая конфигурация и сборка запроса для sync/async клиентов."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ISAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Не задан API-ключ. Передайте api_key=... или установите "
                "переменную окружения ISAI_API_KEY."
            )

        base_url = (
            base_url
            or os.getenv("ISAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://api.ishosting.com/v1"
        )
        # Убираем хвостовой слеш, чтобы не получить двойной // в пути.
        self.base_url = base_url.rstrip("/")

        self.model = model or os.getenv("ISAI_MODEL")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self._default_headers = dict(default_headers or {})

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._default_headers)
        return headers

    def _build_payload(
        self,
        messages: Iterable[MessageInput],
        *,
        model: Optional[str],
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        stop: Optional[Union[str, List[str]]],
        extra: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        chosen_model = model or self.model
        if not chosen_model:
            raise ConfigurationError(
                "Не указана модель. Передайте model=... в вызов или в конструктор "
                "клиента, либо установите переменную окружения ISAI_MODEL."
            )

        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": _normalize_messages(messages),
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        # 0.5s, 1s, 2s, 4s, ... с потолком.
        return min(0.5 * (2 ** attempt), 8.0)


def _coerce_prompt_to_messages(
    prompt: str, system: Optional[str]
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _iter_sse_lines(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Парсит строки потока SSE в словари, пропуская [DONE] и служебные строки."""
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


class LLMClient(_BaseClient):
    """Синхронный клиент.

    Пример:
        client = LLMClient(api_key="...", model="gpt-4o-mini")
        answer = client.complete("Переведи на английский: привет")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(timeout=self.timeout)

    # --- управление ресурсами ---
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- низкоуровневый запрос с ретраями ---
    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    url, headers=self._headers(), json=payload
                )
            except httpx.TimeoutException as exc:
                last_exc = APIConnectionError(f"Таймаут запроса: {exc}")
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(f"Ошибка соединения: {exc}")
            else:
                if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                _raise_for_status(response)
                return response

            if attempt < self.max_retries:
                time.sleep(self._backoff_seconds(attempt))
        assert last_exc is not None
        raise last_exc

    # --- публичный API ---
    def chat(
        self,
        messages: Iterable[MessageInput],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> ChatCompletion:
        """Отправляет список сообщений и возвращает полный ответ."""
        payload = self._build_payload(
            messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            extra=extra,
        )
        response = self._post(payload)
        return ChatCompletion.from_dict(response.json())

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Удобная обёртка: один промпт -> строка ответа.

        Идеально для обработки входящих данных: передаёте данные в prompt,
        получаете готовый текст ответа модели.
        """
        completion = self.chat(
            _coerce_prompt_to_messages(prompt, system), model=model, **kwargs
        )
        return completion.content

    def stream(
        self,
        messages: Iterable[MessageInput],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        """Потоковый ответ: возвращает генератор фрагментов по мере генерации."""
        payload = self._build_payload(
            messages,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            extra=extra,
        )
        url = f"{self.base_url}/chat/completions"
        with self._client.stream(
            "POST", url, headers=self._headers(), json=payload
        ) as response:
            if response.status_code >= 400:
                response.read()
                _raise_for_status(response)
            for data in _iter_sse_lines(response.iter_lines()):
                yield StreamChunk.from_dict(data)


class AsyncLLMClient(_BaseClient):
    """Асинхронный клиент с тем же API, что и LLMClient."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncLLMClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        import asyncio

        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    url, headers=self._headers(), json=payload
                )
            except httpx.TimeoutException as exc:
                last_exc = APIConnectionError(f"Таймаут запроса: {exc}")
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(f"Ошибка соединения: {exc}")
            else:
                if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_seconds(attempt))
                    continue
                _raise_for_status(response)
                return response

            if attempt < self.max_retries:
                await asyncio.sleep(self._backoff_seconds(attempt))
        assert last_exc is not None
        raise last_exc

    async def chat(
        self,
        messages: Iterable[MessageInput],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> ChatCompletion:
        payload = self._build_payload(
            messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            extra=extra,
        )
        response = await self._post(payload)
        return ChatCompletion.from_dict(response.json())

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        completion = await self.chat(
            _coerce_prompt_to_messages(prompt, system), model=model, **kwargs
        )
        return completion.content

    async def stream(
        self,
        messages: Iterable[MessageInput],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(
            messages,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            extra=extra,
        )
        url = f"{self.base_url}/chat/completions"
        async with self._client.stream(
            "POST", url, headers=self._headers(), json=payload
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response)
            async for line in response.aiter_lines():
                for data in _iter_sse_lines([line]):
                    yield StreamChunk.from_dict(data)
