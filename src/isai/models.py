"""Типизированные модели запроса и ответа /chat/completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Сообщение в формате чата. content допускает строку либо список частей
# (для мультимодальных моделей), поэтому тип намеренно широкий.
Content = Union[str, List[Dict[str, Any]]]


@dataclass
class Message:
    """Одно сообщение диалога."""

    role: str
    content: Content
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            name=data.get("name"),
        )


@dataclass
class Usage:
    """Статистика по токенам."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Usage":
        data = data or {}
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class Choice:
    """Один вариант ответа модели."""

    index: int
    message: Message
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Choice":
        return cls(
            index=data.get("index", 0),
            message=Message.from_dict(data.get("message", {})),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class ChatCompletion:
    """Ответ эндпоинта /chat/completions."""

    id: Optional[str]
    model: Optional[str]
    choices: List[Choice]
    usage: Usage
    created: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Текст первого варианта ответа — самый частый случай использования."""
        if not self.choices:
            return ""
        content = self.choices[0].message.content
        return content if isinstance(content, str) else str(content)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletion":
        return cls(
            id=data.get("id"),
            model=data.get("model"),
            created=data.get("created"),
            choices=[Choice.from_dict(c) for c in data.get("choices", [])],
            usage=Usage.from_dict(data.get("usage")),
            raw=data,
        )


@dataclass
class StreamChunk:
    """Фрагмент потокового ответа (server-sent events)."""

    delta: str
    finish_reason: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamChunk":
        choices = data.get("choices") or [{}]
        first = choices[0]
        delta = (first.get("delta") or {}).get("content") or ""
        return cls(
            delta=delta,
            finish_reason=first.get("finish_reason"),
            raw=data,
        )
