"""Сборка мультимодального контента (текст + изображения) для /chat/completions.

Изображение можно задать как:
- URL (``http://`` / ``https://``) — передаётся как есть;
- data-URI (``data:image/...;base64,...``) — передаётся как есть;
- путь к локальному файлу (str или pathlib.Path) — читается и кодируется в base64;
- сырые байты (bytes/bytearray) — кодируются в base64.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

__all__ = ["ImageSource", "image_part", "build_user_content"]

# Допустимые типы источника изображения.
ImageSource = Union[str, "os.PathLike[str]", bytes, bytearray]

# Сигнатуры файлов -> MIME, чтобы определить тип для «сырых» байтов.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP
)


def _sniff_mime(data: bytes) -> str:
    for sig, mime in _MAGIC:
        if data.startswith(sig):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return "application/octet-stream"


def _bytes_to_data_uri(data: bytes, mime: Optional[str] = None) -> str:
    mime = mime or _sniff_mime(data)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _path_to_data_uri(path: Union[str, "os.PathLike[str]"]) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл изображения не найден: {p}")
    mime = mimetypes.guess_type(p.name)[0]
    return _bytes_to_data_uri(p.read_bytes(), mime)


def image_part(
    source: ImageSource, *, detail: Optional[str] = None
) -> Dict[str, Any]:
    """Возвращает одну часть контента типа ``image_url`` для сообщения.

    Args:
        source: URL, data-URI, путь к файлу или байты изображения.
        detail: необязательный уровень детализации (``"low"``/``"high"``/``"auto"``),
            если модель его поддерживает.
    """
    if isinstance(source, (bytes, bytearray)):
        url = _bytes_to_data_uri(bytes(source))
    elif isinstance(source, os.PathLike):
        url = _path_to_data_uri(source)
    elif isinstance(source, str):
        if source.startswith(("http://", "https://", "data:")):
            url = source
        else:
            # Трактуем строку как путь к локальному файлу.
            url = _path_to_data_uri(source)
    else:
        raise TypeError(
            f"Неподдерживаемый тип изображения: {type(source)!r}. "
            "Используйте URL, data-URI, путь к файлу или bytes."
        )

    image_url: Dict[str, Any] = {"url": url}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def build_user_content(
    text: str,
    images: Iterable[ImageSource],
    *,
    detail: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Собирает ``content`` для user-сообщения: текст + список изображений."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        parts.append(image_part(img, detail=detail))
    return parts
