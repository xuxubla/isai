"""Тесты клиента с замоканным HTTP (respx)."""

import httpx
import pytest
import respx

from isai import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    LLMClient,
    Message,
    RateLimitError,
)

BASE_URL = "https://api.example.test/v1"
URL = f"{BASE_URL}/chat/completions"


def _completion_body(text="Привет!"):
    return {
        "id": "cmpl-1",
        "model": "test-model",
        "created": 1,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def make_client(**kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "test-model")
    return LLMClient(**kwargs)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("ISAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        LLMClient(base_url=BASE_URL, model="m")


def test_requires_model(monkeypatch):
    monkeypatch.delenv("ISAI_MODEL", raising=False)
    client = LLMClient(api_key="k", base_url=BASE_URL)
    with pytest.raises(ConfigurationError):
        client.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_complete_returns_text():
    respx.post(URL).mock(return_value=httpx.Response(200, json=_completion_body("ответ")))
    with make_client() as client:
        assert client.complete("вопрос") == "ответ"


@respx.mock
def test_chat_builds_payload():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_completion_body()))
    with make_client() as client:
        client.chat(
            [Message("system", "ты бот"), {"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=64,
        )
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-key"
    import json

    payload = json.loads(sent.content)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64
    assert payload["messages"][0] == {"role": "system", "content": "ты бот"}


@respx.mock
def test_auth_error():
    respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "bad key"}}))
    with make_client() as client:
        with pytest.raises(AuthenticationError):
            client.complete("hi")


@respx.mock
def test_rate_limit_retries_then_succeeds():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=_completion_body("ок")),
        ]
    )
    with make_client(max_retries=1) as client:
        assert client.complete("hi") == "ок"
    assert route.call_count == 2


@respx.mock
def test_rate_limit_exhausted():
    respx.post(URL).mock(return_value=httpx.Response(429, json={"error": {"message": "no"}}))
    with make_client(max_retries=1) as client:
        with pytest.raises(RateLimitError):
            client.complete("hi")


@respx.mock
def test_stream_yields_chunks():
    sse = (
        'data: {"choices":[{"delta":{"content":"При"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"вет"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(URL).mock(return_value=httpx.Response(200, text=sse))
    with make_client() as client:
        out = "".join(chunk.delta for chunk in client.stream([{"role": "user", "content": "hi"}]))
    assert out == "Привет"


@respx.mock
def test_complete_with_images_builds_multimodal_content(tmp_path):
    import base64
    import json

    # Минимальный валидный PNG-заголовок -> детектится как image/png.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    img_path = tmp_path / "pic.png"
    img_path.write_bytes(png)

    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=_completion_body("на картинке png"))
    )
    with make_client() as client:
        out = client.complete_with_images(
            "Что на картинке?",
            images=[
                "https://example.com/a.jpg",  # URL как есть
                str(img_path),                # локальный файл -> base64
                png,                          # сырые байты -> base64
            ],
            detail="low",
        )
    assert out == "на картинке png"

    payload = json.loads(route.calls.last.request.content)
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Что на картинке?"}
    assert content[1]["image_url"]["url"] == "https://example.com/a.jpg"
    assert content[1]["image_url"]["detail"] == "low"
    # файл и байты закодированы в data-URI с правильным MIME
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    b64 = base64.b64encode(png).decode()
    assert content[3]["image_url"]["url"] == f"data:image/png;base64,{b64}"


def test_image_part_rejects_bad_type():
    from isai import image_part

    with pytest.raises(TypeError):
        image_part(123)


@respx.mock
@pytest.mark.asyncio
async def test_async_complete():
    from isai import AsyncLLMClient

    respx.post(URL).mock(return_value=httpx.Response(200, json=_completion_body("async-ok")))
    async with AsyncLLMClient(api_key="k", base_url=BASE_URL, model="m") as client:
        assert await client.complete("hi") == "async-ok"
