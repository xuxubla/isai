# isai

Лёгкий Python-клиент для OpenAI-совместимого LLM API (эндпоинт `POST /chat/completions`).
Предназначен для подключения как пакет в другие проекты: передаёте входящие данные —
получаете ответ модели.

Работает с любым OpenAI-совместимым сервером (включая `ishosting`): базовый URL и
API-ключ настраиваются.

## Возможности

- Синхронный (`LLMClient`) и асинхронный (`AsyncLLMClient`) клиенты с одинаковым API.
- Удобный метод `complete()` — «строка на входе, строка на выходе».
- Полный метод `chat()` с историей сообщений, моделями ответа и статистикой токенов.
- Потоковая генерация `stream()` (SSE).
- Автоматические повторы с экспоненциальной задержкой на `429`/`5xx`.
- Типизированные исключения (`AuthenticationError`, `RateLimitError`, `APIError`, ...).
- Конфигурация через аргументы или переменные окружения.
- Минимум зависимостей — только `httpx`.

## Установка

Из исходников репозитория:

```bash
pip install .
```

Или подключить как зависимость из git:

```bash
pip install "git+https://github.com/xuxubla/isai.git"
```

Требуется Python 3.9+.

## Конфигурация

Параметры можно передать в конструктор или задать переменными окружения:

| Переменная        | Аргумент      | Назначение                                  |
|-------------------|---------------|---------------------------------------------|
| `ISAI_API_KEY`    | `api_key`     | API-ключ (обязательно)                      |
| `ISAI_BASE_URL`   | `base_url`    | Базовый URL API (по умолчанию `https://ai.ishosting.com/api`) |
| `ISAI_MODEL`      | `model`       | Модель по умолчанию                         |

```bash
export ISAI_API_KEY="ваш-ключ"
export ISAI_BASE_URL="https://ai.ishosting.com/api"
export ISAI_MODEL="название-модели"
```

## Быстрый старт

```python
from isai import LLMClient

client = LLMClient(api_key="...", base_url="https://ai.ishosting.com/api", model="...")

# Обработать входящие данные и получить ответ:
answer = client.complete(
    prompt="Извлеки email: Иван, ivan@example.com",
    system="Ты экстрактор данных. Отвечай кратко.",
    temperature=0,
)
print(answer)  # -> ivan@example.com
```

### Полный диалог

```python
from isai import LLMClient, Message

with LLMClient() as client:
    completion = client.chat(
        [
            Message("system", "Ты помощник, отвечающий по-русски."),
            Message("user", "Назови три столицы Европы."),
        ],
        temperature=0.7,
        max_tokens=200,
    )
    print(completion.content)             # текст ответа
    print(completion.usage.total_tokens)  # израсходовано токенов
```

### Изображения на вход (vision)

Для мультимодальных моделей картинку можно передать как URL, data-URI,
путь к локальному файлу или `bytes` — кодирование в base64 произойдёт само:

```python
with LLMClient(model="<vision-модель>") as client:
    answer = client.complete_with_images(
        "Что изображено на картинках?",
        images=[
            "https://example.com/photo.jpg",  # URL
            "/path/to/local.png",             # локальный файл
            open("scan.jpg", "rb").read(),    # сырые байты
        ],
        detail="high",   # необязательно: low / high / auto
    )
    print(answer)
```

Под капотом собирается `content` с частями `text` + `image_url`. Нужен
ручной контроль — есть низкоуровневые `image_part()` и `build_user_content()`.

### Потоковый вывод

```python
with LLMClient() as client:
    for chunk in client.stream([{"role": "user", "content": "Считай от 1 до 5"}]):
        print(chunk.delta, end="", flush=True)
```

### Асинхронно

```python
import asyncio
from isai import AsyncLLMClient

async def main():
    async with AsyncLLMClient() as client:
        print(await client.complete("Скажи привет"))

asyncio.run(main())
```

## Обработка ошибок

```python
from isai import LLMClient, AuthenticationError, RateLimitError, APIError

try:
    text = LLMClient().complete("привет")
except AuthenticationError:
    ...  # неверный ключ
except RateLimitError:
    ...  # превышен лимит запросов
except APIError as e:
    print(e.status_code, e.body)
```

## Параметры вызова

`chat()` / `stream()` принимают: `model`, `temperature`, `max_tokens`, `top_p`,
`stop`, а также `extra` — словарь любых дополнительных полей, которые
будут добавлены прямо в тело запроса (например, `response_format`, `seed`).

## Разработка

```bash
pip install -e ".[dev]"
pytest
```

## Лицензия

MIT
