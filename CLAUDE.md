# CLAUDE.md — isai

Лёгкий Python-клиент для OpenAI-совместимого LLM API (`POST /chat/completions`).
**Это пакет-библиотека**, подключаемая в другие проекты, а не сервис: деплоя на VPS
нет. Общая инфра — в корневом `../CLAUDE.md`. Описание публичного API — в
[`README.md`](README.md).

Репозиторий самостоятельный: `git@github.com:xuxubla/isai.git`. Коммиты и push
делать из папки `isai`, не из корневого `pet`.

## Карта кода
| Путь | Назначение |
|------|-----------|
| `src/isai/client.py` | `LLMClient` (sync) и `AsyncLLMClient` (async), одинаковый API: `complete()`, `chat()`, `stream()`, `complete_with_images()` |
| `src/isai/models.py` | Pydantic-модели запроса/ответа, статистика токенов |
| `src/isai/exceptions.py` | Типизированные ошибки (`AuthenticationError`, `RateLimitError`, `APIError`, ...) |
| `src/isai/images.py` | Сборка multimodal content: URL/data-URI/локальные файлы/bytes → `image_url` parts |
| `examples/basic.py` | Пример использования |
| `tests/test_client.py` | Тесты (pytest + respx, мокают HTTP) |

## Конвенции
- Зависимость только `httpx` — **не добавлять новые runtime-зависимости** без причины.
- Python 3.9+. Конфиг через аргументы или env: `ISAI_API_KEY`, `ISAI_BASE_URL`
  (дефолт `https://ai.ishosting.com/api`), `ISAI_MODEL`. Для совместимости также
  читаются `LLM_API_KEY` и `LLM_BASE_URL`.
- Ретраи с экспоненциальной задержкой на 429/5xx — уже встроены.
- Тесты: `pip install -e ".[dev]"` затем `pytest` (asyncio_mode=auto, см. pyproject).
- Версионирование в `pyproject.toml` — бампать при изменении публичного API
  (пакет ставят как `git+https://github.com/xuxubla/isai.git` в других репо).

## Публичный API
- `complete(prompt, system=None, ...)` — простой текстовый ответ.
- `chat(messages, ...)` — полный `/chat/completions` с типизированным ответом.
- `stream(messages, ...)` — SSE-поток `StreamChunk`.
- `complete_with_images(prompt, images=[...], detail=None, ...)` — vision-ввод для
  мультимодальных моделей.
- `image_part()` и `build_user_content()` экспортируются из пакета для ручной сборки
  multimodal-сообщений.

## Проверки
- Минимальная проверка документационных/служебных правок: `pytest`.
- Не нужны Docker, VPS и deploy-команды: это библиотека.
