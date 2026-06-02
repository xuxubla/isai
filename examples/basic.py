"""Примеры использования пакета isai.

Перед запуском:
    export ISAI_API_KEY="ваш-ключ"
    export ISAI_BASE_URL="https://ai.ishosting.com/api"   # при необходимости
    export ISAI_MODEL="название-модели"
    python examples/basic.py
"""

import asyncio

from isai import AsyncLLMClient, LLMClient, Message


def example_simple():
    """Самый частый сценарий: обработать входные данные -> получить ответ."""
    with LLMClient() as client:
        data = "Иван Петров, тел. +7 999 123-45-67, ivan@example.com"
        answer = client.complete(
            prompt=f"Извлеки email из строки и верни только его:\n{data}",
            system="Ты аккуратный экстрактор данных. Отвечай кратко.",
            temperature=0,
        )
        print("Ответ:", answer)


def example_dialog():
    """Полный диалог с историей сообщений и статистикой токенов."""
    with LLMClient() as client:
        completion = client.chat(
            [
                Message("system", "Ты помощник, отвечающий по-русски."),
                Message("user", "Назови три столицы Европы."),
            ],
            temperature=0.7,
            max_tokens=200,
        )
        print("Текст:", completion.content)
        print("Токены:", completion.usage.total_tokens)


def example_stream():
    """Потоковый вывод по мере генерации."""
    with LLMClient() as client:
        for chunk in client.stream([{"role": "user", "content": "Считай от 1 до 5"}]):
            print(chunk.delta, end="", flush=True)
        print()


async def example_async():
    """Асинхронный клиент — тот же интерфейс."""
    async with AsyncLLMClient() as client:
        answer = await client.complete("Скажи 'привет' одним словом")
        print("Async:", answer)


if __name__ == "__main__":
    example_simple()
    example_dialog()
    example_stream()
    asyncio.run(example_async())
