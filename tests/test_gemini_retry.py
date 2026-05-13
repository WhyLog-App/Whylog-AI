import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

from app.core.gemini import generate_content_with_retry


class _SlowThenSuccessModels:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_content(self, **kwargs):
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.05)
        return SimpleNamespace(text="ok")


@pytest.mark.asyncio
async def test_generate_content_with_retry_retries_timeout() -> None:
    models = _SlowThenSuccessModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))

    response = await generate_content_with_retry(
        client,
        contents="prompt",
        config=types.GenerateContentConfig(temperature=0.1),
        timeout=0.01,
        operation_name="Gemini 테스트",
        backoffs=(0.0,),
    )

    assert response.text == "ok"
    assert models.calls == 2
