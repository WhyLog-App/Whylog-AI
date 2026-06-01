import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

from app.core.gemini import generate_content_with_retry, log_gemini_usage


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


def test_log_gemini_usage_records_metadata(caplog) -> None:
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=3,
            total_token_count=13,
            thoughts_token_count=2,
            cached_content_token_count=1,
        )
    )

    with caplog.at_level("INFO", logger="app.core.gemini"):
        log_gemini_usage(
            response,
            operation_name="Gemini 테스트",
            model="gemini-test",
            attempt=2,
            log_context={"meeting_id": 1},
        )

    assert "operation=Gemini 테스트" in caplog.text
    assert "model=gemini-test" in caplog.text
    assert "attempt=2" in caplog.text
    assert "prompt_tokens=10" in caplog.text
    assert "candidate_tokens=3" in caplog.text
    assert "total_tokens=13" in caplog.text
    assert "context={'meeting_id': 1}" in caplog.text
