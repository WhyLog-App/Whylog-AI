import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_RETRY_BACKOFFS = (1.0, 2.0, 4.0)


def _usage_value(usage: Any, name: str) -> Any:
    return getattr(usage, name, None)


def log_gemini_usage(
    response: Any,
    *,
    operation_name: str,
    model: str,
    attempt: int | None = None,
    log_context: dict[str, Any] | None = None,
) -> None:
    """Gemini 응답 usage_metadata를 원문 없이 구조화 로그로 남긴다."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return

    logger.info(
        "Gemini usage: operation=%s model=%s attempt=%s "
        "prompt_tokens=%s candidate_tokens=%s total_tokens=%s "
        "thoughts_tokens=%s cached_tokens=%s context=%s",
        operation_name,
        model,
        attempt,
        _usage_value(usage, "prompt_token_count"),
        _usage_value(usage, "candidates_token_count"),
        _usage_value(usage, "total_token_count"),
        _usage_value(usage, "thoughts_token_count"),
        _usage_value(usage, "cached_content_token_count"),
        log_context or {},
    )


async def generate_content_with_retry(
    client: genai.Client,
    *,
    contents: Any,
    config: types.GenerateContentConfig,
    timeout: float,
    operation_name: str,
    log_context: dict[str, Any] | None = None,
    backoffs: Sequence[float] = DEFAULT_RETRY_BACKOFFS,
) -> Any:
    """Gemini generate_content 호출에 timeout/retry 정책을 공통 적용한다."""
    last_error: Exception | None = None

    for attempt in range(len(backoffs) + 1):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=settings.gemini_llm_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout,
            )
            log_gemini_usage(
                response,
                operation_name=operation_name,
                model=settings.gemini_llm_model,
                attempt=attempt + 1,
                log_context=log_context,
            )
            return response
        except TimeoutError as e:
            last_error = e
            if attempt >= len(backoffs):
                raise
            backoff = backoffs[attempt]
            logger.warning(
                "%s 타임아웃, %.1f초 후 재시도 (%d/%d)",
                operation_name,
                backoff,
                attempt + 1,
                len(backoffs),
            )
            await asyncio.sleep(backoff)
        except genai_errors.APIError as e:
            last_error = e
            if e.code not in RETRY_STATUS_CODES or attempt >= len(backoffs):
                raise
            backoff = backoffs[attempt]
            logger.warning(
                "%s %s 응답, %.1f초 후 재시도 (%d/%d)",
                operation_name,
                e.code,
                backoff,
                attempt + 1,
                len(backoffs),
            )
            await asyncio.sleep(backoff)

    raise last_error  # type: ignore[misc]
