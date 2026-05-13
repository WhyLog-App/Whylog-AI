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


async def generate_content_with_retry(
    client: genai.Client,
    *,
    contents: Any,
    config: types.GenerateContentConfig,
    timeout: float,
    operation_name: str,
    backoffs: Sequence[float] = DEFAULT_RETRY_BACKOFFS,
) -> Any:
    """Gemini generate_content 호출에 timeout/retry 정책을 공통 적용한다."""
    last_error: Exception | None = None

    for attempt in range(len(backoffs) + 1):
        try:
            return await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=settings.gemini_llm_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout,
            )
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
