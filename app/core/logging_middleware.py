import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")

_MAX_BODY_SIZE = 10_000  # 10KB 초과 시 truncate


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="replace")
        if len(body_text) > _MAX_BODY_SIZE:
            body_text = (
                body_text[:_MAX_BODY_SIZE]
                + f"... (truncated, total {len(body_text)} chars)"
            )

        logger.info(
            ">>> %s %s\nBody: %s",
            request.method,
            request.url.path,
            body_text or "(empty)",
        )

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "<<< %s %s %d (%dms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response
