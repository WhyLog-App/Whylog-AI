import json
import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")

_MAX_BODY_SIZE = 10_000  # 10KB 초과 시 truncate
_BODY_LOG_EXCLUDED_PREFIXES = (
    "/api/commit/analyze",
    "/api/meeting-analysis",
    "/api/transcribe",
)
_LARGE_TEXT_KEYS = {
    "changed_code",
    "content",
    "text",
    "utterance",
}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "authorization",
    "client_secret",
    "cookie",
    "gemini_api_key",
    "key",
    "password",
    "refresh_token",
    "secret",
    "secret_token",
    "token",
}


def _mask_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            key_lower = key.lower()
            if key_lower in _LARGE_TEXT_KEYS and isinstance(item, str):
                masked[key] = f"<omitted, {len(item)} chars>"
            elif any(sensitive in key_lower for sensitive in _SENSITIVE_KEYS):
                masked[key] = "***"
            else:
                masked[key] = _mask_sensitive(item)
        return masked
    if isinstance(value, list):
        return [_mask_sensitive(item) for item in value]
    return value


def _truncate_body(body_text: str) -> str:
    if len(body_text) <= _MAX_BODY_SIZE:
        return body_text
    return body_text[:_MAX_BODY_SIZE] + f"... (truncated, total {len(body_text)} chars)"


def _format_body_for_log(request: Request, body_bytes: bytes) -> str:
    if not body_bytes:
        return "(empty)"

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        return f"(multipart/form-data, {len(body_bytes)} bytes)"

    body_text = body_bytes.decode("utf-8", errors="replace")
    if "application/json" not in content_type:
        return _truncate_body(body_text)

    try:
        masked = _mask_sensitive(json.loads(body_text))
    except json.JSONDecodeError:
        return _truncate_body(body_text)
    return _truncate_body(json.dumps(masked, ensure_ascii=False))


def _should_log_access(path: str) -> bool:
    return path.startswith("/api")


def _should_log_request_body(path: str) -> bool:
    if not _should_log_access(path):
        return False
    return not path.startswith(_BODY_LOG_EXCLUDED_PREFIXES)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        path = request.url.path
        should_log_access = _should_log_access(path)
        should_log_body = _should_log_request_body(path)

        if should_log_access:
            if should_log_body:
                body_summary = _format_body_for_log(request, await request.body())
            else:
                body_summary = "(omitted for privacy)"
            logger.info(
                ">>> %s %s\nBody: %s",
                request.method,
                path,
                body_summary,
            )

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if should_log_access:
            logger.info(
                "<<< %s %s %d (%dms)",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
            )

        return response
