import json
import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")

_MAX_BODY_SIZE = 10_000  # 10KB 초과 시 truncate
_MAX_STRING_LOG_CHARS = 500
_MAX_LIST_LOG_ITEMS = 5
_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_VERBOSE_KEY_PARTS = {
    "audio",
    "changed_code",
    "content",
    "live_messages",
    "text",
    "utterance",
}
_PII_KEYS = {
    "from_name",
    "fromname",
    "speaker",
    "speaker_name",
    "speakername",
}


def _key_contains(key: str, candidates: set[str]) -> bool:
    normalized = key.lower()
    return any(candidate in normalized for candidate in candidates)


def _summarize_string(value: str) -> str:
    return f"<{len(value)} chars>"


def _sanitize_json(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = key.lower()
            if _key_contains(normalized_key, _SENSITIVE_KEY_PARTS):
                sanitized[key] = "<masked>"
            elif normalized_key in _PII_KEYS:
                sanitized[key] = "<masked>"
            elif _key_contains(normalized_key, _VERBOSE_KEY_PARTS):
                sanitized[key] = (
                    _summarize_string(child)
                    if isinstance(child, str)
                    else _sanitize_json(child, key)
                )
            else:
                sanitized[key] = _sanitize_json(child, key)
        return sanitized

    if isinstance(value, list):
        items = [
            _sanitize_json(item, parent_key) for item in value[:_MAX_LIST_LOG_ITEMS]
        ]
        if len(value) > _MAX_LIST_LOG_ITEMS:
            items.append(f"<{len(value) - _MAX_LIST_LOG_ITEMS} more items>")
        return items

    if isinstance(value, str) and len(value) > _MAX_STRING_LOG_CHARS:
        return _summarize_string(value)

    return value


def _truncate(value: str) -> str:
    if len(value) <= _MAX_BODY_SIZE:
        return value
    return value[:_MAX_BODY_SIZE] + f"... (truncated, total {len(value)} chars)"


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_json_media_type(content_type: str) -> bool:
    media_type = _media_type(content_type)
    return media_type == "application/json" or media_type.endswith("+json")


async def _format_body_for_log(request: Request) -> str:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return "(empty)"

    content_type = request.headers.get("content-type", "")
    content_length = request.headers.get("content-length", "unknown")
    if content_type.startswith("multipart/form-data"):
        return f"(multipart/form-data omitted, content_length={content_length})"

    if not _is_json_media_type(content_type):
        media_type = _media_type(content_type) or "non-json"
        return f"({media_type} body omitted, content_length={content_length})"

    body_bytes = await request.body()
    if not body_bytes:
        return "(empty)"

    try:
        parsed = json.loads(body_bytes)
        sanitized = _sanitize_json(parsed)
        return _truncate(
            json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        )
    except json.JSONDecodeError:
        return f"(malformed json, content_length={len(body_bytes)})"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        body_text = await _format_body_for_log(request)

        logger.info(
            ">>> %s %s\nBody: %s",
            request.method,
            request.url.path,
            body_text,
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
