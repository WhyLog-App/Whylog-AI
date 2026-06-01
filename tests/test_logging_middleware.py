from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.logging_middleware import RequestLoggingMiddleware, _sanitize_json


def test_sanitize_json_masks_sensitive_and_verbose_fields() -> None:
    payload = {
        "apiKey": "secret-key",
        "nested": {
            "accessToken": "token-value",
            "fromName": "유상완",
            "changed_code": "+GEMINI_API_KEY=secret",
        },
        "live_messages": [
            {"text": "회의 원문입니다", "from_name": "유진"},
            {"text": "두 번째 발화입니다", "from_name": "김준용"},
            {"text": "세 번째 발화입니다", "from_name": "조윤지"},
            {"text": "네 번째 발화입니다", "from_name": "meme"},
            {"text": "다섯 번째 발화입니다", "from_name": "상완"},
            {"text": "여섯 번째 발화입니다", "from_name": "extra"},
        ],
    }

    sanitized = _sanitize_json(payload)

    assert sanitized["apiKey"] == "<masked>"
    assert sanitized["nested"]["accessToken"] == "<masked>"
    assert sanitized["nested"]["fromName"] == "<masked>"
    assert sanitized["nested"]["changed_code"] == "<22 chars>"
    assert len(sanitized["live_messages"]) == 6
    assert sanitized["live_messages"][-1] == "<1 more items>"
    assert sanitized["live_messages"][0]["text"] == "<8 chars>"
    assert sanitized["live_messages"][0]["from_name"] == "<masked>"


def test_logging_middleware_sanitizes_body_without_consuming_request(
    caplog,
) -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        body = await request.json()
        return {"received": body["message"]}

    client = TestClient(app)

    with caplog.at_level("INFO", logger="api.access"):
        response = client.post(
            "/echo",
            json={
                "message": "ok",
                "password": "should-not-log",
                "changed_code": "+secret=abc",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"received": "ok"}
    log_text = caplog.text
    assert "should-not-log" not in log_text
    assert "+secret=abc" not in log_text
    assert '"password":"<masked>"' in log_text
    assert '"changed_code":"<11 chars>"' in log_text


def test_logging_middleware_omits_multipart_body(caplog) -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/upload")
    async def upload() -> dict:
        return {"ok": True}

    client = TestClient(app)

    with caplog.at_level("INFO", logger="api.access"):
        response = client.post(
            "/upload",
            files={"audio": ("meeting.ogg", b"raw-audio-bytes", "audio/ogg")},
        )

    assert response.status_code == 200
    assert "raw-audio-bytes" not in caplog.text
    assert "multipart/form-data omitted" in caplog.text
