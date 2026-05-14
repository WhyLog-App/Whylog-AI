"""요청 로깅 미들웨어 단위 테스트."""

from types import SimpleNamespace

from app.core.logging_middleware import (
    _format_body_for_log,
    _should_log_request_body,
)


def _request(content_type: str, path: str = "/api/test"):
    return SimpleNamespace(
        headers={"content-type": content_type},
        url=SimpleNamespace(path=path),
    )


def test_json_body_log_masks_sensitive_values():
    body = (
        b'{"commit_id": 1, "password": "pw", "accessToken": "access-secret", "nested": '
        b'{"token": "abc", "message": "safe"}}'
    )

    formatted = _format_body_for_log(_request("application/json"), body)

    assert '"commit_id": 1' in formatted
    assert '"message": "safe"' in formatted
    assert "pw" not in formatted
    assert "abc" not in formatted
    assert "access-secret" not in formatted
    assert '"password": "***"' in formatted
    assert '"accessToken": "***"' in formatted
    assert '"token": "***"' in formatted


def test_json_body_log_omits_large_text_fields():
    body = (
        b'{"changed_file_list": [{"file_name": "config.yml", '
        b'"changed_code": "+api_key: sk-secret"}], '
        b'"utterance": "private meeting text"}'
    )

    formatted = _format_body_for_log(_request("application/json"), body)

    assert "sk-secret" not in formatted
    assert "private meeting text" not in formatted
    assert '"changed_code": "<omitted, 19 chars>"' in formatted
    assert '"utterance": "<omitted, 20 chars>"' in formatted


def test_multipart_body_log_omits_raw_payload():
    body = b"--boundary\r\nbinary-audio-bytes\r\n--boundary--"

    formatted = _format_body_for_log(
        _request("multipart/form-data; boundary=boundary"),
        body,
    )

    assert formatted == f"(multipart/form-data, {len(body)} bytes)"
    assert "binary-audio-bytes" not in formatted


def test_commit_analyze_body_logging_is_excluded():
    assert _should_log_request_body("/api/commit/analyze/runs") is False
    assert _should_log_request_body("/api/commit/analyze") is False


def test_transcribe_and_meeting_analysis_body_logging_is_excluded():
    assert _should_log_request_body("/api/transcribe/applications/runs") is False
    assert _should_log_request_body("/api/meeting-analysis/embeddings") is False


def test_non_api_body_logging_is_excluded():
    assert _should_log_request_body("/.env") is False


def test_malformed_json_body_falls_back_to_truncated_text():
    formatted = _format_body_for_log(_request("application/json"), b'{"broken":')

    assert formatted == '{"broken":'
