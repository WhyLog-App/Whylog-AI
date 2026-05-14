"""공통 도메인 enum 계약 테스트."""

from app.core.enums import (
    CommitAnalyzeRunPhase,
    CommitChangeDirection,
    RunStatus,
    TranscribeRunPhase,
)
from app.main import app


def test_openapi_exposes_run_status_as_named_string_enum():
    """run 상태 enum 값은 API 계약 문자열과 일치한다."""
    schemas = app.openapi()["components"]["schemas"]

    assert schemas["RunStatus"] == {
        "type": "string",
        "enum": ["queued", "processing", "completed", "failed"],
        "title": "RunStatus",
    }
    assert schemas["TranscribeRunPhase"] == {
        "type": "string",
        "enum": [
            "queued",
            "transcribing",
            "transcript_ready",
            "summary_ready",
            "applications_ready",
            "failed",
        ],
        "title": "TranscribeRunPhase",
    }
    assert schemas["CommitAnalyzeRunPhase"] == {
        "type": "string",
        "enum": [
            "queued",
            "summarizing",
            "summary_ready",
            "embedding",
            "embedding_ready",
            "failed",
        ],
        "title": "CommitAnalyzeRunPhase",
    }


def test_enum_values_match_existing_api_contracts():
    """공통 enum 값은 기존 외부 계약 문자열을 유지한다."""
    assert [status.value for status in RunStatus] == [
        "queued",
        "processing",
        "completed",
        "failed",
    ]
    assert [phase.value for phase in TranscribeRunPhase] == [
        "queued",
        "transcribing",
        "transcript_ready",
        "summary_ready",
        "applications_ready",
        "failed",
    ]
    assert [phase.value for phase in CommitAnalyzeRunPhase] == [
        "queued",
        "summarizing",
        "summary_ready",
        "embedding",
        "embedding_ready",
        "failed",
    ]
    assert {direction.value for direction in CommitChangeDirection} == {
        "add",
        "remove",
        "modify",
        "migrate",
    }
