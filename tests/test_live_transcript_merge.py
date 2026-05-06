import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppServiceError
from app.domains.transcribe.schemas import LiveTranscriptMessage, TranscribeSegment
from app.domains.transcribe.services.live_transcript_merge import (
    merge_live_transcript,
    parse_live_messages,
)
from app.main import app


def _segment(
    message_id: int,
    speaker: str,
    start_time: str,
    text: str,
) -> TranscribeSegment:
    return TranscribeSegment(
        message_id=message_id,
        speaker=speaker,
        start_time=start_time,
        end_time=start_time,
        text=text,
        is_final=True,
    )


def test_parse_live_messages_accepts_spring_camel_case_payload():
    raw = """
    [
      {
        "type": "TEXT",
        "meetingId": 123,
        "fromMemberId": 1,
        "fromName": "김준용",
        "timestamp": "00:01:48",
        "targetMemberId": 2,
        "text": "안녕하세요",
        "payload": {"key": "value"}
      }
    ]
    """

    messages = parse_live_messages(raw)

    assert messages[0].meeting_id == 123
    assert messages[0].from_member_id == 1
    assert messages[0].from_name == "김준용"
    assert messages[0].target_member_id == 2


def test_parse_live_messages_invalid_json_raises_422():
    with pytest.raises(AppServiceError) as exc_info:
        parse_live_messages("not-json")

    assert exc_info.value.status_code == 422


def test_merge_live_transcript_replaces_speaker_and_text_from_live_messages():
    segments = [
        _segment(
            1,
            "Speaker 0",
            "00:01:48",
            "보고를 할 때는 지난번에 우리가 어떤 이슈가 있었고 그거를 해결했는데",
        ),
        _segment(
            2,
            "Speaker 0",
            "00:02:10",
            "네 번은 문제가 있던 거를 표로 보여줘야 됩니다",
        ),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="00:01:48",
            text=(
                "보고를 할 때는 지난번에 우리가 어떤 이슈가 있었고 "
                "그걸 어떻게 해결했는지 보여줘야 해"
            ),
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="00:02:10",
            text="네 번은 문제가 있던 거를 표로 보여줘야 됩니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert merged[0].speaker == "김준용"
    assert merged[0].member_id == 1
    assert merged[0].text == (
        "보고를 할 때는 지난번에 우리가 어떤 이슈가 있었고 "
        "그걸 어떻게 해결했는지 보여줘야 해"
    )
    assert merged[1].speaker == "김준용"
    assert merged[1].member_id == 1


def test_merge_live_transcript_keeps_stt_when_only_ambiguous_short_messages_exist():
    segments = [
        _segment(1, "Speaker 0", "00:00:01", "네"),
        _segment(2, "Speaker 0", "00:00:03", "네"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="00:00:01",
            text="네",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="00:00:03",
            text="네",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert merged[0].speaker == "Speaker 0"
    assert merged[0].member_id is None
    assert merged[1].speaker == "Speaker 0"
    assert merged[1].member_id is None


def test_merge_live_transcript_uses_text_and_order_when_timestamp_is_iso_string():
    segments = [
        _segment(1, "Speaker 0", "00:00:01", "안녀 하세요"),
        _segment(2, "Speaker 0", "00:00:03", "회의 시작하겠습니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="2026-05-04T17:00:00",
            text="안녕하세요",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=123,
            fromMemberId=1,
            fromName="김준용",
            timestamp="2026-05-04T17:00:02",
            text="회의 시작하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert merged[0].speaker == "김준용"
    assert merged[0].member_id == 1
    assert merged[1].speaker == "김준용"


class TestLiveMessagesEndpoint:
    client = TestClient(app)

    def test_transcribe_applications_rejects_invalid_live_messages_json(self):
        response = self.client.post(
            "/api/transcribe/applications",
            files={"audio": ("sample.wav", b"fake-audio", "audio/wav")},
            data={"live_messages": "not-json"},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"
