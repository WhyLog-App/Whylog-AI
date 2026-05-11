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


def test_merge_live_transcript_maps_single_speaker_to_dominant_live_member():
    segments = [
        _segment(1, "Speaker 0", "00:00:00", "헤이 시작 오늘은 무슨 게임을 시작할까요"),
        _segment(2, "Speaker 0", "00:00:06", "게임하지 말고 잡시다"),
        _segment(3, "Speaker 0", "00:00:13", "바이바이"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=9,
            fromName="김준용",
            timestamp="00:00:00",
            text="회의 시작 오늘은 무슨 얘기를 할까요",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=9,
            fromName="김준용",
            timestamp="00:00:06",
            text="게임하지 말고 잡시다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=9,
            fromName="김준용",
            timestamp="00:00:13",
            text="바이바이",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [9, 9, 9]
    assert {segment.speaker for segment in merged} == {"김준용"}


def test_merge_live_transcript_does_not_use_dominant_fallback_for_mixed_members():
    segments = [
        _segment(1, "Speaker 0", "00:00:00", "첫 번째 안건입니다"),
        _segment(2, "Speaker 0", "00:00:06", "두 번째 안건입니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=9,
            fromName="김준용",
            timestamp="00:00:00",
            text="완전히 다른 이야기입니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=10,
            fromName="유상완",
            timestamp="00:00:06",
            text="또 다른 내용입니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [9, 10]
    assert [segment.speaker for segment in merged] == ["김준용", "유상완"]
    assert [segment.text for segment in merged] == [
        "완전히 다른 이야기입니다",
        "또 다른 내용입니다",
    ]


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


def test_merge_live_transcript_uses_direct_match_member_without_vote_threshold():
    segments = [
        _segment(1, "Speaker 0", "00:00:00", "웹소켓 발화 로그로 보정해야 합니다"),
        _segment(2, "Speaker 1", "00:00:05", "STT 전사와 합쳐서 보여주겠습니다"),
        _segment(3, "Speaker 0", "00:00:10", "최종 결과를 저장하겠습니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=1,
            fromName="유상완",
            timestamp="00:00:00",
            text="웹소켓 발화 로그로 보정해야 합니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=2,
            fromName="유진",
            timestamp="00:00:05",
            text="STT 전사와 합쳐서 보여주겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=3,
            fromName="김준용",
            timestamp="00:00:10",
            text="최종 결과를 저장하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [1, 2, 3]
    assert [segment.speaker for segment in merged] == ["유상완", "유진", "김준용"]


def test_merge_live_transcript_preserves_unmatched_live_messages_as_segments():
    segments = [
        _segment(
            1,
            "Speaker 0",
            "00:00:00",
            "웹소켓 발화 로그로 화자를 보정하고 STT 전사를 강화하겠습니다",
        ),
        _segment(2, "Speaker 1", "00:00:12", "결과를 저장하겠습니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=1,
            fromName="유상완",
            timestamp="00:00:00",
            text="웹소켓 발화 로그로 화자를 보정하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=2,
            fromName="유진",
            timestamp="00:00:06",
            text="STT 전사를 같이 보고 강화하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=3,
            fromName="김준용",
            timestamp="00:00:12",
            text="결과를 저장하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [1, 2, 3]
    assert [segment.speaker for segment in merged] == ["유상완", "유진", "김준용"]
    assert [segment.text for segment in merged] == [
        "웹소켓 발화 로그로 화자를 보정하겠습니다",
        "STT 전사를 같이 보고 강화하겠습니다",
        "결과를 저장하겠습니다",
    ]


def test_merge_live_transcript_orders_unmatched_iso_live_messages_by_anchor():
    segments = [
        _segment(1, "Speaker 0", "00:01:48", "첫 번째 안건을 공유하겠습니다"),
        _segment(2, "Speaker 1", "00:02:10", "세 번째 안건을 정리하겠습니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=1,
            fromName="유상완",
            timestamp="2026-05-04T17:00:00",
            text="첫 번째 안건을 공유하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=2,
            fromName="유진",
            timestamp="2026-05-04T17:00:02",
            text="두 번째 안건도 추가로 확인하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=3,
            fromName="김준용",
            timestamp="2026-05-04T17:00:04",
            text="세 번째 안건을 정리하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [1, 2, 3]
    assert [segment.text for segment in merged] == [
        "첫 번째 안건을 공유하겠습니다",
        "두 번째 안건도 추가로 확인하겠습니다",
        "세 번째 안건을 정리하겠습니다",
    ]


def test_merge_live_transcript_uses_websocket_only_when_no_stt_matches():
    segments = [
        _segment(1, "Speaker 0", "00:00:00", "노이즈가 심해서 내용이 다릅니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=1,
            fromName="유상완",
            timestamp="2026-05-04T17:00:00",
            text="회의 분석 결과를 확인하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=2,
            fromName="유진",
            timestamp="2026-05-04T17:00:02",
            text="전사 결과도 같이 확인하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [1, 2]
    assert [segment.text for segment in merged] == [
        "회의 분석 결과를 확인하겠습니다",
        "전사 결과도 같이 확인하겠습니다",
    ]


def test_merge_live_transcript_global_match_prevents_greedy_steal():
    segments = [
        _segment(1, "Speaker 0", "00:00:00", "웹소켓 로그 확인"),
        _segment(2, "Speaker 1", "00:00:06", "전사 결과를 저장하겠습니다"),
    ]
    live_messages = [
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=1,
            fromName="유상완",
            timestamp="00:00:00",
            text="웹소켓 로그를 확인하겠습니다",
        ),
        LiveTranscriptMessage(
            type="TEXT",
            meetingId=143,
            fromMemberId=2,
            fromName="유진",
            timestamp="00:00:06",
            text="전사 결과를 저장하겠습니다",
        ),
    ]

    merged = merge_live_transcript(segments, live_messages)

    assert [segment.member_id for segment in merged] == [1, 2]
    assert [segment.speaker for segment in merged] == ["유상완", "유진"]


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
