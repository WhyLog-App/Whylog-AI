from app.domains.meeting_analysis.schemas import (
    Application,
    ApplicationTimelineItem,
    MeetingAnalysisResult,
)
from app.domains.meeting_analysis.services.extraction import (
    APPLICATION_POLICY_PROMPT,
    APPLICATIONS_ONLY_PROMPT,
    _normalize_timeline_member_ids,
)
from app.domains.transcribe.schemas import TranscribeSegment


class TestTimelineMemberId:
    def test_timeline_prompt_uses_member_id_not_legacy_speaker_field(self):
        prompt = f"{APPLICATION_POLICY_PROMPT}\n{APPLICATIONS_ONLY_PROMPT}"

        assert '"member_id": 1' in prompt
        assert "speaker" + "_id" not in prompt

    def test_timeline_member_id_is_inferred_from_transcript_segment(self):
        result = MeetingAnalysisResult(
            applications=[
                Application(
                    application_title="Swagger 에러 응답 예시 문서화",
                    application_reasons=[],
                    timeline=[
                        ApplicationTimelineItem(
                            timestamp="00:00:03",
                            step="적용합의",
                            member_id=None,
                            content="ApiErrorCodeExample 추가 합의",
                            utterance="ApiErrorCodeExample을 추가하는 걸로 하죠.",
                        )
                    ],
                )
            ]
        )
        segments = [
            TranscribeSegment(
                message_id=1,
                speaker="김준용",
                member_id=7,
                start_time="00:00:00",
                end_time="00:00:05",
                text="ApiErrorCodeExample을 추가하는 걸로 하죠.",
                is_final=True,
            )
        ]

        _normalize_timeline_member_ids(result, segments)

        assert result.applications[0].timeline[0].member_id == 7

    def test_timeline_member_id_is_null_when_transcript_has_no_member_id(self):
        result = MeetingAnalysisResult(
            applications=[
                Application(
                    application_title="STT 정확도 개선",
                    application_reasons=[],
                    timeline=[
                        ApplicationTimelineItem(
                            timestamp="00:00:03",
                            step="이슈제기",
                            member_id=999,
                            content="전사 정확도 개선 필요성 제기",
                            utterance="전사 정확도를 개선해야 합니다.",
                        )
                    ],
                )
            ]
        )
        segments = [
            TranscribeSegment(
                message_id=1,
                speaker="Speaker 0",
                member_id=None,
                start_time="00:00:00",
                end_time="00:00:05",
                text="전사 정확도를 개선해야 합니다.",
                is_final=True,
            )
        ]

        _normalize_timeline_member_ids(result, segments)

        assert result.applications[0].timeline[0].member_id is None
