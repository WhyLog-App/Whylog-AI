from app.domains.meeting_analysis.schemas import (
    Application,
    MeetingAnalysis,
    MeetingAnalysisResult,
)
from app.domains.meeting_analysis.services.extraction import (
    CONCRETE_APPLICATION_RULE,
    NOUN_ENDING_REASON_RULE,
    _normalize_application_reason_outputs,
    _normalize_overall_reason_outputs,
)


class TestApplicationReasonFormat:
    def test_reason_prompt_rule_requires_noun_ending(self):
        joined_rule = "\n".join(NOUN_ENDING_REASON_RULE)

        assert "명사형 어미" in joined_rule
        assert "절대 출력하지 않는다" in joined_rule
        assert "응답 속도 개선 필요성" in joined_rule

    def test_application_prompt_rule_requires_concrete_action(self):
        joined_rule = "\n".join(CONCRETE_APPLICATION_RULE)

        assert "실제 실행/반영 단위" in joined_rule
        assert "두루뭉술한 표현은 절대 사용하지 않는다" in joined_rule
        assert "WebSocket 발화로그 기준으로 STT 화자 보정 적용" in joined_rule

    def test_application_reasons_are_normalized_to_noun_endings(self):
        result = MeetingAnalysisResult(
            applications=[
                Application(
                    application_title="전사 정확도 개선",
                    application_reasons=[
                        "사용자 경험이 저하된다.",
                        "응답 속도 개선이 필요하다.",
                        "데이터 정합성을 확보한다.",
                    ],
                )
            ]
        )

        _normalize_application_reason_outputs(result)

        assert result.applications[0].application_reasons == [
            "사용자 경험 저하",
            "응답 속도 개선 필요성",
            "데이터 정합성 확보",
        ]

    def test_overall_reasons_are_normalized_to_noun_endings(self):
        analysis = MeetingAnalysis(
            application_reasons=[
                "운영 리스크를 줄인다.",
                "재시도 정책을 보장한다.",
            ]
        )

        _normalize_overall_reason_outputs(analysis)

        assert analysis.application_reasons == [
            "운영 리스크 감소",
            "재시도 정책 보장",
        ]
