from app.domains.meeting_analysis.schemas import (
    Application,
    MeetingAnalysis,
    MeetingAnalysisResult,
)
from app.domains.meeting_analysis.services.extraction import (
    APPLICATION_POLICY_PROMPT,
    APPLICATIONS_ONLY_PROMPT,
    CONCRETE_APPLICATION_RULE,
    NOUN_ENDING_REASON_RULE,
    TIMELINE_EVIDENCE_ALIGNMENT_RULE,
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

    def test_timeline_prompt_rule_requires_intent_aligned_evidence(self):
        joined_rule = "\n".join(TIMELINE_EVIDENCE_ALIGNMENT_RULE)

        assert "핵심 작업 의도" in joined_rule
        assert "같은 화면, API, 도메인명이 겹쳐도" in joined_rule
        assert "무엇을 바꾸기로 했는가" in joined_rule
        assert "커밋 상세 페이지 로딩 스피너 컴포넌트 적용" in joined_rule
        assert "응답 필드명 통일" in joined_rule

    def test_timeline_evidence_alignment_rule_is_used_by_analysis_prompts(self):
        for prompt in (APPLICATION_POLICY_PROMPT, APPLICATIONS_ONLY_PROMPT):
            assert "타임라인 근거 정렬 규칙" in prompt
            assert "같은 application의 timeline에 넣지 않는다" in prompt
            assert "공통 범위 키워드만 겹치는 발화" in prompt

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
