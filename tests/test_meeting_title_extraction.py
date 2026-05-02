from app.domains.meeting_analysis.schemas import MeetingAnalysis, MeetingInfo
from app.domains.meeting_analysis.services.extraction import (
    _is_generic_meeting_title,
    _refine_meeting_title,
)


class TestMeetingTitleRefinement:
    def test_generic_whylog_title_is_refined_from_topics(self):
        analysis = MeetingAnalysis(
            meeting_info=MeetingInfo(
                title="Whylog 프로젝트 회의",
                purpose="회의 시스템 점검 및 참가자 식별 확인",
            ),
            topics=[
                "회의 시스템 연결 상태 점검",
                "참가자 표시 이름 오류 확인",
            ],
            core_context=[],
            application_titles=[],
            application_reasons=[],
        )

        _refine_meeting_title(analysis)

        assert analysis.meeting_info.title == (
            "회의 시스템 연결 상태 점검 및 참가자 표시 이름 오류 확인 회의"
        )

    def test_generic_feature_check_title_is_refined_from_topics(self):
        analysis = MeetingAnalysis(
            meeting_info=MeetingInfo(
                title="Whylog 프로젝트 기능 점검 회의",
                purpose="기능 동작 확인 및 이슈 식별",
            ),
            topics=[
                "기능 동작 테스트",
                "역선전 이슈 식별",
            ],
            core_context=[],
            application_titles=["역선전 노출 오류 수정"],
            application_reasons=[],
        )

        _refine_meeting_title(analysis)

        assert analysis.meeting_info.title == (
            "기능 동작 테스트 및 역선전 이슈 식별 회의"
        )

    def test_specific_title_is_not_overwritten(self):
        analysis = MeetingAnalysis(
            meeting_info=MeetingInfo(
                title="데모 시나리오와 깃허브 연동 확인 회의",
                purpose="데모 시나리오 구성 및 깃허브 연동 확인",
            ),
            topics=["깃허브 연동 확인"],
            core_context=[],
            application_titles=[],
            application_reasons=[],
        )

        _refine_meeting_title(analysis)

        assert analysis.meeting_info.title == ("데모 시나리오와 깃허브 연동 확인 회의")

    def test_empty_title_falls_back_to_application_title(self):
        analysis = MeetingAnalysis(
            meeting_info=MeetingInfo(title="데이터 없음"),
            topics=[],
            core_context=[],
            application_titles=["역선전 노출 오류 수정"],
            application_reasons=[],
        )

        _refine_meeting_title(analysis)

        assert analysis.meeting_info.title == "역선전 노출 오류 수정 회의"

    def test_generic_title_detection(self):
        assert _is_generic_meeting_title("Whylog meeting")
        assert _is_generic_meeting_title("Whylog 프로젝트 기능 점검 회의")
        assert not _is_generic_meeting_title("참가자 표시 이름 오류 확인 회의")
