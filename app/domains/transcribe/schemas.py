from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LiveTranscriptMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str = Field(description='WebSocket 메시지 타입(예: "TEXT")')
    meeting_id: int | str | None = Field(
        default=None,
        alias="meetingId",
        description="회의 ID",
    )
    from_member_id: int | None = Field(
        default=None,
        alias="fromMemberId",
        description="발화자 멤버 ID",
    )
    from_name: str | None = Field(
        default=None,
        alias="fromName",
        description="발화자 이름",
    )
    timestamp: str = Field(description="회의 기준 발화 시작 시간 문자열")
    target_member_id: int | None = Field(
        default=None,
        alias="targetMemberId",
        description="대상 멤버 ID(선택)",
    )
    text: str = Field(description="WebSocket 실시간 발화 텍스트")
    payload: dict[str, Any] | None = Field(default=None, description="부가 payload")


# 전사 API의 응답 한 줄(발화 세그먼트) 타입 정의
class TranscribeSegment(BaseModel):
    message_id: int = Field(description="발화 순서 번호")
    speaker: str = Field(description='화자 ID/이름 (예: "Speaker 0")')
    member_id: int | None = Field(
        default=None,
        description="WebSocket 발화 로그와 매칭된 발화자 멤버 ID",
    )
    start_time: str = Field(description="발화 시작 시각(HH:MM:SS)")
    end_time: str = Field(description="발화 종료 시각(HH:MM:SS)")
    text: str = Field(description="전사된 발화 텍스트")
    is_final: bool = Field(description="최종 결과 여부(현재는 항상 true)")
