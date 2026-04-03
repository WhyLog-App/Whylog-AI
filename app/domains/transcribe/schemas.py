from pydantic import BaseModel, Field


# 전사 API의 응답 한 줄(발화 세그먼트) 타입 정의
class TranscribeSegment(BaseModel):
    message_id: int = Field(description="발화 순서 번호")
    speaker: str = Field(description='화자 ID/이름 (예: "Speaker 0")')
    start_time: str = Field(description="발화 시작 시각(HH:MM:SS)")
    end_time: str = Field(description="발화 종료 시각(HH:MM:SS)")
    text: str = Field(description="전사된 발화 텍스트")
    is_final: bool = Field(description="최종 결과 여부(현재는 항상 true)")
