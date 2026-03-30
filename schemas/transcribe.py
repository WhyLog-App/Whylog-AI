from pydantic import BaseModel


# 전사 API의 응답 한 줄(발화 세그먼트) 타입 정의
class TranscribeSegment(BaseModel):
    message_id: int  # 발화 순서 번호
    speaker: str  # 화자 이름 (e.g. "Speaker 0")
    start_time: str  # 발화 시작 시각 (HH:MM:SS)
    end_time: str  # 발화 종료 시각 (HH:MM:SS)
    text: str  # 전사된 텍스트
    is_final: bool  # 최종 결과 여부 (현재는 항상 True)
