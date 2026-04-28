from typing import Literal

from pydantic import BaseModel, Field

from app.domains.decision.schemas import MeetingAnalysisResult
from app.domains.transcribe.schemas import TranscribeSegment

RunStatus = Literal["queued", "processing", "completed", "failed"]
RunPhase = Literal[
    "queued",
    "transcribing",
    "transcript_ready",
    "summary_ready",
    "applications_ready",
    "failed",
]


class TranscribeAnalysisResponse(BaseModel):
    meeting_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 meeting_id echo",
    )
    project_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 project_id echo",
    )
    transcript_segments: list[TranscribeSegment] = Field(
        default_factory=list,
        description="STT + 후처리 완료 세그먼트",
    )
    analysis_result: MeetingAnalysisResult = Field(
        description="회의 분석 결과",
    )


class TranscribeAnalysisRunAccepted(BaseModel):
    run_id: str = Field(description="비동기 실행 식별자")
    status: RunStatus = Field(description='실행 상태("queued" 고정으로 시작)')
    phase: RunPhase = Field(description='실행 단계("queued" 고정으로 시작)')
    meeting_id: str | None = Field(default=None, description="요청 meeting_id echo")
    project_id: str | None = Field(default=None, description="요청 project_id echo")


class TranscribeAnalysisRunStatus(BaseModel):
    run_id: str = Field(description="비동기 실행 식별자")
    status: RunStatus = Field(description="queued/processing/completed/failed")
    phase: RunPhase = Field(
        description=(
            "queued/transcribing/transcript_ready/summary_ready/"
            "applications_ready/failed. "
            "summary_ready는 요약 정보가 채워진 중간 상태이며, "
            "applications_ready(completed)가 최종 상태입니다."
        )
    )
    meeting_id: str | None = Field(default=None, description="요청 meeting_id echo")
    project_id: str | None = Field(default=None, description="요청 project_id echo")
    submitted_at: str = Field(description="실행 접수 시각(UTC ISO8601)")
    started_at: str | None = Field(
        default=None,
        description="실행 시작 시각(UTC ISO8601)",
    )
    finished_at: str | None = Field(
        default=None,
        description="실행 종료 시각(UTC ISO8601)",
    )
    error: str | None = Field(default=None, description="실패 시 오류 메시지")
    result: TranscribeAnalysisResponse | None = Field(
        default=None,
        description=(
            "중간/최종 결과. "
            "transcribing 단계에서는 null, "
            "transcript_ready에서는 transcript_segments 중심, "
            "summary_ready에서는 overall_analysis가 추가되며, "
            "applications_ready에서는 applications 포함 최종 결과가 채워집니다."
        ),
    )
