from pydantic import BaseModel, Field

from app.domains.transcribe.schemas import TranscribeSegment


class MeetingInfo(BaseModel):
    title: str = Field(default="", description="회의 제목")
    purpose: str = Field(default="", description="회의 목적")
    duration: str = Field(default="", description="회의 길이/시간")


class MeetingAnalysis(BaseModel):
    meeting_info: MeetingInfo = Field(default_factory=MeetingInfo)
    topics: list[str] = Field(default_factory=list, description="논의된 주제 목록")
    core_context: list[str] = Field(
        default_factory=list, description="프로젝트 배경/제약 컨텍스트"
    )
    application_titles: list[str] = Field(
        default_factory=list,
        description=(
            "회의에서 도출된 적용사항 제목 목록. "
            "비동기 실행의 summary_ready 단계에서는 임시값일 수 있으며, "
            "completed 단계에서 applications 기준으로 재동기화될 수 있습니다."
        ),
    )
    application_reasons: list[str] = Field(
        default_factory=list,
        description="적용사항 근거 통합 목록(문장 단위)",
    )


class ApplicationTimelineItem(BaseModel):
    timestamp: str = Field(description="해당 단계 시각(HH:MM:SS)")
    step: str = Field(description="이슈제기/대안논의/적용합의")
    member_id: int | None = Field(
        default=None,
        description=(
            "WebSocket 발화 로그와 매칭된 Spring 멤버 ID. "
            "실시간 발화 로그가 없거나 매칭 신뢰도가 낮은 경우 "
            "null로 반환될 수 있습니다."
        ),
    )
    content: str = Field(description="타임라인 요약 한 문장")
    utterance: str = Field(description="실제 발화 원문")


class Application(BaseModel):
    application_id: int | None = Field(
        default=None,
        description=(
            "호출 측(예: Spring)에서 관리하는 적용사항 ID. "
            "AI가 처음 생성한 분석 결과에서는 null이며, "
            "Spring이 DB 저장 후 발급한 applicationId를 임베딩 요청 때 채워 넣습니다."
        ),
    )
    application_title: str = Field(description="적용사항 제목")
    application_reasons: list[str] = Field(
        default_factory=list,
        description="해당 적용사항의 근거 목록(1근거=1문장)",
    )
    timeline: list[ApplicationTimelineItem] = Field(
        default_factory=list,
        description="적용사항 도출 타임라인(이슈제기→대안논의→적용합의)",
    )


class MeetingAnalysisResult(BaseModel):
    overall_analysis: MeetingAnalysis = Field(default_factory=MeetingAnalysis)
    applications: list[Application] = Field(
        default_factory=list,
        description="회의에서 도출된 적용사항 목록",
    )
    other_mentions: list[str] = Field(
        default_factory=list,
        description="적용사항으로 확정되지 않은 기술 제언/추가 과제",
    )


class MeetingAnalysisRequest(BaseModel):
    meeting_id: str | None = Field(
        default=None,
        description="호출 측(예: Spring)에서 관리하는 회의 ID(선택)",
    )
    project_id: str | None = Field(
        default=None,
        description="호출 측(예: Spring)에서 관리하는 프로젝트 ID(선택)",
    )
    transcript_segments: list[TranscribeSegment] = Field(
        description=(
            "후처리 완료 전사 세그먼트 배열. "
            "원칙적으로 /api/transcribe 또는 /api/transcribe/applications 계열에서 "
            "생성된 포맷을 그대로 전달하는 것을 권장합니다."
        )
    )


class MeetingAnalysisResponse(BaseModel):
    meeting_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 meeting_id echo",
    )
    project_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 project_id echo",
    )
    analysis_result: MeetingAnalysisResult


# ── Application Embedding DTO ──


class EmbeddedDocument(BaseModel):
    document_id: str = Field(description="ChromaDB 문서 ID")
    text: str = Field(description="임베딩에 사용된 정규화 텍스트")
    application_id: int | None = Field(
        default=None,
        description="Spring이 전달한 원본 적용사항 ID",
    )
    application_title: str = Field(description="원본 적용사항 제목")


class ApplicationEmbeddingRequest(BaseModel):
    meeting_id: str = Field(
        description="회의 ID (필수)",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$",
    )
    project_id: str | None = Field(
        default=None,
        description="프로젝트 ID (선택)",
    )
    analysis_result: MeetingAnalysisResult = Field(
        description="회의 분석 결과(applications 포함)",
    )


class ApplicationEmbeddingResponse(BaseModel):
    meeting_id: str = Field(description="처리된 회의 ID")
    project_id: str | None = Field(default=None, description="프로젝트 ID")
    total_documents: int = Field(description="저장된 문서 수")
    document_ids: list[str] = Field(description="저장된 문서 ID 목록")
    documents: list[EmbeddedDocument] = Field(
        description="저장된 문서 상세 목록",
    )
