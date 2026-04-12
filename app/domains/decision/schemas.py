from pydantic import BaseModel, Field

from app.domains.transcribe.schemas import TranscribeSegment


class MeetingInfo(BaseModel):
    title: str = Field(default="", description="회의 제목")
    purpose: str = Field(default="", description="회의 목적")
    duration: str = Field(default="", description="회의 길이/시간")


class OverallAnalysis(BaseModel):
    meeting_info: MeetingInfo = Field(default_factory=MeetingInfo)
    topics: list[str] = Field(default_factory=list, description="논의된 주제 목록")
    core_context: list[str] = Field(
        default_factory=list, description="프로젝트 배경/제약 컨텍스트"
    )
    final_decisions_list: list[str] = Field(
        default_factory=list,
        description=(
            "최종 결정사항 제목 목록. "
            "비동기 실행의 summary_ready 단계에서는 임시값일 수 있으며, "
            "completed 단계에서 decision_cards 기준으로 재동기화될 수 있습니다."
        ),
    )
    all_decision_reasons: list[str] = Field(
        default_factory=list,
        description="결정 근거 통합 목록(문장 단위)",
    )


class DecisionTimelineItem(BaseModel):
    timestamp: str = Field(description="해당 단계 시각(HH:MM:SS)")
    step: str = Field(description="이슈제기/대안논의/최종합의")
    speaker_id: str | None = Field(
        default=None,
        description=(
            '타임라인 발화 화자 ID(예: "Speaker 0"). '
            "짧은 응답어(예: 네/확인)처럼 화자 추정 신뢰도가 낮은 경우 "
            "억지 매핑을 피하기 위해 null로 반환될 수 있습니다. "
            'UI에서는 "미확인 화자" 등으로 표시를 권장합니다.'
        ),
    )
    content: str = Field(description="타임라인 요약 한 문장")
    utterance: str = Field(description="실제 발화 원문")


class DecisionCard(BaseModel):
    decision_title: str = Field(description="결정사항 제목")
    applied_items: list[str] = Field(default_factory=list, description="액션 아이템")
    decision_reasons: list[str] = Field(
        default_factory=list,
        description="해당 결정의 근거 목록(1근거=1문장)",
    )
    timeline: list[DecisionTimelineItem] = Field(
        default_factory=list,
        description="결정 타임라인(이슈제기→대안논의→최종합의)",
    )


class DecisionExtractionResult(BaseModel):
    overall_analysis: OverallAnalysis = Field(default_factory=OverallAnalysis)
    decision_cards: list[DecisionCard] = Field(
        default_factory=list,
        description="결정사항 카드 목록",
    )
    other_mentions: list[str] = Field(
        default_factory=list,
        description="결정으로 확정되지 않은 기술 제언/추가 과제",
    )


class DecisionExtractRequest(BaseModel):
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
            "원칙적으로 /api/transcribe 또는 /api/transcribe/decisions 계열에서 "
            "생성된 포맷을 그대로 전달하는 것을 권장합니다."
        )
    )


class DecisionExtractResponse(BaseModel):
    meeting_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 meeting_id echo",
    )
    project_id: str | None = Field(
        default=None,
        description="요청에서 전달받은 project_id echo",
    )
    decision_result: DecisionExtractionResult


# ── Decision Embedding DTO ──


class EmbeddedDocument(BaseModel):
    document_id: str = Field(description="ChromaDB 문서 ID")
    text: str = Field(description="임베딩에 사용된 정규화 텍스트")
    decision_title: str = Field(description="원본 결정사항 제목")
    applied_item: str = Field(description="원본 적용사항")


class DecisionEmbeddingRequest(BaseModel):
    meeting_id: str = Field(
        description="회의 ID (필수)",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$",
    )
    project_id: str | None = Field(
        default=None,
        description="프로젝트 ID (선택)",
    )
    decision_result: DecisionExtractionResult = Field(
        description="의사결정 추출 결과 (decision_cards 포함)",
    )


class DecisionEmbeddingResponse(BaseModel):
    meeting_id: str = Field(description="처리된 회의 ID")
    project_id: str | None = Field(default=None, description="프로젝트 ID")
    total_documents: int = Field(description="저장된 문서 수")
    document_ids: list[str] = Field(description="저장된 문서 ID 목록")
    documents: list[EmbeddedDocument] = Field(
        description="저장된 문서 상세 목록",
    )
