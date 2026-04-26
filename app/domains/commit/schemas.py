from pydantic import BaseModel, Field


class ChangedFile(BaseModel):
    file_name: str = Field(min_length=1, description="변경된 파일 경로")
    changed_code: str = Field(min_length=1, description="unified diff 형식의 변경 코드")


class CommitAnalyzeRequest(BaseModel):
    commit_id: int = Field(ge=0, description="커밋 ID")
    commit_hash: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{7,64}$",
        description="Git 커밋 해시",
    )
    repository_id: int = Field(ge=0, description="레포지토리 ID")
    message: str = Field(min_length=1, description="커밋 메시지")
    changed_file_list: list[ChangedFile] = Field(
        min_length=1, description="변경된 파일 목록"
    )


class CommitAnalyzeResponse(BaseModel):
    commit_id: int = Field(description="커밋 ID")
    summary: str = Field(description="커밋 요약")


class MatchScoreBreakdown(BaseModel):
    semantic: int = Field(description="의미 유사성 점수(0~50)")
    keyword: int = Field(description="기술 키워드 일치도 점수(0~30)")
    context: int = Field(description="파일/모듈 맥락 점수(0~20)")
    penalty: int = Field(description="보정 감점 합계(0~20)")
    total: int = Field(description="최종 신뢰도 점수(0~100)")


class ApplicationCommitMatchRequest(BaseModel):
    meeting_id: str = Field(
        description="적용사항 임베딩을 조회할 회의 ID",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$",
    )
    repository_id: int | None = Field(
        default=None,
        ge=0,
        description="레포지토리 ID (선택, 미지정 시 전체 후보 조회)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=30,
        description="적용사항별 추천 커밋 상위 K",
    )


class MatchedCommit(BaseModel):
    commit_id: int | None = Field(default=None, description="커밋 ID")
    commit_ref: str | None = Field(default=None, description="커밋 참조 ID")
    commit_hash: str | None = Field(default=None, description="커밋 해시")
    commit_message: str | None = Field(default=None, description="커밋 메시지")
    repository_id: int | None = Field(default=None, description="레포지토리 ID")
    confidence: int = Field(description="신뢰도 점수(0~100)")
    reason: str = Field(description="추천 사유 요약")
    score_breakdown: MatchScoreBreakdown = Field(description="점수 구성 상세")
    direction_primary: str | None = Field(default=None, description="대표 변경 방향")
    direction_multi: list[str] = Field(
        default_factory=list, description="다중 변경 방향 목록"
    )
    tech_keywords: list[str] = Field(
        default_factory=list, description="커밋 기술 키워드 목록"
    )
    module_tags: list[str] = Field(default_factory=list, description="모듈/경로 태그")


class ApplicationCommitMatchItem(BaseModel):
    application_id: int | None = Field(default=None, description="적용사항 ID")
    application_document_id: str = Field(description="적용사항 문서 ID")
    application_title: str = Field(description="적용사항 제목")
    recommended_commits: list[MatchedCommit] = Field(
        default_factory=list,
        description="신뢰도 내림차순 추천 커밋 목록",
    )


class ApplicationCommitMatchResponse(BaseModel):
    meeting_id: str = Field(description="회의 ID")
    repository_id: int | None = Field(default=None, description="레포지토리 ID 필터")
    total_applications: int = Field(description="조회된 적용사항 문서 수")
    matched_applications: int = Field(description="추천 결과가 존재하는 적용사항 수")
    applications: list[ApplicationCommitMatchItem] = Field(
        default_factory=list, description="적용사항 단위 매칭 결과"
    )
    notice: str = Field(
        default="신뢰도는 AI 분석 기반 추정값입니다.",
        description="신뢰도 안내 문구",
    )
