from fastapi import APIRouter, BackgroundTasks

from app.core.errors import AppServiceError
from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.commit.schemas import (
    CommitAnalyzeRequest,
    CommitAnalyzeResponse,
    DecisionCommitMatchRequest,
    DecisionCommitMatchResponse,
)
from app.domains.commit.services.diff_filter import filter_changed_files
from app.domains.commit.services.matching import match_decisions_with_commits
from app.domains.commit.services.summarize import (
    generate_embedding_text,
    summarize_commit,
)

router = APIRouter(prefix="/commit", tags=["commit"])


# POST /api/commit/analyze — Spring에서 커밋 데이터를 받아 LLM 요약 후 반환
@router.post(
    "/analyze",
    response_model=ApiResponse[CommitAnalyzeResponse],
    summary="커밋 분석 API",
    description="Spring에서 커밋 메시지와 diff를 받아 "
    "LLM(Gemini)으로 요약한 결과를 반환합니다.\n\n"
    "**백그라운드 임베딩 저장:**\n"
    "- 구조화 임베딩 텍스트 생성 → Gemini Embedding API 벡터 변환 → "
    "ChromaDB(commit_embeddings) 저장\n"
    "- 동일 commit_id 재호출 시 기존 데이터를 덮어씁니다 (upsert)\n"
    "- commit_hash가 포함되면 매칭 결과 식별용 메타데이터로 함께 저장합니다\n"
    "- 문서 ID: commit_{commit_id}\n"
    "- 임베딩 텍스트: title: {repo} {subject} | text: "
    "변경요약: | 기술키워드: | 변경방향: | 파일맥락: ",
    responses={
        400: {
            "model": ApiErrorResponse,
            "description": "분석할 수 있는 변경 파일이 없습니다.",
        },
        422: {
            "model": ApiErrorResponse,
            "description": "요청 스키마 검증 실패(예: message 누락, 빈 파일 목록).",
        },
        500: {
            "model": ApiErrorResponse,
            "description": "서버 설정 오류(예: GEMINI_API_KEY 누락).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "Gemini 호출 실패 또는 응답 파싱 오류.",
        },
        504: {
            "model": ApiErrorResponse,
            "description": "Gemini 응답 시간 초과.",
        },
    },
)
async def analyze_commit(
    request: CommitAnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> ApiResponse[CommitAnalyzeResponse]:
    filtered_files = filter_changed_files(request.changed_file_list)
    if not filtered_files:
        raise AppServiceError("분석할 수 있는 변경 파일이 없습니다.", status_code=400)

    summary = await summarize_commit(request.message, filtered_files)
    background_tasks.add_task(
        generate_embedding_text,
        request.commit_id,
        request.commit_hash,
        request.repository,
        request.message,
        filtered_files,
    )
    return ok_response(
        CommitAnalyzeResponse(commit_id=request.commit_id, summary=summary)
    )


@router.post(
    "/match",
    response_model=ApiResponse[DecisionCommitMatchResponse],
    summary="결정사항-커밋 유사도 매칭",
    description=(
        "회의 결정사항(applied_item 단위)과 "
        "커밋 임베딩 후보를 유사도 기반으로 매칭합니다.\n\n"
        "점수 정책(100점):\n"
        "- 의미 유사성 50\n"
        "- 기술 키워드 일치도 30\n"
        "- 파일/모듈 맥락 일치도 20\n"
        "- 반대 의미(도입 vs 제거)는 semantic 0 처리\n"
        "- 추상 커밋/모호한 결정사항은 보정 감점\n\n"
        "상태 구간:\n"
        "- 70~100: APPLIED\n"
        "- 50~69: PARTIAL\n"
        "- 0~49: 미표시(UNAPPLIED)"
    ),
    responses={
        422: {
            "model": ApiErrorResponse,
            "description": "요청 스키마 검증 실패(예: meeting_id 형식, top_k 범위).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "ChromaDB 조회 실패.",
        },
    },
)
async def match_decision_commits(
    request: DecisionCommitMatchRequest,
) -> ApiResponse[DecisionCommitMatchResponse]:
    result = await match_decisions_with_commits(request)
    return ok_response(
        result=result,
        code="COMMIT_MATCH_200",
        message="결정사항-커밋 매칭 분석이 완료되었습니다.",
    )
