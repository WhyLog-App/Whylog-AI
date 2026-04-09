from fastapi import APIRouter, BackgroundTasks

from app.core.errors import AppServiceError
from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.commit.schemas import CommitAnalyzeRequest, CommitAnalyzeResponse
from app.domains.commit.services.diff_filter import filter_changed_files
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
    "LLM으로 요약한 결과를 반환합니다. "
    "임베딩용 상세 텍스트는 백그라운드에서 생성됩니다.",
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
        request.message,
        filtered_files,
    )
    return ok_response(
        CommitAnalyzeResponse(commit_id=request.commit_id, summary=summary)
    )
