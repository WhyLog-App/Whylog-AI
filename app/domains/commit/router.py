from fastapi import APIRouter

from app.core.responses import ApiResponse, ok_response
from app.domains.commit.schemas import CommitAnalyzeRequest, CommitAnalyzeResponse
from app.domains.commit.services.summarize import summarize_commit

router = APIRouter(prefix="/commit", tags=["commit"])


# POST /api/commit/analyze — Spring에서 커밋 데이터를 받아 LLM 요약 후 반환
@router.post(
    "/analyze",
    response_model=ApiResponse[CommitAnalyzeResponse],
    summary="커밋 분석 API",
    description="Spring에서 커밋 메시지와 diff를 받아 LLM으로 요약한 결과를 반환합니다.",
)
async def analyze_commit(request: CommitAnalyzeRequest) -> ApiResponse[CommitAnalyzeResponse]:
    summary = await summarize_commit(request.message, request.changed_file_list)
    return ok_response(CommitAnalyzeResponse(commit_id=request.commit_id, summary=summary))
