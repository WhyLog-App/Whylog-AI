from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.errors import AppServiceError
from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.commit.schemas import (
    ApplicationCommitMatchRequest,
    ApplicationCommitMatchResponse,
    CommitAnalyzeRequest,
    CommitAnalyzeResponse,
    CommitAnalyzeRunAccepted,
    CommitAnalyzeRunStatus,
)
from app.domains.commit.services.analyze_runs import (
    create_commit_analyze_run as create_commit_analyze_run_record,
)
from app.domains.commit.services.analyze_runs import (
    get_commit_analyze_run_status,
    run_commit_analyze_pipeline,
)
from app.domains.commit.services.diff_filter import filter_changed_files
from app.domains.commit.services.matching import match_applications_with_commits
from app.domains.commit.services.summarize import (
    generate_embedding_text,
    summarize_commit,
)

router = APIRouter(prefix="/commit", tags=["commit"])

COMMIT_ANALYZE_ASYNC_GUIDE = (
    "Spring 연동 가이드:\n"
    "1) 레포지토리 커밋 동기화 시 커밋별로 "
    "POST /api/commit/analyze/runs를 호출합니다.\n"
    "2) 응답의 run_id로 GET /api/commit/analyze/runs/{run_id}를 폴링합니다.\n"
    "3) phase=summary_ready부터 커밋 요약을 확인할 수 있습니다.\n"
    "4) status=completed && phase=embedding_ready 확인 후 "
    "POST /api/commit/match를 호출합니다.\n"
    "5) embedding_ready 전에는 방금 분석한 커밋이 추천 후보에 "
    "아직 반영되지 않았을 수 있습니다.\n"
    "6) status=failed 시 error를 기록하고 필요 시 재시도합니다.\n"
    "7) run 조회 404는 만료/정리/재기동 유실 가능성이 있으므로 "
    "재요청 정책을 둡니다."
)


# POST /api/commit/analyze — Spring에서 커밋 데이터를 받아 LLM 요약 후 반환
@router.post(
    "/analyze",
    response_model=ApiResponse[CommitAnalyzeResponse],
    summary="커밋 분석 API",
    description="Spring에서 커밋 메시지와 diff를 받아 "
    "LLM(Gemini)으로 요약한 결과를 반환합니다.\n\n"
    "**Spring 연동 입력:**\n"
    "- `commit_hash`는 외부 식별자로 필수입니다 (ChromaDB 저장 키).\n"
    "- `commit_id`는 Spring DB PK로 선택입니다. 보내면 매칭 응답 join 편의를 위해 "
    "메타에 함께 저장됩니다.\n"
    "- Spring repository 식별자는 `repository_id`로 전달합니다.\n\n"
    "**백그라운드 임베딩 저장:**\n"
    "- 구조화 임베딩 텍스트 생성 → Gemini Embedding API 벡터 변환 → "
    "ChromaDB(commit_embeddings) 저장\n"
    "- 동일 commit_hash 재호출 시 기존 데이터를 덮어씁니다 (upsert)\n"
    "- 문서 ID: commit_{commit_hash}\n"
    "- 임베딩 텍스트: title: repository-{repository_id} {subject} | text: "
    "변경요약: | 기술키워드: | 변경방향: | 파일맥락: ",
    responses={
        200: {
            "description": "커밋 분석 성공.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "COMMON_200",
                        "message": "요청이 성공적으로 처리되었습니다.",
                        "result": {
                            "commit_hash": ("cb2222fb915f9dfbd5b22eded57dadd57f225798"),
                            "commit_id": 4,
                            "summary": (
                                "Swagger 에러 문서화를 위한 "
                                "ApiErrorCodeExample 어노테이션을 추가했습니다."
                            ),
                        },
                    }
                }
            },
        },
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
        commit_hash=request.commit_hash,
        repository_id=request.repository_id,
        message=request.message,
        changed_file_list=filtered_files,
        commit_id=request.commit_id,
    )
    return ok_response(
        CommitAnalyzeResponse(
            commit_hash=request.commit_hash,
            commit_id=request.commit_id,
            summary=summary,
        )
    )


@router.post(
    "/analyze/runs",
    response_model=ApiResponse[CommitAnalyzeRunAccepted],
    status_code=202,
    summary="커밋 분석 비동기 실행 생성",
    description=(
        "커밋 요약과 임베딩 저장을 백그라운드 run으로 실행합니다. "
        "즉시 run_id를 반환하며, 상태는 "
        "GET /api/commit/analyze/runs/{run_id}로 조회합니다.\n\n"
        "기존 /api/commit/analyze는 즉시 요약 응답 후 임베딩을 "
        "BackgroundTasks로 저장하므로, 안정적인 매칭 플로우에서는 "
        "이 비동기 run API 사용을 권장합니다.\n\n"
        f"{COMMIT_ANALYZE_ASYNC_GUIDE}"
    ),
    responses={
        202: {
            "description": "커밋 분석 비동기 작업이 접수되었습니다.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "COMMIT_ANALYZE_202",
                        "message": "커밋 분석 비동기 실행이 접수되었습니다.",
                        "result": {
                            "run_id": "550e8400e29b41d4a716446655440000",
                            "status": "queued",
                            "phase": "queued",
                            "commit_hash": ("cb2222fb915f9dfbd5b22eded57dadd57f225798"),
                            "commit_id": 4,
                            "repository_id": 1,
                        },
                    }
                }
            },
        },
        422: {
            "model": ApiErrorResponse,
            "description": "요청 스키마 검증 실패(예: message 누락, 빈 파일 목록).",
        },
    },
)
async def create_commit_analyze_run(
    request: CommitAnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> ApiResponse[CommitAnalyzeRunAccepted]:
    accepted = await create_commit_analyze_run_record(request)
    background_tasks.add_task(run_commit_analyze_pipeline, accepted.run_id)
    return ok_response(
        accepted,
        code="COMMIT_ANALYZE_202",
        message="커밋 분석 비동기 실행이 접수되었습니다.",
    )


@router.get(
    "/analyze/runs/{run_id}",
    response_model=ApiResponse[CommitAnalyzeRunStatus],
    summary="커밋 분석 비동기 실행 상태 조회",
    description=(
        "run_id 기준으로 커밋 분석 상태를 조회합니다. "
        "phase 값은 queued/summarizing/summary_ready/embedding/"
        "embedding_ready/failed 중 하나입니다.\n\n"
        "Spring 처리 규칙:\n"
        "- phase=summary_ready: 커밋 summary 사용 가능\n"
        "- phase=embedding: ChromaDB 저장 진행 중\n"
        "- status=completed, phase=embedding_ready: /api/commit/match 호출 권장\n"
        "- status=failed: error 기준으로 재시도/장애 처리\n\n"
        f"{COMMIT_ANALYZE_ASYNC_GUIDE}"
    ),
    responses={
        200: {
            "description": "현재 커밋 분석 실행 상태 또는 중간/최종 결과.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "COMMIT_ANALYZE_200",
                        "message": "커밋 분석 비동기 실행 상태 조회에 성공했습니다.",
                        "result": {
                            "run_id": "550e8400e29b41d4a716446655440000",
                            "status": "completed",
                            "phase": "embedding_ready",
                            "commit_hash": ("cb2222fb915f9dfbd5b22eded57dadd57f225798"),
                            "commit_id": 4,
                            "repository_id": 1,
                            "submitted_at": "2026-04-27T09:00:00Z",
                            "started_at": "2026-04-27T09:00:01Z",
                            "finished_at": "2026-04-27T09:00:10Z",
                            "error": None,
                            "result": {
                                "commit_hash": (
                                    "cb2222fb915f9dfbd5b22eded57dadd57f225798"
                                ),
                                "commit_id": 4,
                                "repository_id": 1,
                                "summary": (
                                    "Swagger 에러 문서화를 위한 "
                                    "ApiErrorCodeExample 어노테이션을 추가했습니다."
                                ),
                                "embedding_ready": True,
                            },
                        },
                    }
                }
            },
        },
        404: {
            "model": ApiErrorResponse,
            "description": "해당 run_id를 찾을 수 없습니다(만료/정리 포함).",
        },
    },
)
async def get_commit_analyze_run(
    run_id: str,
) -> ApiResponse[CommitAnalyzeRunStatus]:
    status = await get_commit_analyze_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="해당 실행을 찾을 수 없습니다.")
    return ok_response(
        status,
        code="COMMIT_ANALYZE_200",
        message="커밋 분석 비동기 실행 상태 조회에 성공했습니다.",
    )


@router.post(
    "/match",
    response_model=ApiResponse[ApplicationCommitMatchResponse],
    summary="적용사항-커밋 추천 매칭",
    description=(
        "회의 적용사항(application 단위)에 대해 "
        "커밋 임베딩 후보를 유사도 기반으로 추천합니다.\n\n"
        "점수 정책(100점):\n"
        "- 의미 유사성 50\n"
        "- 기술 키워드 일치도 30\n"
        "- 파일/모듈 맥락 일치도 20\n"
        "- 반대 의미(도입 vs 제거)는 semantic 0 처리\n"
        "- 추상 커밋/모호한 적용사항은 보정 감점\n\n"
        "응답은 적용사항별 추천 커밋을 신뢰도 내림차순으로 반환합니다.\n"
        "사용자 연결 상태는 Spring 서버에서 별도로 관리합니다."
    ),
    responses={
        200: {
            "description": "적용사항별 추천 커밋 매칭 성공.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "COMMIT_MATCH_200",
                        "message": "적용사항-커밋 추천 분석이 완료되었습니다.",
                        "result": {
                            "meeting_id": "meeting-123",
                            "repository_id": 1,
                            "total_applications": 1,
                            "matched_applications": 1,
                            "applications": [
                                {
                                    "application_id": 101,
                                    "application_document_id": (
                                        "meeting-123_application0"
                                    ),
                                    "application_title": (
                                        "Swagger 에러 응답 예시 문서화"
                                    ),
                                    "recommended_commits": [
                                        {
                                            "commit_id": 4,
                                            "commit_ref": None,
                                            "commit_hash": (
                                                "cb2222fb915f9dfbd5b22eded57dadd57f225798"
                                            ),
                                            "commit_message": (
                                                "chore: Swagger 에러 문서화를 위한 "
                                                "ApiErrorCodeExample 어노테이션 추가"
                                            ),
                                            "repository_id": 1,
                                            "confidence": 94,
                                            "reason": (
                                                "총 94점: 의미 44/50, 키워드 30/30, "
                                                "맥락 20/20. 겹친 키워드: swagger, "
                                                "error, api. 겹친 모듈: swagger, api."
                                            ),
                                            "score_breakdown": {
                                                "semantic": 44,
                                                "keyword": 30,
                                                "context": 20,
                                                "penalty": 0,
                                                "total": 94,
                                            },
                                            "direction_primary": "add",
                                            "direction_multi": ["add"],
                                            "tech_keywords": [
                                                "api",
                                                "error",
                                                "swagger",
                                            ],
                                            "module_tags": ["api", "swagger"],
                                        }
                                    ],
                                }
                            ],
                            "notice": "신뢰도는 AI 분석 기반 추정값입니다.",
                        },
                    }
                }
            },
        },
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
async def match_application_commits(
    request: ApplicationCommitMatchRequest,
) -> ApiResponse[ApplicationCommitMatchResponse]:
    result = await match_applications_with_commits(request)
    return ok_response(
        result=result,
        code="COMMIT_MATCH_200",
        message="적용사항-커밋 추천 분석이 완료되었습니다.",
    )
