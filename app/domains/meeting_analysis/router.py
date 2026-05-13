import logging
from typing import Annotated

from fastapi import APIRouter, Body

from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.meeting_analysis.schemas import (
    ApplicationEmbeddingRequest,
    ApplicationEmbeddingResponse,
    MeetingAnalysisRequest,
    MeetingAnalysisResponse,
)
from app.domains.meeting_analysis.services.embedding import embed_and_store_applications
from app.domains.meeting_analysis.services.extraction import extract_meeting_analysis

router = APIRouter(prefix="/meeting-analysis", tags=["meeting-analysis"])
logger = logging.getLogger(__name__)


@router.post(
    "/extract",
    response_model=ApiResponse[MeetingAnalysisResponse],
    summary="전사 세그먼트 기반 회의 분석 재추출",
    description=(
        "저장된 transcript_segments(JSON)만으로 "
        "회의 분석 결과와 적용사항 목록을 추출합니다. "
        "오디오 재업로드 없이 프롬프트 변경 실험, "
        "실패 재시도, 운영 재처리에 사용합니다.\n\n"
        "팀 공유 FAQ:\n"
        "- timeline.member_id가 null일 수 있습니다(오탐 방지 목적).\n"
        "- summary_ready 단계 값과 completed 단계 값은 일부 필드가 다를 수 있습니다.\n"
        "- 이 API가 생성하는 applications[].application_id는 보통 null입니다. "
        "Spring이 적용사항을 DB에 저장한 뒤 발급한 applicationId를 "
        "/api/meeting-analysis/embeddings 요청의 application_id로 전달해야 합니다.\n"
        "- 이 API는 completed 결과 재생성/프롬프트 재실험 용도로 권장합니다."
    ),
    responses={
        422: {
            "model": ApiErrorResponse,
            "description": "요청 스키마 검증 실패(예: transcript_segments 누락).",
        },
        500: {
            "model": ApiErrorResponse,
            "description": "서버 설정 오류(예: GEMINI_API_KEY 누락).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "Gemini 연결/응답/JSON 파싱 오류.",
        },
        200: {
            "description": "회의 분석 재추출 성공.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "MEETING_ANALYSIS_200",
                        "message": "회의 분석 재추출이 완료되었습니다.",
                        "result": {
                            "meeting_id": "meeting-123",
                            "project_id": "project-abc",
                            "analysis_result": {
                                "overall_analysis": {
                                    "meeting_info": {
                                        "title": "API 문서화 개선 회의",
                                        "purpose": (
                                            "Swagger 에러 응답 예시 개선 방향 합의"
                                        ),
                                        "duration": "00:12:30",
                                    },
                                    "topics": ["Swagger 에러 응답 문서화"],
                                    "core_context": [
                                        "프론트에서 에러 응답 형식을 예측하기 어렵다."
                                    ],
                                    "application_titles": [
                                        "Swagger 에러 응답 예시 문서화"
                                    ],
                                    "application_reasons": [
                                        (
                                            "API 사용자가 에러 응답 형식을 "
                                            "쉽게 확인해야 한다."
                                        )
                                    ],
                                },
                                "applications": [
                                    {
                                        "application_id": None,
                                        "application_title": (
                                            "Swagger 에러 응답 예시 문서화"
                                        ),
                                        "application_reasons": [
                                            (
                                                "API 사용자가 에러 응답 형식을 "
                                                "쉽게 확인해야 한다."
                                            )
                                        ],
                                        "timeline": [
                                            {
                                                "timestamp": "00:03:12",
                                                "step": "이슈제기",
                                                "member_id": 1,
                                                "content": (
                                                    "Swagger에서 에러 응답 예시가 "
                                                    "부족하다는 문제가 제기됨"
                                                ),
                                                "utterance": (
                                                    "Swagger에 에러 응답 예시가 "
                                                    "잘 안 보여요."
                                                ),
                                            },
                                            {
                                                "timestamp": "00:06:20",
                                                "step": "대안논의",
                                                "member_id": 2,
                                                "content": (
                                                    "어노테이션 기반 예시 "
                                                    "문서화가 논의됨"
                                                ),
                                                "utterance": (
                                                    "어노테이션으로 예시를 붙이면 "
                                                    "관리하기 좋겠습니다."
                                                ),
                                            },
                                            {
                                                "timestamp": "00:09:40",
                                                "step": "적용합의",
                                                "member_id": 1,
                                                "content": (
                                                    "ApiErrorCodeExample 어노테이션을 "
                                                    "추가하기로 합의함"
                                                ),
                                                "utterance": (
                                                    "그럼 ApiErrorCodeExample을 "
                                                    "추가하는 걸로 하죠."
                                                ),
                                            },
                                        ],
                                    }
                                ],
                                "other_mentions": [],
                            },
                        },
                    }
                }
            },
        },
    },
)
async def extract_meeting_analysis_from_transcript(
    payload: Annotated[
        MeetingAnalysisRequest,
        Body(
            description=(
                "Spring 연동 가이드:\n"
                "- 본 API는 오디오 업로드 없이, 이미 저장된 transcript_segments로 "
                "회의 분석 결과와 적용사항을 재추출/재시도할 때 사용합니다.\n"
                "- 요청 본문은 반드시 객체형(JSON object)이며 "
                "transcript_segments 필드를 포함해야 합니다.\n"
                "- meeting_id/project_id는 선택이며, "
                "전달 시 응답에 그대로 포함됩니다.\n"
                "- 권장 입력은 /api/transcribe 계열 응답의 "
                "transcript_segments 원본입니다."
            ),
            examples={
                "object_with_metadata": {
                    "summary": "권장: 메타데이터 + STT 세그먼트",
                    "value": {
                        "meeting_id": "meeting-123",
                        "project_id": "project-abc",
                        "transcript_segments": [
                            {
                                "message_id": 1,
                                "speaker": "김준용",
                                "member_id": 1,
                                "start_time": "00:00:00",
                                "end_time": "00:00:03",
                                "text": (
                                    "이번 주에는 배포 방식을 단순화하기로 합의했습니다."
                                ),
                                "is_final": True,
                            }
                        ],
                    },
                },
            },
        ),
    ],
) -> ApiResponse[MeetingAnalysisResponse]:
    # 전달받은 전사 세그먼트로 회의 분석 결과를 재추출
    analysis_result = await extract_meeting_analysis(payload.transcript_segments)

    return ok_response(
        MeetingAnalysisResponse(
            meeting_id=payload.meeting_id,
            project_id=payload.project_id,
            analysis_result=analysis_result,
        ),
        code="MEETING_ANALYSIS_200",
        message="회의 분석 재추출이 완료되었습니다.",
    )


@router.post(
    "/embeddings",
    response_model=ApiResponse[ApplicationEmbeddingResponse],
    summary="적용사항 임베딩 생성 및 ChromaDB 저장",
    description=(
        "회의 분석 결과(applications)를 application 단위로 정규화한 뒤, "
        "Gemini Embedding API로 벡터를 생성하고 ChromaDB에 저장합니다.\n\n"
        "Spring 연동 순서:\n"
        "1) /api/transcribe/applications/runs 또는 /api/meeting-analysis/extract로 "
        "application_id가 없는 적용사항 목록을 생성합니다.\n"
        "2) Spring이 applications를 DB에 저장하고 applicationId를 발급합니다.\n"
        "3) 이 API를 호출할 때 각 applications[].application_id에 "
        "Spring applicationId를 넣어 전달합니다.\n\n"
        "- 동일 meeting_id로 재호출 시 기존 문서를 삭제 후 새로 저장합니다.\n"
        "- 문서 ID 형식: `{meeting_id}_application{i}`\n"
        "- 임베딩 텍스트: `title: 제목 | text: 적용사항: 제목 | 근거: 핵심근거`"
    ),
    responses={
        422: {
            "model": ApiErrorResponse,
            "description": "요청 스키마 검증 실패.",
        },
        500: {
            "model": ApiErrorResponse,
            "description": "서버 설정 오류(예: GEMINI_API_KEY 누락).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "Gemini 임베딩 또는 ChromaDB 저장 오류.",
        },
    },
)
async def create_application_embeddings(
    payload: Annotated[
        ApplicationEmbeddingRequest,
        Body(
            examples={
                "spring_application_id": {
                    "summary": "권장: Spring applicationId 포함",
                    "value": {
                        "meeting_id": "meeting-123",
                        "project_id": "project-abc",
                        "analysis_result": {
                            "applications": [
                                {
                                    "application_id": 101,
                                    "application_title": (
                                        "Swagger 에러 응답 예시 문서화"
                                    ),
                                    "application_reasons": [
                                        (
                                            "API 사용자가 에러 응답 형식을 "
                                            "쉽게 확인해야 한다."
                                        ),
                                    ],
                                    "timeline": [
                                        {
                                            "timestamp": "00:09:40",
                                            "step": "적용합의",
                                            "member_id": 1,
                                            "content": (
                                                "ApiErrorCodeExample 어노테이션을 "
                                                "추가하기로 합의함"
                                            ),
                                            "utterance": (
                                                "그럼 ApiErrorCodeExample을 "
                                                "추가하는 걸로 하죠."
                                            ),
                                        }
                                    ],
                                }
                            ],
                            "other_mentions": [],
                        },
                    },
                },
            },
        ),
    ],
) -> ApiResponse[ApplicationEmbeddingResponse]:
    logger.info(
        "임베딩 요청 수신 meeting_id=%s application_count=%d application_ids=%s",
        payload.meeting_id,
        len(payload.analysis_result.applications),
        [a.application_id for a in payload.analysis_result.applications],
    )
    documents = await embed_and_store_applications(
        meeting_id=payload.meeting_id,
        project_id=payload.project_id,
        analysis_result=payload.analysis_result,
    )

    return ok_response(
        ApplicationEmbeddingResponse(
            meeting_id=payload.meeting_id,
            project_id=payload.project_id,
            total_documents=len(documents),
            document_ids=[doc.document_id for doc in documents],
            documents=documents,
        ),
        code="APPLICATION_EMBEDDING_200",
        message="적용사항 임베딩이 생성 및 저장되었습니다.",
    )
