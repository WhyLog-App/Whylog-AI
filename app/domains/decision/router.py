from typing import Annotated

from fastapi import APIRouter, Body

from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.decision.schemas import (
    DecisionEmbeddingRequest,
    DecisionEmbeddingResponse,
    DecisionExtractRequest,
    DecisionExtractResponse,
)
from app.domains.decision.services.embedding import embed_and_store_decisions
from app.domains.decision.services.extraction import extract_decisions

router = APIRouter(prefix="/decisions", tags=["decision"])


@router.post(
    "/extract",
    response_model=ApiResponse[DecisionExtractResponse],
    summary="전사 세그먼트 기반 의사결정 재추출",
    description=(
        "저장된 transcript_segments(JSON)만으로 "
        "의사결정 결과를 추출합니다. "
        "오디오 재업로드 없이 프롬프트 변경 실험, "
        "실패 재시도, 운영 재처리에 사용합니다.\n\n"
        "팀 공유 FAQ:\n"
        "- timeline.speaker_id가 null일 수 있습니다(오탐 방지 목적).\n"
        "- summary_ready 단계 값과 completed 단계 값은 일부 필드가 다를 수 있습니다.\n"
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
    },
)
async def extract_decisions_from_transcript(
    payload: Annotated[
        DecisionExtractRequest,
        Body(
            description=(
                "Spring 연동 가이드:\n"
                "- 본 API는 오디오 업로드 없이, 이미 저장된 transcript_segments로 "
                "의사결정을 재추출/재시도할 때 사용합니다.\n"
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
                                "speaker": "Speaker 0",
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
) -> ApiResponse[DecisionExtractResponse]:
    # 전달받은 전사 세그먼트로 의사결정 결과를 재추출
    decision_result = await extract_decisions(payload.transcript_segments)

    return ok_response(
        DecisionExtractResponse(
            meeting_id=payload.meeting_id,
            project_id=payload.project_id,
            decision_result=decision_result,
        ),
        code="DECISION_200",
        message="의사결정 재추출이 완료되었습니다.",
    )


@router.post(
    "/embeddings",
    response_model=ApiResponse[DecisionEmbeddingResponse],
    summary="결정사항 임베딩 생성 및 ChromaDB 저장",
    description=(
        "의사결정 추출 결과(decision_cards)를 applied_item 단위로 정규화한 뒤, "
        "Gemini Embedding API로 벡터를 생성하고 ChromaDB에 저장합니다.\n\n"
        "- 동일 meeting_id로 재호출 시 기존 문서를 삭제 후 새로 저장합니다.\n"
        "- 문서 ID 형식: `{meeting_id}_card{i}_item{j}`\n"
        "- 임베딩 텍스트: `title: 제목 | text: 적용사항: 항목 | 근거: 핵심근거`"
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
async def create_decision_embeddings(
    payload: Annotated[
        DecisionEmbeddingRequest,
        Body(
            examples={
                "minimal": {
                    "summary": "최소 요청 예시",
                    "value": {
                        "meeting_id": "meeting-123",
                        "decision_result": {
                            "decision_cards": [
                                {
                                    "decision_title": "Redis 캐시 도입",
                                    "applied_items": [
                                        "사용자 세션 캐싱 적용",
                                        "API 응답 캐싱 적용",
                                    ],
                                    "decision_reasons": [
                                        "DB 부하를 줄여 응답 속도를 개선한다.",
                                    ],
                                    "timeline": [],
                                }
                            ],
                            "other_mentions": [],
                        },
                    },
                },
            },
        ),
    ],
) -> ApiResponse[DecisionEmbeddingResponse]:
    documents = await embed_and_store_decisions(
        meeting_id=payload.meeting_id,
        project_id=payload.project_id,
        decision_result=payload.decision_result,
    )

    return ok_response(
        DecisionEmbeddingResponse(
            meeting_id=payload.meeting_id,
            project_id=payload.project_id,
            total_documents=len(documents),
            document_ids=[doc.document_id for doc in documents],
            documents=documents,
        ),
        code="DECISION_EMBEDDING_200",
        message="결정사항 임베딩이 생성 및 저장되었습니다.",
    )
