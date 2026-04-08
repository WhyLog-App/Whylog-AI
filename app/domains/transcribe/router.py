import logging
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.core.errors import AppServiceError
from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.decision.schemas import DecisionExtractionResult
from app.domains.decision.services.extraction import (
    build_decision_result,
    extract_decision_cards_only,
    extract_decisions,
    extract_overall_analysis,
)
from app.domains.pipeline.schemas import (
    TranscribeDecisionResponse,
    TranscribeDecisionRunAccepted,
    TranscribeDecisionRunStatus,
)
from app.domains.pipeline.services.decision_runs import (
    create_run,
    get_run_status,
    mark_run_completed,
    mark_run_failed,
    mark_run_phase,
    mark_run_processing,
)
from app.domains.transcribe.schemas import TranscribeSegment
from app.domains.transcribe.services import deepgram, transcript_correction
from app.domains.transcribe.services.deepgram import CONTENT_TYPE_MAP

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
MAX_FILE_SIZE = 100 * 1024 * 1024
logger = logging.getLogger(__name__)
AUDIO_FILE_DESCRIPTION = (
    "회의 녹음 파일. 지원 포맷: wav/mp3/m4a/aac/flac/ogg/webm, 최대 100MB."
)
SPRING_ASYNC_GUIDE = (
    "Spring 연동 가이드:\n"
    "1) POST /api/transcribe/decisions/runs 호출로 run_id를 발급받습니다.\n"
    "2) GET /api/transcribe/decisions/runs/{run_id}를 "
    "2~5초 간격으로 폴링합니다.\n"
    "3) phase=transcript_ready 시 transcript_segments를 "
    "회의 요약 화면에 먼저 반영할 수 있습니다.\n"
    "4) phase=summary_ready 시 overall_analysis를 반영해 "
    "요약 화면을 고도화할 수 있습니다.\n"
    "5) status=completed && phase=decisions_ready 시 "
    "decision_cards 포함 최종 결과를 저장/전파합니다.\n"
    "6) status=failed 시 error를 기록하고 필요 시 재시도를 수행합니다.\n"
    "7) run 조회 404는 만료/정리/재기동 유실 가능성이 있으므로 "
    "재요청 정책을 둡니다."
)
SPRING_DECISION_FAQ = (
    "팀 공유 FAQ:\n"
    "- timeline.speaker_id는 null일 수 있습니다. "
    "짧은 응답어/모호 발화에서 오탐을 피하기 위한 설계입니다.\n"
    "- summary_ready 단계의 final_decisions_list는 임시값일 수 있으며, "
    "completed에서 최종 동기화됩니다.\n"
    "- 재추출이 필요하면 /api/decisions/extract에 "
    "저장된 transcript_segments를 전달하세요."
)


def _resolve_content_type(audio: UploadFile) -> str:
    # 파일 확장자/업로드 content-type을 기반으로 MIME 타입 결정
    ext = Path(audio.filename or "").suffix.lower().lstrip(".")
    return CONTENT_TYPE_MAP.get(ext, audio.content_type or "application/octet-stream")


async def _read_audio_bytes(audio: UploadFile) -> bytes:
    # 업로드 파일을 바이트로 읽고 최대 크기 제한을 검증
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 100MB를 초과합니다.")
    return audio_bytes


async def _transcribe_and_correct_from_bytes(
    audio_bytes: bytes,
    content_type: str,
    num_speakers: int | None,
) -> list[TranscribeSegment]:
    # 1단계: Deepgram STT + 화자 분리
    segments = await deepgram.transcribe(audio_bytes, content_type, num_speakers)

    # 2단계: Gemini LLM으로 화자 오인식·짧은 발화 등 후처리
    return await transcript_correction.correct_transcript(segments, num_speakers)


async def _transcribe_and_correct(
    audio: UploadFile,
    num_speakers: int | None,
) -> list[TranscribeSegment]:
    # 업로드 파일을 읽어 STT + 후처리 파이프라인을 실행
    content_type = _resolve_content_type(audio)
    audio_bytes = await _read_audio_bytes(audio)
    return await _transcribe_and_correct_from_bytes(
        audio_bytes=audio_bytes,
        content_type=content_type,
        num_speakers=num_speakers,
    )


async def _run_transcribe_decision_run(
    run_id: str,
    audio_bytes: bytes,
    content_type: str,
    num_speakers: int | None,
    meeting_id: str | None,
    project_id: str | None,
) -> None:
    # 비동기 run의 전체 파이프라인을 단계별(phase)로 실행
    try:
        await mark_run_processing(run_id)
        transcript_segments = await _transcribe_and_correct_from_bytes(
            audio_bytes=audio_bytes,
            content_type=content_type,
            num_speakers=num_speakers,
        )
        partial_result = TranscribeDecisionResponse(
            meeting_id=meeting_id,
            project_id=project_id,
            transcript_segments=transcript_segments,
            decision_result=DecisionExtractionResult(),
        )
        await mark_run_phase(
            run_id=run_id,
            phase="transcript_ready",
            result=partial_result,
        )

        overall_analysis = await extract_overall_analysis(transcript_segments)
        partial_result.decision_result.overall_analysis = overall_analysis
        await mark_run_phase(
            run_id=run_id,
            phase="summary_ready",
            result=partial_result,
        )

        cards_result = await extract_decision_cards_only(transcript_segments)
        partial_result.decision_result = build_decision_result(
            overall_analysis=partial_result.decision_result.overall_analysis,
            cards_result=cards_result,
        )

        await mark_run_completed(
            run_id=run_id,
            result=partial_result,
        )
    except AppServiceError as e:
        await mark_run_failed(run_id, f"{e.status_code}: {e.message}")
    except Exception as e:
        logger.exception(
            "transcribe/decisions run failed",
            extra={"run_id": run_id},
        )
        await mark_run_failed(run_id, f"unexpected_error: {e}")


# POST /api/transcribe — 오디오 파일을 받아 화자별 전사 결과 반환
@router.post(
    "",
    response_model=ApiResponse[list[TranscribeSegment]],
    summary="회의 음성 전사(STT) + 후처리",
    description=(
        "오디오 파일을 업로드하면 Deepgram STT와 Gemini 후처리를 거쳐 "
        "화자 분리 전사 세그먼트를 반환합니다.\n\n"
        "Spring 가이드: 이 엔드포인트는 전사 결과만 필요할 때 사용합니다. "
        "결정사항까지 필요하면 /api/transcribe/decisions(동기) 또는 "
        "/api/transcribe/decisions/runs(비동기)를 사용하세요."
    ),
    responses={
        413: {
            "model": ApiErrorResponse,
            "description": "파일 크기가 100MB를 초과했습니다.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": False,
                        "code": "COMMON_413",
                        "message": "파일 크기가 100MB를 초과합니다.",
                        "result": None,
                    }
                }
            },
        },
        422: {
            "model": ApiErrorResponse,
            "description": "요청 검증 실패(예: num_speakers 범위 오류).",
        },
        500: {
            "model": ApiErrorResponse,
            "description": "서버 설정 오류(예: API 키 누락).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "외부 AI API 연결/응답 오류.",
        },
        504: {
            "model": ApiErrorResponse,
            "description": "외부 STT API 타임아웃.",
        },
    },
)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description=AUDIO_FILE_DESCRIPTION)],
    # 화자 수를 알고 있으면 전달 (정확도 향상, 선택사항) — 1~20 범위만 허용
    num_speakers: int | None = Form(
        default=None,
        ge=1,
        le=20,
        description="화자 수 힌트(선택). 전달 시 화자 분리 정확도 개선에 도움.",
    ),
) -> ApiResponse[list[TranscribeSegment]]:
    # 전사+후처리 결과만 필요한 경우 사용하는 엔드포인트
    segments = await _transcribe_and_correct(audio, num_speakers)
    return ok_response(
        segments,
        code="TRANSCRIBE_200",
        message="전사 및 후처리가 완료되었습니다.",
    )


@router.post(
    "/decisions",
    response_model=ApiResponse[TranscribeDecisionResponse],
    summary="전사 + 후처리 + 의사결정 추출(동기)",
    description=(
        "오디오 업로드 후 STT, 후처리, 의사결정 추출을 한 번에 완료해 "
        "최종 결과를 동기 응답으로 반환합니다.\n\n"
        "Spring 가이드: 긴 회의에서는 타임아웃 위험이 크므로 "
        "운영 환경에서는 /api/transcribe/decisions/runs 비동기 방식을 권장합니다."
    ),
    responses={
        413: {
            "model": ApiErrorResponse,
            "description": "파일 크기가 100MB를 초과했습니다.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": False,
                        "code": "COMMON_413",
                        "message": "파일 크기가 100MB를 초과합니다.",
                        "result": None,
                    }
                }
            },
        },
        422: {
            "model": ApiErrorResponse,
            "description": "요청 검증 실패(예: num_speakers 범위 오류).",
        },
        500: {
            "model": ApiErrorResponse,
            "description": "서버 설정 오류(예: API 키 누락).",
        },
        502: {
            "model": ApiErrorResponse,
            "description": "외부 AI API 연결/응답 오류 또는 JSON 파싱 오류.",
        },
        504: {
            "model": ApiErrorResponse,
            "description": "외부 STT API 타임아웃.",
        },
    },
)
async def transcribe_and_extract_decisions(
    audio: Annotated[UploadFile, File(description=AUDIO_FILE_DESCRIPTION)],
    num_speakers: int | None = Form(
        default=None,
        ge=1,
        le=20,
        description="화자 수 힌트(선택).",
    ),
    meeting_id: str | None = Form(
        default=None,
        description="호출 측이 관리하는 회의 ID(선택).",
    ),
    project_id: str | None = Form(
        default=None,
        description="호출 측이 관리하는 프로젝트 ID(선택).",
    ),
) -> ApiResponse[TranscribeDecisionResponse]:
    # 전사부터 의사결정 추출까지 동기로 한 번에 수행
    transcript_segments = await _transcribe_and_correct(audio, num_speakers)
    decision_result = await extract_decisions(transcript_segments)

    return ok_response(
        TranscribeDecisionResponse(
            meeting_id=meeting_id,
            project_id=project_id,
            transcript_segments=transcript_segments,
            decision_result=decision_result,
        ),
        code="TRANSCRIBE_200",
        message="전사, 후처리, 의사결정 추출이 완료되었습니다.",
    )


@router.post(
    "/decisions/runs",
    response_model=ApiResponse[TranscribeDecisionRunAccepted],
    status_code=202,
    summary="전사 + 의사결정 추출 비동기 실행 생성",
    description=(
        "장시간 작업을 백그라운드로 실행합니다. "
        "즉시 run_id를 반환하며, 상태는 "
        "GET /api/transcribe/decisions/runs/{run_id}로 조회합니다.\n\n"
        f"{SPRING_ASYNC_GUIDE}"
    ),
    responses={
        202: {
            "description": "비동기 작업이 접수되었습니다.",
        },
        413: {
            "model": ApiErrorResponse,
            "description": "파일 크기가 100MB를 초과했습니다.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": False,
                        "code": "COMMON_413",
                        "message": "파일 크기가 100MB를 초과합니다.",
                        "result": None,
                    }
                }
            },
        },
        422: {
            "model": ApiErrorResponse,
            "description": "요청 검증 실패(예: num_speakers 범위 오류).",
        },
    },
)
async def create_transcribe_decision_run(
    background_tasks: BackgroundTasks,
    audio: Annotated[UploadFile, File(description=AUDIO_FILE_DESCRIPTION)],
    num_speakers: int | None = Form(
        default=None,
        ge=1,
        le=20,
        description="화자 수 힌트(선택).",
    ),
    meeting_id: str | None = Form(
        default=None,
        description="호출 측이 관리하는 회의 ID(선택).",
    ),
    project_id: str | None = Form(
        default=None,
        description="호출 측이 관리하는 프로젝트 ID(선택).",
    ),
) -> ApiResponse[TranscribeDecisionRunAccepted]:
    # 장시간 처리를 백그라운드 run으로 접수하고 run_id를 반환
    content_type = _resolve_content_type(audio)
    audio_bytes = await _read_audio_bytes(audio)
    accepted = await create_run(meeting_id=meeting_id, project_id=project_id)
    background_tasks.add_task(
        _run_transcribe_decision_run,
        accepted.run_id,
        audio_bytes,
        content_type,
        num_speakers,
        meeting_id=meeting_id,
        project_id=project_id,
    )
    return ok_response(
        accepted,
        code="TRANSCRIBE_202",
        message="비동기 실행이 접수되었습니다.",
    )


@router.get(
    "/decisions/runs/{run_id}",
    response_model=ApiResponse[TranscribeDecisionRunStatus],
    summary="비동기 실행 상태/중간결과 조회",
    description=(
        "run_id 기준으로 상태를 조회합니다. "
        "phase 값은 queued/transcribing/transcript_ready/"
        "summary_ready/decisions_ready/failed 중 하나입니다.\n\n"
        "Spring 처리 규칙:\n"
        "- phase=transcript_ready: transcript_segments 사용 가능(요약 화면 1차 반영)\n"
        "- phase=summary_ready: decision_result.overall_analysis까지 사용 가능\n"
        "  (주의: final_decisions_list는 completed 시점에 재동기화되어 바뀔 수 있음)\n"
        "- status=completed, phase=decisions_ready: decision_cards 포함 최종 반영\n"
        "- status=failed: error 기준으로 재시도/장애 처리\n\n"
        f"{SPRING_DECISION_FAQ}"
    ),
    responses={
        200: {
            "description": "현재 실행 상태 또는 중간/최종 결과.",
        },
        404: {
            "model": ApiErrorResponse,
            "description": "해당 run_id를 찾을 수 없습니다(만료/정리 포함).",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": False,
                        "code": "COMMON_404",
                        "message": "해당 실행을 찾을 수 없습니다.",
                        "result": None,
                    }
                }
            },
        },
    },
)
async def get_transcribe_decision_run(
    run_id: str,
) -> ApiResponse[TranscribeDecisionRunStatus]:
    # run_id 기준으로 비동기 실행 상태/중간결과/최종결과 조회
    status = await get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="해당 실행을 찾을 수 없습니다.")
    return ok_response(
        status,
        code="TRANSCRIBE_200",
        message="비동기 실행 상태 조회에 성공했습니다.",
    )
