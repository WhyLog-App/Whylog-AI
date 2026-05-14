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

from app.core.enums import TranscribeRunPhase
from app.core.errors import AppServiceError
from app.core.responses import ApiErrorResponse, ApiResponse, ok_response
from app.domains.meeting_analysis.schemas import MeetingAnalysisResult
from app.domains.meeting_analysis.services.extraction import (
    build_analysis_result,
    extract_applications_only,
    extract_meeting_analysis,
    extract_overall_analysis,
)
from app.domains.pipeline.schemas import (
    TranscribeAnalysisResponse,
    TranscribeAnalysisRunAccepted,
    TranscribeAnalysisRunStatus,
)
from app.domains.pipeline.services.analysis_runs import (
    create_run,
    get_run_status,
    mark_run_completed,
    mark_run_failed,
    mark_run_phase,
    mark_run_processing,
)
from app.domains.transcribe.schemas import LiveTranscriptMessage, TranscribeSegment
from app.domains.transcribe.services import deepgram, transcript_correction
from app.domains.transcribe.services.deepgram import CONTENT_TYPE_MAP
from app.domains.transcribe.services.live_transcript_merge import (
    merge_live_transcript,
    parse_live_messages,
)

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
MAX_FILE_SIZE = 100 * 1024 * 1024
logger = logging.getLogger(__name__)
AUDIO_FILE_DESCRIPTION = (
    "회의 녹음 파일. 지원 포맷: wav/mp3/m4a/aac/flac/ogg/webm, 최대 100MB."
)
LIVE_MESSAGES_GUIDE = (
    "실시간 발화 로그 연동:\n"
    "- live_messages는 Spring WebSocket TEXT 발화 로그 JSON 배열 문자열입니다.\n"
    "- 각 항목은 type, meetingId, fromMemberId, fromName, "
    "timestamp, text를 포함합니다.\n"
    "- FastAPI는 Deepgram STT 결과와 live_messages를 "
    "시간/순서/텍스트 유사도로 매칭합니다.\n"
    "- 매칭 성공 시 transcript_segments[].speaker는 실제 이름으로 보강되고 "
    "member_id가 함께 반환됩니다.\n"
    "- 매칭 신뢰도가 충분하면 transcript_segments[].text도 "
    "WebSocket text 기준으로 보정합니다.\n"
    "- 매칭 실패 시 기존 Speaker N/STT text를 유지합니다."
)
LIVE_MESSAGES_FIELD_DESCRIPTION = (
    "WebSocket 실시간 발화 로그 JSON 배열 문자열(선택). "
    "Spring WebSocket TEXT 메시지를 아래 형식 그대로 배열로 전달합니다.\n\n"
    "예시:\n"
    "[\n"
    "  {\n"
    '    "type": "TEXT",\n'
    '    "meetingId": 123,\n'
    '    "fromMemberId": 1,\n'
    '    "fromName": "김준용",\n'
    '    "timestamp": "00:01:48",\n'
    '    "targetMemberId": 2,\n'
    '    "text": "안녕하세요",\n'
    '    "payload": {}\n'
    "  }\n"
    "]\n\n"
    "필수 사용 필드: type, meetingId, fromMemberId, fromName, timestamp, text. "
    "targetMemberId와 payload는 선택입니다. "
    'timestamp는 가능하면 "00:01:48"처럼 회의 기준 발화 시작 시간으로 전달하세요.'
)
SPRING_ASYNC_GUIDE = (
    "Spring 연동 가이드:\n"
    "1) POST /api/transcribe/applications/runs 호출로 run_id를 발급받습니다.\n"
    "2) GET /api/transcribe/applications/runs/{run_id}를 "
    "2~5초 간격으로 폴링합니다.\n"
    "3) phase=transcript_ready 시 transcript_segments를 "
    "회의 요약 화면에 먼저 반영할 수 있습니다.\n"
    "4) phase=summary_ready 시 overall_analysis를 반영해 "
    "요약 화면을 고도화할 수 있습니다.\n"
    "5) status=completed && phase=applications_ready 시 "
    "applications 포함 최종 결과를 저장/전파합니다.\n"
    "6) 최종 applications[].application_id는 null이 정상입니다. "
    "Spring이 적용사항 저장 후 발급한 applicationId를 "
    "/api/meeting-analysis/embeddings 요청의 application_id로 다시 전달합니다.\n"
    "7) status=failed 시 error를 기록하고 필요 시 재시도를 수행합니다.\n"
    "8) run 조회 404는 만료/정리/재기동 유실 가능성이 있으므로 "
    "재요청 정책을 둡니다."
)
SPRING_APPLICATION_FAQ = (
    "팀 공유 FAQ:\n"
    "- timeline.member_id는 null일 수 있습니다. "
    "짧은 응답어/모호 발화에서 오탐을 피하기 위한 설계입니다.\n"
    "- summary_ready 단계의 application_titles는 임시값일 수 있으며, "
    "completed에서 최종 동기화됩니다.\n"
    "- 재추출이 필요하면 /api/meeting-analysis/extract에 "
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
    live_messages: list[LiveTranscriptMessage] | None = None,
) -> list[TranscribeSegment]:
    # 1단계: Deepgram STT + 화자 분리
    segments = await deepgram.transcribe(audio_bytes, content_type, num_speakers)

    # 2단계: Gemini LLM으로 화자 오인식·짧은 발화 등 후처리
    corrected_segments = await transcript_correction.correct_transcript(
        segments, num_speakers
    )

    # 3단계: WebSocket 실시간 발화 로그 기준으로 최종 전사 보강
    return merge_live_transcript(corrected_segments, live_messages or [])


async def _transcribe_and_correct(
    audio: UploadFile,
    num_speakers: int | None,
    live_messages: list[LiveTranscriptMessage] | None = None,
) -> list[TranscribeSegment]:
    # 업로드 파일을 읽어 STT + 후처리 파이프라인을 실행
    content_type = _resolve_content_type(audio)
    audio_bytes = await _read_audio_bytes(audio)
    return await _transcribe_and_correct_from_bytes(
        audio_bytes=audio_bytes,
        content_type=content_type,
        num_speakers=num_speakers,
        live_messages=live_messages,
    )


async def _run_transcribe_application_run(
    run_id: str,
    audio_bytes: bytes,
    content_type: str,
    num_speakers: int | None,
    meeting_id: str | None,
    project_id: str | None,
    live_messages: list[LiveTranscriptMessage] | None = None,
) -> None:
    # 비동기 run의 전체 파이프라인을 단계별(phase)로 실행
    try:
        await mark_run_processing(run_id)
        transcript_segments = await _transcribe_and_correct_from_bytes(
            audio_bytes=audio_bytes,
            content_type=content_type,
            num_speakers=num_speakers,
            live_messages=live_messages,
        )
        partial_result = TranscribeAnalysisResponse(
            meeting_id=meeting_id,
            project_id=project_id,
            transcript_segments=transcript_segments,
            analysis_result=MeetingAnalysisResult(),
        )
        await mark_run_phase(
            run_id=run_id,
            phase=TranscribeRunPhase.TRANSCRIPT_READY,
            result=partial_result,
        )

        overall_analysis = await extract_overall_analysis(transcript_segments)
        partial_result.analysis_result.overall_analysis = overall_analysis
        await mark_run_phase(
            run_id=run_id,
            phase=TranscribeRunPhase.SUMMARY_READY,
            result=partial_result,
        )

        applications_result = await extract_applications_only(transcript_segments)
        partial_result.analysis_result = build_analysis_result(
            overall_analysis=partial_result.analysis_result.overall_analysis,
            applications_result=applications_result,
        )

        await mark_run_completed(
            run_id=run_id,
            result=partial_result,
        )
    except AppServiceError as e:
        await mark_run_failed(run_id, f"{e.status_code}: {e.message}")
    except Exception as e:
        logger.exception(
            "transcribe/applications run failed",
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
        "적용사항까지 필요하면 /api/transcribe/applications(동기) 또는 "
        "/api/transcribe/applications/runs(비동기)를 사용하세요.\n\n"
        f"{LIVE_MESSAGES_GUIDE}"
    ),
    responses={
        200: {
            "description": "전사, 후처리, 회의 분석/적용사항 추출 성공.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "TRANSCRIBE_200",
                        "message": (
                            "전사, 후처리, 회의 분석/적용사항 추출이 완료되었습니다."
                        ),
                        "result": {
                            "meeting_id": "meeting-123",
                            "project_id": "project-abc",
                            "transcript_segments": [
                                {
                                    "message_id": 1,
                                    "speaker": "Speaker 0",
                                    "member_id": None,
                                    "start_time": "00:00:00",
                                    "end_time": "00:00:04",
                                    "text": "Swagger 에러 응답 예시가 부족합니다.",
                                    "is_final": True,
                                }
                            ],
                            "analysis_result": {
                                "overall_analysis": {
                                    "meeting_info": {
                                        "title": "API 문서화 개선 회의",
                                        "purpose": "Swagger 에러 응답 예시 개선",
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
                    }
                }
            },
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
    live_messages: str | None = Form(
        default=None,
        description=LIVE_MESSAGES_FIELD_DESCRIPTION,
    ),
) -> ApiResponse[list[TranscribeSegment]]:
    # 전사+후처리 결과만 필요한 경우 사용하는 엔드포인트
    parsed_live_messages = parse_live_messages(live_messages)
    segments = await _transcribe_and_correct(
        audio,
        num_speakers,
        live_messages=parsed_live_messages,
    )
    return ok_response(
        segments,
        code="TRANSCRIBE_200",
        message="전사 및 후처리가 완료되었습니다.",
    )


@router.post(
    "/applications",
    response_model=ApiResponse[TranscribeAnalysisResponse],
    summary="전사 + 후처리 + 회의 분석/적용사항 추출(동기)",
    description=(
        "오디오 업로드 후 STT, 후처리, 회의 분석/적용사항 추출을 한 번에 완료해 "
        "최종 결과를 동기 응답으로 반환합니다.\n\n"
        "Spring 가이드: 긴 회의에서는 타임아웃 위험이 크므로 "
        "운영 환경에서는 /api/transcribe/applications/runs "
        "비동기 방식을 권장합니다.\n\n"
        f"{LIVE_MESSAGES_GUIDE}"
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
async def transcribe_and_extract_applications(
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
    live_messages: str | None = Form(
        default=None,
        description=LIVE_MESSAGES_FIELD_DESCRIPTION,
    ),
) -> ApiResponse[TranscribeAnalysisResponse]:
    # 전사부터 회의 분석/적용사항 추출까지 동기로 한 번에 수행
    parsed_live_messages = parse_live_messages(live_messages)
    transcript_segments = await _transcribe_and_correct(
        audio,
        num_speakers,
        live_messages=parsed_live_messages,
    )
    analysis_result = await extract_meeting_analysis(transcript_segments)

    return ok_response(
        TranscribeAnalysisResponse(
            meeting_id=meeting_id,
            project_id=project_id,
            transcript_segments=transcript_segments,
            analysis_result=analysis_result,
        ),
        code="TRANSCRIBE_200",
        message="전사, 후처리, 회의 분석/적용사항 추출이 완료되었습니다.",
    )


@router.post(
    "/applications/runs",
    response_model=ApiResponse[TranscribeAnalysisRunAccepted],
    status_code=202,
    summary="전사 + 회의 분석/적용사항 추출 비동기 실행 생성",
    description=(
        "장시간 작업을 백그라운드로 실행합니다. "
        "즉시 run_id를 반환하며, 상태는 "
        "GET /api/transcribe/applications/runs/{run_id}로 조회합니다.\n\n"
        f"{LIVE_MESSAGES_GUIDE}\n\n"
        f"{SPRING_ASYNC_GUIDE}"
    ),
    responses={
        202: {
            "description": "비동기 작업이 접수되었습니다.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "TRANSCRIBE_202",
                        "message": "비동기 실행이 접수되었습니다.",
                        "result": {
                            "run_id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "queued",
                            "phase": "queued",
                            "meeting_id": "meeting-123",
                            "project_id": "project-abc",
                        },
                    }
                }
            },
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
async def create_transcribe_application_run(
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
    live_messages: str | None = Form(
        default=None,
        description=LIVE_MESSAGES_FIELD_DESCRIPTION,
    ),
) -> ApiResponse[TranscribeAnalysisRunAccepted]:
    # 장시간 처리를 백그라운드 run으로 접수하고 run_id를 반환
    content_type = _resolve_content_type(audio)
    audio_bytes = await _read_audio_bytes(audio)
    parsed_live_messages = parse_live_messages(live_messages)
    accepted = await create_run(meeting_id=meeting_id, project_id=project_id)
    background_tasks.add_task(
        _run_transcribe_application_run,
        accepted.run_id,
        audio_bytes,
        content_type,
        num_speakers,
        meeting_id=meeting_id,
        project_id=project_id,
        live_messages=parsed_live_messages,
    )
    return ok_response(
        accepted,
        code="TRANSCRIBE_202",
        message="비동기 실행이 접수되었습니다.",
    )


@router.get(
    "/applications/runs/{run_id}",
    response_model=ApiResponse[TranscribeAnalysisRunStatus],
    summary="비동기 실행 상태/중간결과 조회",
    description=(
        "run_id 기준으로 상태를 조회합니다. "
        "phase 값은 queued/transcribing/transcript_ready/"
        "summary_ready/applications_ready/failed 중 하나입니다.\n\n"
        "Spring 처리 규칙:\n"
        "- phase=transcript_ready: transcript_segments 사용 가능(요약 화면 1차 반영)\n"
        "- phase=summary_ready: analysis_result.overall_analysis까지 사용 가능\n"
        "  (주의: application_titles는 completed 시점에 재동기화되어 바뀔 수 있음)\n"
        "- status=completed, phase=applications_ready: applications 포함 최종 반영\n"
        "- status=failed: error 기준으로 재시도/장애 처리\n\n"
        f"{SPRING_APPLICATION_FAQ}"
    ),
    responses={
        200: {
            "description": "현재 실행 상태 또는 중간/최종 결과.",
            "content": {
                "application/json": {
                    "example": {
                        "isSuccess": True,
                        "code": "TRANSCRIBE_200",
                        "message": "비동기 실행 상태 조회에 성공했습니다.",
                        "result": {
                            "run_id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "completed",
                            "phase": "applications_ready",
                            "meeting_id": "meeting-123",
                            "project_id": "project-abc",
                            "submitted_at": "2026-04-27T09:00:00Z",
                            "started_at": "2026-04-27T09:00:02Z",
                            "finished_at": "2026-04-27T09:00:40Z",
                            "error": None,
                            "result": {
                                "meeting_id": "meeting-123",
                                "project_id": "project-abc",
                                "transcript_segments": [
                                    {
                                        "message_id": 1,
                                        "speaker": "Speaker 0",
                                        "member_id": None,
                                        "start_time": "00:00:00",
                                        "end_time": "00:00:04",
                                        "text": (
                                            "Swagger 에러 응답 예시가 부족합니다."
                                        ),
                                        "is_final": True,
                                    }
                                ],
                                "analysis_result": {
                                    "overall_analysis": {
                                        "meeting_info": {
                                            "title": "API 문서화 개선 회의",
                                            "purpose": ("Swagger 에러 응답 예시 개선"),
                                            "duration": "00:12:30",
                                        },
                                        "topics": ["Swagger 에러 응답 문서화"],
                                        "core_context": [
                                            (
                                                "프론트에서 에러 응답 형식을 "
                                                "예측하기 어렵다."
                                            )
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
                                                        "ApiErrorCodeExample "
                                                        "어노테이션을 "
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
                        },
                    }
                }
            },
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
async def get_transcribe_application_run(
    run_id: str,
) -> ApiResponse[TranscribeAnalysisRunStatus]:
    # run_id 기준으로 비동기 실행 상태/중간결과/최종결과 조회
    status = await get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="해당 실행을 찾을 수 없습니다.")
    return ok_response(
        status,
        code="TRANSCRIBE_200",
        message="비동기 실행 상태 조회에 성공했습니다.",
    )
