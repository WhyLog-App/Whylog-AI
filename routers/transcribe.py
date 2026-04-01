from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile

from schemas.transcribe import TranscribeSegment
from services import deepgram, transcript_correction
from services.deepgram import CONTENT_TYPE_MAP

router = APIRouter(prefix="/api", tags=["transcribe"])


# POST /api/transcribe — 오디오 파일을 받아 화자별 전사 결과 반환
@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile,
    # 화자 수를 알고 있으면 전달 (정확도 향상, 선택사항) — 1~20 범위만 허용
    num_speakers: int | None = Form(default=None, ge=1, le=20),
) -> list[TranscribeSegment]:
    # 확장자로 MIME 타입 결정 (없으면 파일 자체의 content_type 사용)
    ext = Path(audio.filename or "").suffix.lower().lstrip(".")
    content_type = CONTENT_TYPE_MAP.get(
        ext, audio.content_type or "application/octet-stream"
    )

    # 대용량 파일 메모리 보호 (500MB 초과 차단)
    MAX_FILE_SIZE = 500 * 1024 * 1024
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 500MB를 초과합니다.")

    # 1단계: Deepgram STT + 화자 분리
    segments = await deepgram.transcribe(audio_bytes, content_type, num_speakers)

    # 2단계: Gemini LLM으로 화자 오인식·짧은 발화 등 후처리
    return await transcript_correction.correct_transcript(segments, num_speakers)
