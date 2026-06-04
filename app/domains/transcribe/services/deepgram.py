import logging
import os

import httpx

from app.core.errors import AppServiceError
from app.domains.transcribe.services.audio import (
    format_time,
    merge_consecutive_speaker_segments,
)

# Deepgram STT API 엔드포인트
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
logger = logging.getLogger(__name__)

# 오디오 파일 확장자 → MIME 타입 매핑
CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
}


# 오디오 바이트를 Deepgram에 전송하고 화자별 전사 결과를 반환
async def transcribe(
    audio_bytes: bytes, content_type: str, num_speakers: int | None = None
) -> list[dict]:
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise AppServiceError(
            "DEEPGRAM_API_KEY가 설정되지 않았습니다.",
            status_code=500,
        )

    # Deepgram 요청 옵션: 한국어, 화자 분리, 문장 부호 자동 삽입
    params = {
        "model": "nova-3",  # nova-3: nova-2 대비 한국어 정확도 향상
        "language": "ko",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "true",  # 화자 분리 활성화
        "utterances": "true",  # 발화 단위로 결과 반환
        "filler_words": "false",  # 음, 어 같은 필러 단어 제거
    }

    # 화자 수를 알고 있으면 Deepgram에 힌트로 전달해 정확도 향상
    if num_speakers:
        params["diarize_min_speakers"] = str(num_speakers)
        params["diarize_max_speakers"] = str(num_speakers)
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type,
    }

    try:
        async with httpx.AsyncClient(
            timeout=300
        ) as client:  # 1시간 이상 음원 대비 여유있게 설정
            response = await client.post(
                DEEPGRAM_URL, params=params, headers=headers, content=audio_bytes
            )
    except httpx.TimeoutException:
        raise AppServiceError(
            "Deepgram 요청 타임아웃: 파일이 너무 크거나 서버 응답이 지연됩니다.",
            status_code=504,
        ) from None
    except httpx.RequestError as e:
        raise AppServiceError(
            f"Deepgram 연결 실패: {e}",
            status_code=502,
        ) from e

    if response.is_error:
        raise AppServiceError(
            f"Deepgram 오류({response.status_code}): {response.text[:500]}",
            status_code=502,
        )

    response_payload = response.json()
    diagnostics = _response_diagnostics(response_payload)
    logger.info(
        "Deepgram transcript response: audio_bytes=%s content_type=%s "
        "utterances=%s channels=%s transcript_chars=%s words=%s",
        len(audio_bytes),
        content_type,
        diagnostics["utterance_count"],
        diagnostics["channel_count"],
        diagnostics["transcript_chars"],
        diagnostics["word_count"],
    )
    if diagnostics["utterance_count"] == 0:
        logger.warning(
            "Deepgram utterances empty: transcript_chars=%s words=%s",
            diagnostics["transcript_chars"],
            diagnostics["word_count"],
        )
        logger.warning("Deepgram raw response when utterances empty: %s", response.text)

    raw_segments = _extract_raw_segments(response_payload)

    # 연속된 같은 화자 발화 병합 후 최종 응답 포맷으로 변환
    merged = merge_consecutive_speaker_segments(raw_segments)

    return [
        {
            "message_id": i + 1,
            "speaker": f"Speaker {seg['speaker']}",
            "start_time": format_time(seg["start"]),
            "end_time": format_time(seg["end"]),
            "text": seg["text"],
            "is_final": True,
        }
        for i, seg in enumerate(merged)
    ]


def _extract_raw_segments(response_payload: dict) -> list[dict]:
    results = response_payload.get("results") or {}
    utterances = results.get("utterances") or []
    raw_segments = [
        {
            "speaker": utt.get("speaker", 0),
            "start": utt.get("start", 0.0),
            "end": utt.get("end", 0.0),
            "text": (utt.get("transcript") or "").strip(),
        }
        for utt in utterances
        if (utt.get("transcript") or "").strip()
    ]
    if raw_segments:
        return raw_segments

    alternative = _primary_alternative(response_payload)
    words = alternative.get("words") or []
    word_segments = _segments_from_words(words)
    if word_segments:
        return word_segments

    transcript = (alternative.get("transcript") or "").strip()
    if transcript:
        return [
            {
                "speaker": 0,
                "start": 0.0,
                "end": 0.0,
                "text": transcript,
            }
        ]
    return []


def _segments_from_words(words: list[dict]) -> list[dict]:
    segments: list[dict] = []
    current: dict | None = None
    for word in words:
        text = _word_text(word)
        if not text:
            continue
        speaker = word.get("speaker")
        if speaker is None:
            speaker = 0
        start = _float_or_zero(word.get("start"))
        end = _float_or_zero(word.get("end"))
        if current is None or current["speaker"] != speaker:
            current = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
            segments.append(current)
            continue
        current["end"] = end
        current["text"] = f"{current['text']} {text}".strip()
    return segments


def _word_text(word: dict) -> str:
    return str(word.get("punctuated_word") or word.get("word") or "").strip()


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _response_diagnostics(response_payload: dict) -> dict[str, int]:
    results = response_payload.get("results") or {}
    utterances = results.get("utterances") or []
    channels = results.get("channels") or []
    alternative = _primary_alternative(response_payload)
    transcript = alternative.get("transcript") or ""
    words = alternative.get("words") or []
    return {
        "utterance_count": len(utterances),
        "channel_count": len(channels),
        "transcript_chars": len(transcript),
        "word_count": len(words),
    }


def _primary_alternative(response_payload: dict) -> dict:
    results = response_payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return {}
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return {}
    return alternatives[0]
