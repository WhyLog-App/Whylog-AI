import os

import httpx
from fastapi import HTTPException

from utils.audio import format_time, merge_consecutive_speaker_segments

# Deepgram STT API 엔드포인트
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

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
        raise HTTPException(
            status_code=500, detail="DEEPGRAM_API_KEY가 설정되지 않았습니다."
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
        raise HTTPException(
            status_code=504,
            detail="Deepgram 요청 타임아웃: 파일이 너무 크거나 서버 응답이 지연됩니다.",
        ) from None
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Deepgram 연결 실패: {e}") from e

    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Deepgram 오류: {response.text[:500]}",
        )

    # 응답에서 발화(utterance) 목록 추출
    utterances = response.json().get("results", {}).get("utterances") or []
    raw_segments = [
        {
            "speaker": utt.get("speaker", 0),
            "start": utt.get("start", 0.0),
            "end": utt.get("end", 0.0),
            "text": utt.get("transcript", ""),
        }
        for utt in utterances
    ]

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
