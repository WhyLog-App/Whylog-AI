import json
import logging
import os

from google import genai
from google.genai import types

from app.core.errors import AppServiceError
from app.core.gemini import generate_content_with_retry
from app.domains.transcribe.schemas import TranscribeSegment

logger = logging.getLogger(__name__)

# LLM에게 전달할 후처리 지침
SYSTEM_PROMPT = """
너는 음성 전사 결과를 교정하는 전문가야.
아래 규칙을 순서대로 적용해서 JSON 세그먼트 배열을 교정해줘.

## 절대 원칙
- 원본에 없는 텍스트를 절대 추가하거나 만들어내지 마.
  텍스트 보정은 명백한 오타 수준에만 허용해.
- start_time, end_time은 원본 값을 최대한 유지해.
- 모든 세그먼트의 start_time은 오름차순이어야 해. 시간이 역전되면 안 돼.

## 1단계: 노이즈 제거
- 텍스트가 한 단어 이하이고 맥락상 의미가 없는 세그먼트는 제거해.
  (예: "볼트전", "요거는", "있게.")
- 단, "네", "맞습니다", "아니요" 같은 명확한 응답어는 유지해.

## 2단계: 분리
- 한 세그먼트 안에 질문(?)과 그에 대한 답변이 함께 있으면
  반드시 두 세그먼트로 분리해.
  예: "구현 됐어? 네 됐습니다." → Speaker A: "구현 됐어?" / Speaker B: "네 됐습니다."
- 한 세그먼트에서 문체가 급격히 바뀌거나(존댓말↔반말),
  "맞습니다/네/아니요"처럼 상대 발화에 반응하는 구절이 섞여 있으면 분리해.
- 분리 시 start_time/end_time은 원본 범위 내에서 텍스트 길이 비율로 나눠 할당해.

## 3단계: 병합
- 2단계 분리 이후, 인접한 동일 화자 세그먼트는 하나로 합쳐.
  (사이에 다른 화자가 없어야 함)
- "네", "맞습니다" 같은 짧은 응답어는 병합하지 말고 독립 세그먼트로 유지해.

## 4단계: 화자 재배정
- 화자 수 힌트가 주어지면 그 수의 화자 번호만 사용해.
  (예: 5명이면 Speaker 0~4)
- 화자 번호는 첫 등장 순서 기준으로 0부터 배정해.

## 5단계: 출력
- message_id는 1부터 순서대로 재배정해.
- 반드시 아래 형식의 JSON 배열만 반환해.
  설명, 마크다운, 코드블록 없이 JSON만.

반환 형식:
[
  {
    "message_id": 1,
    "speaker": "Speaker 0",
    "start_time": "00:00:00",
    "end_time": "00:00:05",
    "text": "전사된 텍스트",
    "is_final": true
  },
  ...
]
""".strip()


# Gemini를 사용해 전사 결과의 화자 오인식 및 텍스트 오류를 보정
async def correct_transcript(
    segments: list[dict],
    num_speakers: int | None = None,
) -> list[TranscribeSegment]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise AppServiceError(
            "GEMINI_API_KEY가 설정되지 않았습니다.",
            status_code=500,
        )

    client = genai.Client(api_key=api_key)

    # 화자 수 힌트를 프롬프트에 포함
    speaker_hint = f"\n화자 수: {num_speakers}명" if num_speakers else ""
    segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
    user_message = f"{speaker_hint}\n\n전사 결과:\n{segments_json}"

    def _validate_raw_segments(raw_segments: list[dict]) -> list[TranscribeSegment]:
        # LLM/원본 세그먼트를 최종 응답 스키마로 강제 검증
        validated: list[TranscribeSegment] = []
        for idx, segment in enumerate(raw_segments, start=1):
            try:
                validated.append(TranscribeSegment(**segment))
            except Exception as e:
                raise AppServiceError(
                    f"전사 세그먼트 스키마 검증 실패(index={idx}): {e}",
                    status_code=502,
                ) from e
        return validated

    try:
        # JSON 모드 강제 + temperature 낮춰 일관성 확보
        response = await generate_content_with_retry(
            client,
            contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
            timeout=90.0,
            operation_name="Gemini 전사 후처리",
        )
        parsed = json.loads(response.text)
        return _validate_raw_segments(parsed)
    except AppServiceError:
        # 검증 실패는 후처리 실패와 구분해 상위로 전파한다.
        raise
    except Exception:
        logger.exception("Gemini 후처리 실패로 원본 전사 세그먼트를 반환합니다.")
        # Gemini API 오류(쿼터/인증/타임아웃) 또는 파싱 실패 시 원본 반환
        return _validate_raw_segments(segments)
