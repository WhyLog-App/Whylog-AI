import asyncio
import logging
import os

from google import genai
from google.genai import types

from app.core.errors import AppServiceError
from app.domains.commit.schemas import ChangedFile

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
너는 Git 커밋을 분석해서 개발 의도를 자연어로 요약하는 전문가야.
이 요약은 회의 내용과 의미적으로 비교·매칭하는 벡터 DB에 저장돼.

## 절대 원칙
- 커밋 메시지와 diff에 없는 내용을 추측하거나 만들어내지 마.
- 파일명, 메서드명, 클래스명 같은 기술적 식별자는 포함하지 마.
- 기술 구현 방식(어떻게)보다 기능·목적(무엇을, 왜)에 집중해.

## 1단계: 분석
- 커밋 메시지에서 변경의 의도를 파악해.
- diff에서 실제로 무엇이 추가/제거/수정됐는지 확인해.
- 둘을 종합해서 이 커밋이 해결하려는 문제나 구현하려는 기능을 파악해.

## 2단계: 요약
- 회의 발언처럼 읽히는 자연스러운 한국어로 작성해.
- 1~2문장으로 압축해.

## 응답 형식
- 순수 텍스트만 반환해.
- 마크다운, 코드블록, 설명 없이 요약문만 출력해.
""".strip()


async def summarize_commit(
    message: str,
    changed_file_list: list[ChangedFile],
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise AppServiceError("GEMINI_API_KEY가 설정되지 않았습니다.", status_code=500)

    client = genai.Client(api_key=api_key)

    files_text = "\n\n".join(
        f"[{f.file_name}]\n{f.changed_code}" for f in changed_file_list
    )
    user_message = f"커밋 메시지: {message}\n\n변경 파일:\n{files_text}"

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
                config=types.GenerateContentConfig(temperature=0.3),
            ),
            timeout=30.0,
        )
        return response.text.strip()
    except TimeoutError as e:
        logger.error("Gemini 커밋 요약 타임아웃 (30초 초과)")
        raise AppServiceError(
            "Gemini 응답 시간이 초과되었습니다.", status_code=504
        ) from e
    except Exception as e:
        logger.exception("Gemini 커밋 요약 실패")
        raise AppServiceError(
            "커밋 요약 중 오류가 발생했습니다.", status_code=502
        ) from e
