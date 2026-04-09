import asyncio
import logging
import os

from google import genai
from google.genai import types

from app.core.errors import AppServiceError
from app.domains.commit.schemas import ChangedFile

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
logger = logging.getLogger(__name__)

EMBEDDING_PROMPT = """
너는 Git 커밋을 분석해서 개발 의도를 자연어로 서술하는 전문가야.

## 절대 원칙
- 커밋 메시지와 diff에 없는 내용을 추측하거나 만들어내지 마.
- 파일명, 메서드명, 클래스명 같은 기술적 식별자는 포함하지 마.
- 기술 구현 방식(어떻게)보다 기능·목적(무엇을, 왜)에 집중해.
- 회의 발언처럼 읽히는 자연스러운 한국어로 작성해.

## 분석 방법
- 커밋 메시지에서 변경의 의도를 파악해.
- diff에서 실제로 무엇이 추가/제거/수정됐는지 확인해.
- 둘을 종합해서 이 커밋이 해결하려는 문제나 구현하려는 기능을 파악해.

## 응답 형식
아래 항목을 모두 담아 회의록 발언처럼 서술해. 변경량에 따라 길이는 자연스럽게
조절해도 되지만, 항목은 빠짐없이 포함해. 마크다운, 코드블록, 접두어 없이 본문만
출력해.
- 이 커밋이 해결하는 문제 또는 구현하는 기능
- 변경의 배경과 동기
- 영향받는 도메인과 범위
- 기대 효과
""".strip()

SUMMARY_PROMPT = """
너는 Git 커밋을 분석해서 1~2문장으로 요약하는 전문가야.

## 절대 원칙
- 커밋 메시지와 diff에 없는 내용을 추측하거나 만들어내지 마.
- 파일명, 메서드명, 클래스명 같은 기술적 식별자는 포함하지 마.
- 기술 구현 방식(어떻게)보다 기능·목적(무엇을, 왜)에 집중해.
- 회의 발언처럼 읽히는 자연스러운 한국어로 작성해.

## 응답 형식
마크다운, 코드블록, 접두어, 추가 설명 없이 핵심 변경 내용만 1문장으로 출력해.
""".strip()


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise AppServiceError("GEMINI_API_KEY가 설정되지 않았습니다.", status_code=500)
    return genai.Client(api_key=api_key)


def _build_commit_input(message: str, changed_file_list: list[ChangedFile]) -> str:
    files_text = "\n\n".join(
        f"[{f.file_name}]\n{f.changed_code}" for f in changed_file_list
    )
    return f"커밋 메시지: {message}\n\n변경 파일:\n{files_text}"


async def _call_gemini(
    client: genai.Client, prompt: str, user_message: str, timeout: float
) -> str:
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{prompt}\n\n{user_message}",
            config=types.GenerateContentConfig(temperature=0.3),
        ),
        timeout=timeout,
    )
    return response.text.strip()


async def summarize_commit(
    message: str,
    changed_file_list: list[ChangedFile],
) -> str:
    client = _get_client()
    commit_input = _build_commit_input(message, changed_file_list)

    try:
        summary = await _call_gemini(client, SUMMARY_PROMPT, commit_input, timeout=15.0)
        if not summary:
            raise ValueError("LLM 응답이 비어 있습니다.")
        return summary
    except ValueError as e:
        logger.error("Gemini 응답 파싱 실패: %s", e)
        raise AppServiceError(
            "커밋 요약 응답을 파싱할 수 없습니다.", status_code=502
        ) from e
    except TimeoutError as e:
        logger.error("Gemini 커밋 요약 타임아웃")
        raise AppServiceError(
            "Gemini 응답 시간이 초과되었습니다.", status_code=504
        ) from e
    except AppServiceError:
        raise
    except Exception as e:
        logger.exception("Gemini 커밋 요약 실패")
        raise AppServiceError(
            "커밋 요약 중 오류가 발생했습니다.", status_code=502
        ) from e


async def generate_embedding_text(
    commit_id: int,
    message: str,
    changed_file_list: list[ChangedFile],
) -> None:
    """백그라운드에서 임베딩용 상세 텍스트 생성. 응답을 블로킹하지 않음."""
    try:
        client = _get_client()
        commit_input = _build_commit_input(message, changed_file_list)
        embedding_text = await _call_gemini(
            client, EMBEDDING_PROMPT, commit_input, timeout=60.0
        )
        embedding_text = " ".join(embedding_text.split())
        # TODO: ChromaDB에 commit_id + embedding_text 저장
        logger.info("커밋 %d 임베딩 텍스트 생성 완료: %s", commit_id, embedding_text)
    except Exception:
        logger.exception("커밋 %d 임베딩 텍스트 생성 실패", commit_id)
