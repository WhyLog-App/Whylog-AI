import asyncio
import logging
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.chroma import get_commit_collection
from app.core.config import settings
from app.core.errors import AppServiceError
from app.core.gemini import RETRY_STATUS_CODES, generate_content_with_retry
from app.domains.commit.schemas import ChangedFile

RETRY_BACKOFFS = (1.0, 2.0)  # 1차 실패 후 1초, 2차 실패 후 2초 대기
logger = logging.getLogger(__name__)

EMBEDDING_PROMPT = """
너는 Git 커밋의 diff를 분석해서 구조화된 임베딩용 텍스트를 생성하는 전문가야.

## 절대 원칙
- 커밋 메시지와 diff에 없는 내용을 추측하거나 만들어내지 마.
- 코드 변경 사실만 기술해. 의도나 동기를 추론하지 마.

## 응답 형식 (정확히 이 형식으로만 출력해)
변경요약: (코드 변경 사실을 1~2문장으로)
기술키워드: (DB, 프레임워크, 라이브러리, 모듈 등 기술 요소만 쉼표 구분)
변경방향: (이 커밋의 기능 방향을
  add/remove/modify/migrate 중 해당하는 것을 쉼표 구분으로 모두 선택)
파일맥락: (변경된 파일 경로에서 비즈니스 도메인 토큰만 쉼표 구분)

## 주의
- 위 4줄만 출력해. 다른 설명이나 마크다운, 코드블록을 추가하지 마.
- 기술키워드에 일반 단어(함수, 파일, 코드 등)는 넣지 마. 구체적 기술명만 넣어.
- 변경방향은 코드 라인 단위가 아니라 커밋 전체의 기능적 목적 기준으로 판단해.
  예) 새 API 추가 → add, 버그 수정 → modify, 기능 삭제 → remove,
  구조 이전 → migrate, 기능 추가+기존 수정 → add,modify
- 파일맥락은 프로젝트 구조 디렉토리는 제외하고
  (controller, service, domain, components, pages 등)
  auth, meeting, billing 같은 비즈니스 도메인만 추출해.
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


VALID_DIRECTIONS = {"add", "remove", "modify", "migrate"}


@dataclass
class ParsedEmbedding:
    """LLM 응답을 파싱한 구조화 결과."""

    summary: str
    tech_keywords: list[str]
    directions: list[str]
    module_tags: list[str]


def _parse_embedding_response(text: str) -> ParsedEmbedding:
    """LLM 구조화 응답에서 변경요약/기술키워드/변경방향/파일맥락을 추출한다."""
    summary = ""
    tech_keywords: list[str] = []
    directions: list[str] = []
    module_tags: list[str] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("변경요약:"):
            summary = line.removeprefix("변경요약:").strip()
        elif line.startswith("기술키워드:"):
            raw = line.removeprefix("기술키워드:").strip()
            tech_keywords = [k.strip() for k in raw.split(",") if k.strip()]
        elif line.startswith("변경방향:"):
            raw = line.removeprefix("변경방향:").strip().lower()
            directions = [
                d.strip() for d in raw.split(",") if d.strip() in VALID_DIRECTIONS
            ]
        elif line.startswith("파일맥락:"):
            raw = line.removeprefix("파일맥락:").strip()
            module_tags = [t.strip() for t in raw.split(",") if t.strip()]

    if not summary:
        raise ValueError("LLM 응답에서 변경요약을 추출할 수 없습니다.")

    return ParsedEmbedding(
        summary=summary,
        tech_keywords=tech_keywords,
        directions=directions,
        module_tags=module_tags,
    )


def _get_client() -> genai.Client:
    api_key = settings.gemini_api_key
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
    response = await generate_content_with_retry(
        client,
        contents=f"{prompt}\n\n{user_message}",
        config=types.GenerateContentConfig(temperature=0.3),
        timeout=timeout,
        operation_name="Gemini 커밋 분석",
        backoffs=RETRY_BACKOFFS,
    )
    return response.text.strip()


async def _generate_embedding(text: str, timeout: float = 30.0) -> list[float]:
    """Gemini Embedding API로 단일 텍스트의 임베딩 벡터를 생성한다."""
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        try:
            response = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model=settings.embedding_model,
                    contents=text,
                ),
                timeout=timeout,
            )
            if not response.embeddings:
                raise ValueError("Gemini 임베딩 응답이 비어 있습니다.")
            return response.embeddings[0].values
        except genai_errors.APIError as e:
            if e.code not in RETRY_STATUS_CODES or attempt >= len(RETRY_BACKOFFS):
                raise AppServiceError(
                    f"Gemini 임베딩 생성 실패: {e}",
                    status_code=502,
                ) from e
            backoff = RETRY_BACKOFFS[attempt]
            logger.warning(
                "Gemini 임베딩 %s 응답, %.1f초 후 재시도 (%d/%d)",
                e.code,
                backoff,
                attempt + 1,
                len(RETRY_BACKOFFS),
            )
            last_error = e
            await asyncio.sleep(backoff)
    raise last_error  # type: ignore[misc]


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


async def store_commit_embedding(
    commit_hash: str,
    repository_id: int,
    message: str,
    changed_file_list: list[ChangedFile],
    commit_id: int | None = None,
) -> None:
    """커밋 임베딩용 구조화 텍스트를 생성하고 ChromaDB에 저장한다."""
    client = _get_client()
    commit_input = _build_commit_input(message, changed_file_list)

    # 1) LLM으로 구조화 텍스트 생성
    raw_text = await _call_gemini(client, EMBEDDING_PROMPT, commit_input, timeout=60.0)
    parsed = _parse_embedding_response(raw_text)

    # 2) 스펙에 맞는 임베딩용 텍스트 조합
    commit_subject = message.split("\n", 1)[0].strip()
    title = f"repository-{repository_id} {commit_subject}"
    text_parts = [f"변경요약: {parsed.summary}"]
    if parsed.tech_keywords:
        text_parts.append(f"기술키워드: {','.join(parsed.tech_keywords)}")
    if parsed.directions:
        text_parts.append(f"변경방향: {','.join(parsed.directions)}")
    if parsed.module_tags:
        text_parts.append(f"파일맥락: {','.join(parsed.module_tags)}")
    embedding_text = f"title: {title} | text: {' | '.join(text_parts)}"

    # 3) 임베딩 벡터 생성
    embedding = await _generate_embedding(embedding_text)

    # 4) ChromaDB 저장 (doc_id는 hash 기반, commit_id는 보조 메타로 저장)
    collection = get_commit_collection()
    doc_id = f"commit_{commit_hash}"
    metadata: dict[str, str | int] = {
        "commit_hash": commit_hash,
        "commit_message": commit_subject,
        "repository_id": repository_id,
        "direction": ",".join(parsed.directions),
        "tech_keywords_csv": ",".join(parsed.tech_keywords),
        "module_tags_csv": ",".join(parsed.module_tags),
    }
    if commit_id is not None:
        metadata["commit_id"] = commit_id
    await asyncio.to_thread(
        collection.upsert,
        ids=[doc_id],
        documents=[embedding_text],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    logger.info(
        "커밋 임베딩 저장 완료: doc_id=%s commit_id=%s",
        doc_id,
        commit_id,
    )


async def generate_embedding_text(
    commit_hash: str,
    repository_id: int,
    message: str,
    changed_file_list: list[ChangedFile],
    commit_id: int | None = None,
) -> None:
    """백그라운드에서 임베딩용 구조화 텍스트 생성. 응답을 블로킹하지 않음."""
    try:
        await store_commit_embedding(
            commit_hash=commit_hash,
            repository_id=repository_id,
            message=message,
            changed_file_list=changed_file_list,
            commit_id=commit_id,
        )
    except Exception:
        logger.exception(
            "커밋 임베딩 텍스트 생성 실패: commit_hash=%s",
            commit_hash,
        )
