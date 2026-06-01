import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.chroma import get_commit_collection
from app.core.config import settings
from app.core.enums import CommitChangeDirection
from app.core.errors import AppServiceError
from app.core.gemini import RETRY_STATUS_CODES, generate_content_with_retry
from app.domains.commit.schemas import ChangedFile

RETRY_BACKOFFS = (1.0, 2.0)  # 1차 실패 후 1초, 2차 실패 후 2초 대기
logger = logging.getLogger(__name__)

COMMIT_ANALYSIS_PROMPT = """
너는 Git 커밋의 diff를 분석해서 사용자 표시용 요약과
임베딩용 구조화 정보를 동시에 생성하는 전문가야.

## 절대 원칙
- 커밋 메시지와 diff에 없는 내용을 추측하거나 만들어내지 마.
- 코드 변경 사실만 기술해. 의도나 동기를 추론하지 마.
- 요약은 사용자에게 보여줄 문장이고, 변경요약은 검색/임베딩에 사용할 기술 맥락이야.

## 응답 형식 (정확히 이 형식으로만 출력해)
요약: (사용자에게 보여줄 핵심 변경 내용 1문장)
변경요약: (코드 변경 사실을 1~2문장으로)
기술키워드: (DB, 프레임워크, 라이브러리, 모듈 등 기술 요소만 쉼표 구분)
변경방향: (이 커밋의 기능 방향을
  add/remove/modify/migrate 중 해당하는 것을 쉼표 구분으로 모두 선택)
파일맥락: (변경된 파일 경로에서 비즈니스 도메인 토큰만 쉼표 구분)

## 주의
- 위 5줄만 출력해. 다른 설명이나 마크다운, 코드블록을 추가하지 마.
- 요약에는 파일명, 메서드명, 클래스명 같은 기술적 식별자를 포함하지 마.
- 요약은 기술 구현 방식(어떻게)보다 기능·목적(무엇을, 왜)에 집중해.
- 요약은 회의 발언처럼 읽히는 자연스러운 한국어로 작성해.
- 기술키워드에 일반 단어(함수, 파일, 코드 등)는 넣지 마. 구체적 기술명만 넣어.
- 변경방향은 코드 라인 단위가 아니라 커밋 전체의 기능적 목적 기준으로 판단해.
  예) 새 API 추가 → add, 버그 수정 → modify, 기능 삭제 → remove,
  구조 이전 → migrate, 기능 추가+기존 수정 → add,modify
- 파일맥락은 프로젝트 구조 디렉토리는 제외하고
  (controller, service, domain, components, pages 등)
  auth, meeting, billing 같은 비즈니스 도메인만 추출해.
""".strip()


VALID_DIRECTIONS = {direction.value for direction in CommitChangeDirection}
PATH_TOKEN_STOPWORDS = {
    "api",
    "app",
    "abstract",
    "base",
    "build",
    "common",
    "com",
    "component",
    "components",
    "config",
    "constant",
    "constants",
    "controller",
    "core",
    "domain",
    "domains",
    "dto",
    "entity",
    "factory",
    "gradle",
    "helper",
    "impl",
    "java",
    "kotlin",
    "main",
    "manager",
    "mapper",
    "model",
    "org",
    "page",
    "pages",
    "provider",
    "repository",
    "resources",
    "service",
    "settings",
    "src",
    "support",
    "test",
    "tests",
    "util",
    "utils",
    "vo",
    "whylog",
}
PATH_DIRECTORY_DENYLIST = {
    ".git",
    "build",
    "dist",
    "generated",
    "gen",
    "node_modules",
    "out",
    "target",
}
PATH_TOKEN_MIN_LENGTH = 2
MODULE_TAG_MAX_COUNT = 20


@dataclass
class ParsedEmbedding:
    """LLM 응답을 파싱한 구조화 결과."""

    summary: str
    tech_keywords: list[str]
    directions: list[str]
    module_tags: list[str]


@dataclass
class CommitAnalysis:
    """커밋 요약 응답과 임베딩 저장 재료를 함께 담은 분석 결과."""

    summary: str
    embedding: ParsedEmbedding


def _parse_embedding_response(text: str) -> ParsedEmbedding:
    """LLM 구조화 응답에서 변경요약/기술키워드/변경방향/파일맥락을 추출한다."""
    return _parse_commit_analysis_response(text).embedding


def _parse_commit_analysis_response(text: str) -> CommitAnalysis:
    """LLM 통합 분석 응답에서 사용자 요약과 임베딩 재료를 추출한다."""
    display_summary = ""
    summary = ""
    tech_keywords: list[str] = []
    directions: list[str] = []
    module_tags: list[str] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("요약:"):
            display_summary = line.removeprefix("요약:").strip()
        elif line.startswith("변경요약:"):
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

    if not summary and display_summary:
        summary = display_summary
    if not display_summary and summary:
        display_summary = summary
    if not summary:
        raise ValueError("LLM 응답에서 변경요약을 추출할 수 없습니다.")

    return CommitAnalysis(
        summary=display_summary,
        embedding=ParsedEmbedding(
            summary=summary,
            tech_keywords=tech_keywords,
            directions=directions,
            module_tags=module_tags,
        ),
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


def _split_path_token(value: str) -> set[str]:
    raw_parts = re.split(r"[^A-Za-z0-9가-힣]+", value)
    split_parts: set[str] = set()
    for part in raw_parts:
        if not part:
            continue
        split_parts.update(
            re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+|[가-힣]+",
                part,
            )
            or [part]
        )
    return {part.lower() for part in split_parts if part}


def _extract_path_module_tokens(changed_file_list: list[ChangedFile]) -> list[str]:
    tokens: set[str] = set()
    for changed_file in sorted(changed_file_list, key=lambda f: f.file_name):
        path = PurePosixPath(changed_file.file_name)
        path_parts = [part.lower() for part in path.parts]
        if any(part in PATH_DIRECTORY_DENYLIST for part in path_parts):
            continue
        for part in path.parts:
            if part in {"/", ".", ".."}:
                continue
            stem = PurePosixPath(part).stem
            for token in _split_path_token(stem):
                if len(token) < PATH_TOKEN_MIN_LENGTH:
                    continue
                if token in PATH_TOKEN_STOPWORDS:
                    continue
                tokens.add(token)
    return sorted(tokens)


def _merge_unique_tokens(
    *token_groups: list[str], limit: int = MODULE_TAG_MAX_COUNT
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in token_groups:
        for token in group:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
            if len(merged) >= limit:
                return merged
    return merged


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


async def analyze_commit_content(
    message: str,
    changed_file_list: list[ChangedFile],
) -> CommitAnalysis:
    client = _get_client()
    commit_input = _build_commit_input(message, changed_file_list)

    try:
        raw_text = await _call_gemini(
            client,
            COMMIT_ANALYSIS_PROMPT,
            commit_input,
            timeout=60.0,
        )
        analysis = _parse_commit_analysis_response(raw_text)
        if not analysis.summary:
            raise ValueError("LLM 응답이 비어 있습니다.")
        return analysis
    except ValueError as e:
        logger.error("Gemini 응답 파싱 실패: %s", e)
        raise AppServiceError(
            "커밋 분석 응답을 파싱할 수 없습니다.", status_code=502
        ) from e
    except TimeoutError as e:
        logger.error("Gemini 커밋 분석 타임아웃")
        raise AppServiceError(
            "Gemini 응답 시간이 초과되었습니다.", status_code=504
        ) from e
    except AppServiceError:
        raise
    except Exception as e:
        logger.exception("Gemini 커밋 분석 실패")
        raise AppServiceError(
            "커밋 분석 중 오류가 발생했습니다.", status_code=502
        ) from e


async def summarize_commit(
    message: str,
    changed_file_list: list[ChangedFile],
) -> str:
    return (await analyze_commit_content(message, changed_file_list)).summary


async def store_commit_embedding(
    commit_hash: str,
    repository_id: int,
    message: str,
    changed_file_list: list[ChangedFile],
    commit_id: int | None = None,
    analysis: CommitAnalysis | None = None,
) -> None:
    """커밋 임베딩용 구조화 텍스트를 생성하고 ChromaDB에 저장한다."""
    # 1) 커밋 분석 결과 확보.
    # 호출자가 이미 분석한 경우 같은 diff를 LLM에 다시 보내지 않는다.
    if analysis is None:
        analysis = await analyze_commit_content(message, changed_file_list)
    parsed = analysis.embedding
    path_module_tokens = _extract_path_module_tokens(changed_file_list)
    module_tags = _merge_unique_tokens(parsed.module_tags, path_module_tokens)

    # 2) 스펙에 맞는 임베딩용 텍스트 조합
    commit_subject = message.split("\n", 1)[0].strip()
    title = f"repository-{repository_id} {commit_subject}"
    text_parts = [f"변경요약: {parsed.summary}"]
    if parsed.tech_keywords:
        text_parts.append(f"기술키워드: {','.join(parsed.tech_keywords)}")
    if parsed.directions:
        text_parts.append(f"변경방향: {','.join(parsed.directions)}")
    if module_tags:
        text_parts.append(f"파일맥락: {','.join(module_tags)}")
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
        "module_tags_csv": ",".join(module_tags),
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
    analysis: CommitAnalysis | None = None,
) -> None:
    """백그라운드에서 임베딩용 구조화 텍스트 생성. 응답을 블로킹하지 않음."""
    try:
        await store_commit_embedding(
            commit_hash=commit_hash,
            repository_id=repository_id,
            message=message,
            changed_file_list=changed_file_list,
            commit_id=commit_id,
            analysis=analysis,
        )
    except Exception:
        logger.exception(
            "커밋 임베딩 텍스트 생성 실패: commit_hash=%s",
            commit_hash,
        )
