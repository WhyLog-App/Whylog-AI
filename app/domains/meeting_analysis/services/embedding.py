import asyncio
import logging

from google import genai

from app.core.chroma import get_application_collection
from app.core.config import settings
from app.core.errors import AppServiceError
from app.domains.meeting_analysis.schemas import (
    Application,
    EmbeddedDocument,
    MeetingAnalysisResult,
)

logger = logging.getLogger(__name__)
_meeting_locks: dict[str, asyncio.Lock] = {}
_meeting_locks_guard = asyncio.Lock()
TIMELINE_CONTEXT_MAX_LENGTH = 400
TIMELINE_AGREEMENT_STEP = "적용합의"
TIMELINE_DISCUSSION_STEP = "대안논의"


async def _get_meeting_lock(meeting_id: str) -> asyncio.Lock:
    """meeting_id 단위 직렬화를 위한 잠금을 가져온다."""
    async with _meeting_locks_guard:
        # 같은 meeting_id 요청은 같은 잠금을 공유해 delete/add 순서를 보장한다.
        lock = _meeting_locks.get(meeting_id)
        if lock is None:
            lock = asyncio.Lock()
            _meeting_locks[meeting_id] = lock
        return lock


# ── 텍스트 정규화 ──


def _normalize_text(value: str) -> str:
    """공백 정규화 및 양쪽 공백 제거."""
    return " ".join((value or "").split())


def _build_reason_text(application_reasons: list[str]) -> tuple[str, int]:
    """중복을 제거한 근거 문장을 전체 반영 텍스트로 생성한다."""
    normalized_reasons: list[str] = []
    seen: set[str] = set()
    for reason in application_reasons:
        normalized = _normalize_text(reason)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_reasons.append(normalized)

    total_count = len(normalized_reasons)
    if total_count == 0:
        return "", 0

    return " | ".join(normalized_reasons), total_count


def _append_with_budget(
    values: list[str],
    item: str,
    used_length: int,
    max_length: int,
) -> int:
    if used_length >= max_length:
        return used_length
    remaining = max_length - used_length
    trimmed = item[:remaining]
    if not trimmed:
        return used_length
    values.append(trimmed)
    return used_length + len(trimmed)


def _build_timeline_context(application: Application) -> tuple[str, str]:
    """적용사항 timeline에서 임베딩에 쓸 결론/논의 맥락을 추출한다."""
    agreements: list[str] = []
    discussions: list[str] = []
    seen: set[str] = set()
    used_length = 0

    for target_step, target_values in (
        (TIMELINE_AGREEMENT_STEP, agreements),
        (TIMELINE_DISCUSSION_STEP, discussions),
    ):
        for item in application.timeline:
            if item.step != target_step:
                continue
            content = _normalize_text(item.content)
            if not content or content in seen:
                continue
            seen.add(content)
            used_length = _append_with_budget(
                target_values,
                content,
                used_length,
                TIMELINE_CONTEXT_MAX_LENGTH,
            )
            if used_length >= TIMELINE_CONTEXT_MAX_LENGTH:
                break
        if used_length >= TIMELINE_CONTEXT_MAX_LENGTH:
            break

    return " | ".join(agreements), " | ".join(discussions)


def build_embedding_documents(
    meeting_id: str,
    analysis_result: MeetingAnalysisResult,
) -> list[EmbeddedDocument]:
    """회의 분석 결과에서 application 단위 임베딩 문서를 생성한다.

    문서 ID: {meeting_id}_application{i}
    텍스트: "title: {title} | text: 적용사항: {title} | 근거: {근거들}
    | 결론: {적용합의 content} | 논의: {대안논의 content}"
    """
    documents: list[EmbeddedDocument] = []
    total_reasons = 0

    # 적용사항 자체가 커밋 매칭의 최소 단위 문서다.
    for application_idx, application in enumerate(analysis_result.applications):
        title = _normalize_text(application.application_title)
        reason_text, reason_total = _build_reason_text(application.application_reasons)
        agreement_text, discussion_text = _build_timeline_context(application)
        total_reasons += reason_total
        doc_id = f"{meeting_id}_application{application_idx}"
        text = f"title: {title or 'none'} | text: 적용사항: {title or 'none'}"
        if reason_text:
            text += f" | 근거: {reason_text}"
        if agreement_text:
            text += f" | 결론: {agreement_text}"
        if discussion_text:
            text += f" | 논의: {discussion_text}"
        documents.append(
            EmbeddedDocument(
                document_id=doc_id,
                text=text,
                application_id=application.application_id,
                application_title=title,
            )
        )

    logger.info(
        "Application reasons appended (meeting_id=%s, reason_count=%d)",
        meeting_id,
        total_reasons,
    )
    return documents


# ── Gemini 임베딩 생성 ──


async def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Gemini Embedding API를 배치 호출하여 벡터 리스트를 반환한다."""
    api_key = settings.gemini_api_key
    if not api_key:
        raise AppServiceError(
            "GEMINI_API_KEY가 설정되지 않았습니다.",
            status_code=500,
        )

    client = genai.Client(api_key=api_key)
    try:
        # 네트워크 왕복 비용을 줄이기 위해 텍스트를 배치로 임베딩한다.
        response = await client.aio.models.embed_content(
            model=settings.embedding_model,
            contents=texts,
        )
    except Exception as e:
        raise AppServiceError(
            f"Gemini 임베딩 생성 실패: {e}",
            status_code=502,
        ) from e

    return [emb.values for emb in response.embeddings]


# ── ChromaDB 저장 ──


async def embed_and_store_applications(
    meeting_id: str,
    project_id: str | None,
    analysis_result: MeetingAnalysisResult,
) -> list[EmbeddedDocument]:
    """적용사항을 정규화 → 임베딩 → ChromaDB 저장한다.

    동일 meeting_id 재처리 시 기존 문서를 삭제 후 새로 추가한다.
    """
    documents = build_embedding_documents(meeting_id, analysis_result)
    texts = [doc.text for doc in documents]
    embeddings = await _generate_embeddings(texts) if texts else []
    missing_application_id_count = sum(
        1 for doc in documents if doc.application_id is None
    )
    if missing_application_id_count:
        logger.warning(
            "Application IDs missing for %d embedding documents (meeting_id=%s)",
            missing_application_id_count,
            meeting_id,
        )

    collection = get_application_collection()
    lock = await _get_meeting_lock(meeting_id)

    # 동일 meeting_id 요청이 동시에 들어올 때 delete/add 순서를 직렬화한다.
    async with lock:
        try:
            # 재처리 시 좀비 문서를 막기 위해 기존 문서를 먼저 조회한다.
            existing = await asyncio.to_thread(
                collection.get,
                where={"meeting_id": meeting_id},
            )
        except Exception as e:
            logger.warning(
                "기존 임베딩 문서 조회 실패(meeting_id=%s)",
                meeting_id,
                exc_info=True,
            )
            raise AppServiceError(
                "기존 임베딩 문서 조회에 실패했습니다.",
                status_code=502,
            ) from e

        existing_ids = existing.get("ids", [])
        if existing_ids:
            try:
                # 기존 결과를 제거한 뒤 최신 결과만 다시 적재한다.
                await asyncio.to_thread(collection.delete, ids=existing_ids)
                logger.info(
                    "Deleted %d existing documents for meeting_id=%s",
                    len(existing_ids),
                    meeting_id,
                )
            except Exception as e:
                raise AppServiceError(
                    f"기존 임베딩 문서 삭제 실패: {e}",
                    status_code=502,
                ) from e

        if not documents:
            return []

        ids = [doc.document_id for doc in documents]
        metadatas = [
            {
                "meeting_id": meeting_id,
                "project_id": project_id or "",
                "application_id": (
                    doc.application_id if doc.application_id is not None else ""
                ),
                "application_title": doc.application_title,
            }
            for doc in documents
        ]

        try:
            # Chroma 동기 API는 스레드로 위임해 이벤트 루프 블로킹을 줄인다.
            await asyncio.to_thread(
                collection.add,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            raise AppServiceError(
                f"ChromaDB 저장 실패: {e}",
                status_code=502,
            ) from e

        logger.info(
            "Stored %d embedding documents for meeting_id=%s",
            len(documents),
            meeting_id,
        )
        return documents
