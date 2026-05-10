import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.core.chroma import get_application_collection, get_commit_collection
from app.core.errors import AppServiceError
from app.domains.commit.schemas import (
    ApplicationCommitMatchItem,
    ApplicationCommitMatchRequest,
    ApplicationCommitMatchResponse,
    MatchedCommit,
    MatchScoreBreakdown,
)
from app.domains.meeting_analysis.services.matching_scoring import (
    ScoringInput,
    calculate_match_score,
    extract_direction_labels_from_text,
    extract_module_tokens,
    extract_tech_keywords,
    is_opposite_direction,
    normalize_direction_labels,
    parse_csv_tokens,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationEntry:
    document_id: str
    text: str
    embedding: list[float] | None
    application_id: int | None
    application_title: str
    direction_labels: set[str]
    keywords: set[str]
    modules: set[str]


@dataclass(frozen=True)
class MatchRecord:
    commit_key: str
    application_index: int
    application_direction_labels: set[str]
    commit: MatchedCommit


def _first_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _first_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _csv_to_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [token for token in (part.strip() for part in value.split(",")) if token]


def _format_tokens(tokens: set[str], *, limit: int = 3) -> str:
    if not tokens:
        return "없음"
    sorted_tokens = sorted(tokens)
    shown = sorted_tokens[:limit]
    suffix = (
        "" if len(sorted_tokens) <= limit else f" 외 {len(sorted_tokens) - limit}개"
    )
    return ", ".join(shown) + suffix


def _build_recommendation_reason(
    *,
    score: Any,
    application_keywords: set[str],
    commit_keywords: set[str],
    application_modules: set[str],
    commit_modules: set[str],
) -> str:
    keyword_overlap = application_keywords & commit_keywords
    module_overlap = application_modules & commit_modules

    parts = [
        (
            f"총 {score.total}점: 의미 {score.semantic}/50, "
            f"키워드 {score.keyword}/30, 맥락 {score.context}/20"
        )
    ]
    if score.penalty:
        parts.append(f"보정 감점 {score.penalty}점")
    parts.append(f"겹친 키워드: {_format_tokens(keyword_overlap)}")
    parts.append(f"겹친 모듈: {_format_tokens(module_overlap)}")
    return ". ".join(parts) + "."


def _to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _build_commit_key(
    commit_ref: str | None,
    commit_hash: str | None,
    fallback_id: str,
) -> str:
    if commit_hash:
        return f"hash:{commit_hash}"
    if commit_ref:
        return f"ref:{commit_ref}"
    return f"id:{fallback_id}"


def _to_application_entries(raw: dict[str, Any]) -> list[ApplicationEntry]:
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    embeddings = raw.get("embeddings")
    if embeddings is None:
        embeddings = []

    entries: list[ApplicationEntry] = []
    for idx, document_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        text = documents[idx] if idx < len(documents) and documents[idx] else ""
        embedding = embeddings[idx] if idx < len(embeddings) else None

        application_title = _first_str(metadata.get("application_title")) or ""
        application_id = _first_int(metadata.get("application_id"))
        full_text = f"{application_title} {text}".strip()

        direction_labels = normalize_direction_labels(
            _first_str(metadata.get("direction_primary")),
            _first_str(metadata.get("direction_multi_csv")),
            _first_str(metadata.get("direction")),
        )
        if not direction_labels:
            direction_labels = extract_direction_labels_from_text(full_text)

        entries.append(
            ApplicationEntry(
                document_id=document_id,
                text=text,
                embedding=_to_float_list(embedding),
                application_id=application_id,
                application_title=application_title,
                direction_labels=direction_labels,
                keywords=extract_tech_keywords(full_text),
                modules=extract_module_tokens(full_text),
            )
        )
    return entries


def _normalize_distance(distance: float | None) -> float | None:
    if distance is None:
        return None
    # Chroma cosine distance를 0~1 구간으로 정규화한다.
    if distance < 0:
        return 0.0
    if distance > 1:
        return 1.0
    return distance


def _resolve_conflicting_matches(records: list[MatchRecord]) -> list[MatchRecord]:
    grouped: dict[str, list[MatchRecord]] = {}
    for record in records:
        grouped.setdefault(record.commit_key, []).append(record)

    kept_ids: set[int] = set()
    for group in grouped.values():
        sorted_group = sorted(
            group,
            key=lambda record: record.commit.score_breakdown.total,
            reverse=True,
        )

        kept_for_commit: list[MatchRecord] = []
        for candidate in sorted_group:
            if any(
                is_opposite_direction(
                    candidate.application_direction_labels,
                    kept.application_direction_labels,
                )
                for kept in kept_for_commit
            ):
                continue
            kept_for_commit.append(candidate)

        kept_ids.update(id(record) for record in kept_for_commit)

    return [record for record in records if id(record) in kept_ids]


async def _load_application_entries(meeting_id: str) -> list[ApplicationEntry]:
    collection = get_application_collection()
    try:
        raw = await asyncio.to_thread(
            collection.get,
            where={"meeting_id": meeting_id},
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception as e:
        raise AppServiceError(
            "적용사항 임베딩 조회에 실패했습니다.",
            status_code=502,
        ) from e

    return _to_application_entries(raw)


async def _query_commit_candidates(
    query_embedding: list[float],
    *,
    n_results: int,
    repository_ids: list[int],
) -> dict[str, Any]:
    collection = get_commit_collection()
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if len(repository_ids) == 1:
        query_kwargs["where"] = {"repository_id": repository_ids[0]}
    else:
        query_kwargs["where"] = {"repository_id": {"$in": repository_ids}}

    try:
        return await asyncio.to_thread(
            collection.query,
            **query_kwargs,
        )
    except Exception as e:
        raise AppServiceError(
            "커밋 후보 조회에 실패했습니다.",
            status_code=502,
        ) from e


def _build_match_record(
    *,
    application: ApplicationEntry,
    application_index: int,
    commit_id: str,
    commit_document: str,
    metadata: dict[str, Any],
    distance: float | None,
) -> MatchRecord | None:
    stored_commit_id = _first_int(metadata.get("commit_id"))
    commit_ref = _first_str(metadata.get("commit_ref"))
    commit_hash = _first_str(metadata.get("commit_hash"))
    commit_message = _first_str(metadata.get("commit_message")) or commit_document
    repository_id = _first_int(metadata.get("repository_id"))

    direction_primary = _first_str(metadata.get("direction_primary")) or _first_str(
        metadata.get("direction")
    )
    direction_multi_csv = _first_str(metadata.get("direction_multi_csv")) or _first_str(
        metadata.get("direction")
    )
    direction_multi = _csv_to_list(direction_multi_csv)

    tech_keywords_csv = _first_str(metadata.get("tech_keywords_csv"))
    module_tags_csv = _first_str(metadata.get("module_tags_csv")) or _first_str(
        metadata.get("path_tokens_csv")
    )

    commit_keywords = extract_tech_keywords(
        commit_document,
        tech_keywords_csv,
    )
    commit_modules = extract_module_tokens(commit_document, module_tags_csv)
    commit_direction_labels = normalize_direction_labels(
        direction_primary,
        direction_multi_csv,
    )
    if not commit_direction_labels:
        commit_direction_labels = extract_direction_labels_from_text(commit_document)

    score = calculate_match_score(
        ScoringInput(
            semantic_distance=_normalize_distance(distance),
            application_text=application.text,
            commit_text=commit_document,
            commit_message=commit_message,
            application_direction_labels=application.direction_labels,
            commit_direction_labels=commit_direction_labels,
            application_keywords=application.keywords,
            commit_keywords=commit_keywords,
            application_modules=application.modules,
            commit_modules=commit_modules,
        )
    )

    if score.total < 50:
        return None

    matched_commit = MatchedCommit(
        commit_id=stored_commit_id,
        commit_ref=commit_ref,
        commit_hash=commit_hash,
        commit_message=commit_message,
        repository_id=repository_id,
        confidence=score.total,
        reason=_build_recommendation_reason(
            score=score,
            application_keywords=application.keywords,
            commit_keywords=commit_keywords,
            application_modules=application.modules,
            commit_modules=commit_modules,
        ),
        score_breakdown=MatchScoreBreakdown(
            semantic=score.semantic,
            keyword=score.keyword,
            context=score.context,
            penalty=score.penalty,
            total=score.total,
        ),
        direction_primary=direction_primary,
        direction_multi=direction_multi,
        tech_keywords=sorted(parse_csv_tokens(tech_keywords_csv)),
        module_tags=sorted(parse_csv_tokens(module_tags_csv)),
    )

    return MatchRecord(
        commit_key=_build_commit_key(commit_ref, commit_hash, commit_id),
        application_index=application_index,
        application_direction_labels=application.direction_labels,
        commit=matched_commit,
    )


async def match_applications_with_commits(
    payload: ApplicationCommitMatchRequest,
) -> ApplicationCommitMatchResponse:
    application_entries = await _load_application_entries(payload.meeting_id)
    if not application_entries:
        return ApplicationCommitMatchResponse(
            meeting_id=payload.meeting_id,
            repository_ids=payload.repository_ids,
            total_applications=0,
            matched_applications=0,
            applications=[],
        )

    pool_size = min(100, max(payload.top_k, payload.top_k * 5))
    matched_by_application: dict[int, dict[str, MatchRecord]] = {
        idx: {} for idx in range(len(application_entries))
    }
    repository_id_set = set(payload.repository_ids)

    for application_index, application in enumerate(application_entries):
        if not application.embedding:
            logger.warning(
                "적용사항 문서에 임베딩이 없어 후보 조회를 건너뜁니다: %s",
                application.document_id,
            )
            continue

        query_result = await _query_commit_candidates(
            application.embedding,
            n_results=pool_size,
            repository_ids=payload.repository_ids,
        )

        ids_nested = query_result.get("ids") or [[]]
        docs_nested = query_result.get("documents") or [[]]
        metas_nested = query_result.get("metadatas") or [[]]
        distances_nested = query_result.get("distances") or [[]]

        ids = ids_nested[0] if ids_nested else []
        docs = docs_nested[0] if docs_nested else []
        metadatas = metas_nested[0] if metas_nested else []
        distances = distances_nested[0] if distances_nested else []

        for idx, commit_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            commit_repository_id = _first_int(metadata.get("repository_id"))

            # Chroma where 필터 이후에도 metadata 불일치에 대비해 한 번 더 검증한다.
            if commit_repository_id not in repository_id_set:
                continue

            commit_document = docs[idx] if idx < len(docs) and docs[idx] else ""
            distance = distances[idx] if idx < len(distances) else None
            record = _build_match_record(
                application=application,
                application_index=application_index,
                commit_id=commit_id,
                commit_document=commit_document,
                metadata=metadata,
                distance=distance,
            )
            if not record:
                continue

            existing = matched_by_application[application_index].get(record.commit_key)
            if (
                existing is None
                or record.commit.score_breakdown.total
                > existing.commit.score_breakdown.total
            ):
                matched_by_application[application_index][record.commit_key] = record

    flattened_records = [
        record
        for records in matched_by_application.values()
        for record in records.values()
    ]
    resolved_records = _resolve_conflicting_matches(flattened_records)

    records_by_application: dict[int, list[MatchRecord]] = {
        idx: [] for idx in matched_by_application
    }
    for record in resolved_records:
        records_by_application[record.application_index].append(record)

    application_items: list[ApplicationCommitMatchItem] = []
    matched_applications = 0
    for idx, entry in enumerate(application_entries):
        sorted_records = sorted(
            records_by_application[idx],
            key=lambda record: record.commit.score_breakdown.total,
            reverse=True,
        )[: payload.top_k]

        recommended_commits = [record.commit for record in sorted_records]

        if recommended_commits:
            matched_applications += 1

        application_items.append(
            ApplicationCommitMatchItem(
                application_id=entry.application_id,
                application_document_id=entry.document_id,
                application_title=entry.application_title,
                recommended_commits=recommended_commits,
            )
        )

    return ApplicationCommitMatchResponse(
        meeting_id=payload.meeting_id,
        repository_ids=payload.repository_ids,
        total_applications=len(application_entries),
        matched_applications=matched_applications,
        applications=application_items,
    )
