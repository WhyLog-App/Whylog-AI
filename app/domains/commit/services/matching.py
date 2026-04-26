import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.core.chroma import get_commit_collection, get_decision_collection
from app.core.errors import AppServiceError
from app.domains.commit.schemas import (
    DecisionCommitMatchItem,
    DecisionCommitMatchRequest,
    DecisionCommitMatchResponse,
    MatchedCommit,
    MatchScoreBreakdown,
)
from app.domains.decision.services.matching_scoring import (
    ScoringInput,
    build_connection_reason,
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
class DecisionEntry:
    document_id: str
    text: str
    embedding: list[float] | None
    decision_title: str
    applied_item: str
    direction_labels: set[str]
    keywords: set[str]
    modules: set[str]


@dataclass(frozen=True)
class MatchRecord:
    commit_key: str
    decision_index: int
    decision_direction_labels: set[str]
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


def _to_decision_entries(raw: dict[str, Any]) -> list[DecisionEntry]:
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    embeddings = raw.get("embeddings")
    if embeddings is None:
        embeddings = []

    entries: list[DecisionEntry] = []
    for idx, document_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        text = documents[idx] if idx < len(documents) and documents[idx] else ""
        embedding = embeddings[idx] if idx < len(embeddings) else None

        decision_title = _first_str(metadata.get("decision_title")) or ""
        applied_item = _first_str(metadata.get("applied_item")) or ""
        full_text = f"{decision_title} {applied_item} {text}".strip()

        direction_labels = normalize_direction_labels(
            _first_str(metadata.get("direction_primary")),
            _first_str(metadata.get("direction_multi_csv")),
            _first_str(metadata.get("direction")),
        )
        if not direction_labels:
            direction_labels = extract_direction_labels_from_text(full_text)

        entries.append(
            DecisionEntry(
                document_id=document_id,
                text=text,
                embedding=_to_float_list(embedding),
                decision_title=decision_title,
                applied_item=applied_item,
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
                    candidate.decision_direction_labels,
                    kept.decision_direction_labels,
                )
                for kept in kept_for_commit
            ):
                continue
            kept_for_commit.append(candidate)

        kept_ids.update(id(record) for record in kept_for_commit)

    return [record for record in records if id(record) in kept_ids]


async def _load_decision_entries(meeting_id: str) -> list[DecisionEntry]:
    collection = get_decision_collection()
    try:
        raw = await asyncio.to_thread(
            collection.get,
            where={"meeting_id": meeting_id},
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception as e:
        raise AppServiceError(
            "결정사항 임베딩 조회에 실패했습니다.",
            status_code=502,
        ) from e

    return _to_decision_entries(raw)


async def _query_commit_candidates(
    query_embedding: list[float],
    *,
    n_results: int,
) -> dict[str, Any]:
    collection = get_commit_collection()
    try:
        return await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise AppServiceError(
            "커밋 후보 조회에 실패했습니다.",
            status_code=502,
        ) from e


def _build_match_record(
    *,
    decision: DecisionEntry,
    decision_index: int,
    commit_id: str,
    commit_document: str,
    metadata: dict[str, Any],
    distance: float | None,
) -> MatchRecord | None:
    stored_commit_id = _first_int(metadata.get("commit_id"))
    commit_ref = _first_str(metadata.get("commit_ref"))
    commit_hash = _first_str(metadata.get("commit_hash"))
    repository = _first_str(metadata.get("repository"))

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
            decision_text=decision.text,
            commit_text=commit_document,
            commit_message=(
                _first_str(metadata.get("commit_message")) or commit_document
            ),
            decision_direction_labels=decision.direction_labels,
            commit_direction_labels=commit_direction_labels,
            decision_keywords=decision.keywords,
            commit_keywords=commit_keywords,
            decision_modules=decision.modules,
            commit_modules=commit_modules,
        )
    )

    if score.total < 50:
        return None

    matched_commit = MatchedCommit(
        commit_id=stored_commit_id,
        commit_ref=commit_ref,
        commit_hash=commit_hash,
        repository=repository,
        status=score.status,
        confidence=score.total,
        reason=build_connection_reason(score),
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
        decision_index=decision_index,
        decision_direction_labels=decision.direction_labels,
        commit=matched_commit,
    )


async def match_decisions_with_commits(
    payload: DecisionCommitMatchRequest,
) -> DecisionCommitMatchResponse:
    decision_entries = await _load_decision_entries(payload.meeting_id)
    if not decision_entries:
        return DecisionCommitMatchResponse(
            meeting_id=payload.meeting_id,
            project_id=payload.project_id,
            repository=payload.repository,
            total_decision_items=0,
            matched_decision_items=0,
            decisions=[],
        )

    pool_size = min(100, max(payload.top_k, payload.top_k * 5))
    matched_by_decision: dict[int, dict[str, MatchRecord]] = {
        idx: {} for idx in range(len(decision_entries))
    }

    for decision_index, decision in enumerate(decision_entries):
        if not decision.embedding:
            logger.warning(
                "결정사항 문서에 임베딩이 없어 후보 조회를 건너뜁니다: %s",
                decision.document_id,
            )
            continue

        query_result = await _query_commit_candidates(
            decision.embedding,
            n_results=pool_size,
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
            commit_repository = _first_str(metadata.get("repository"))

            if payload.repository and commit_repository != payload.repository:
                continue
            commit_project = _first_str(metadata.get("project_id"))
            if (
                payload.project_id
                and commit_project
                and commit_project != payload.project_id
            ):
                continue

            commit_document = docs[idx] if idx < len(docs) and docs[idx] else ""
            distance = distances[idx] if idx < len(distances) else None
            record = _build_match_record(
                decision=decision,
                decision_index=decision_index,
                commit_id=commit_id,
                commit_document=commit_document,
                metadata=metadata,
                distance=distance,
            )
            if not record:
                continue

            existing = matched_by_decision[decision_index].get(record.commit_key)
            if (
                existing is None
                or record.commit.score_breakdown.total
                > existing.commit.score_breakdown.total
            ):
                matched_by_decision[decision_index][record.commit_key] = record

    flattened_records = [
        record
        for records in matched_by_decision.values()
        for record in records.values()
    ]
    resolved_records = _resolve_conflicting_matches(flattened_records)

    records_by_decision: dict[int, list[MatchRecord]] = {
        idx: [] for idx in matched_by_decision
    }
    for record in resolved_records:
        records_by_decision[record.decision_index].append(record)

    decision_items: list[DecisionCommitMatchItem] = []
    matched_decision_items = 0
    for idx, entry in enumerate(decision_entries):
        sorted_records = sorted(
            records_by_decision[idx],
            key=lambda record: record.commit.score_breakdown.total,
            reverse=True,
        )[: payload.top_k]

        connected_commits = [
            record.commit
            for record in sorted_records
            if record.commit.status == "APPLIED"
        ]
        recommended_commits = [
            record.commit
            for record in sorted_records
            if record.commit.status == "PARTIAL"
        ]

        if connected_commits:
            decision_status = "APPLIED"
            matched_decision_items += 1
        elif recommended_commits:
            decision_status = "PARTIAL"
            matched_decision_items += 1
        else:
            decision_status = "UNAPPLIED"

        decision_items.append(
            DecisionCommitMatchItem(
                decision_document_id=entry.document_id,
                decision_title=entry.decision_title,
                applied_item=entry.applied_item,
                decision_status=decision_status,
                connected_commits=connected_commits,
                recommended_commits=recommended_commits,
            )
        )

    return DecisionCommitMatchResponse(
        meeting_id=payload.meeting_id,
        project_id=payload.project_id,
        repository=payload.repository,
        total_decision_items=len(decision_entries),
        matched_decision_items=matched_decision_items,
        decisions=decision_items,
    )
