import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from string import punctuation

from pydantic import TypeAdapter, ValidationError

from app.core.errors import AppServiceError
from app.domains.transcribe.schemas import LiveTranscriptMessage, TranscribeSegment

logger = logging.getLogger(__name__)

AMBIGUOUS_SHORT_TEXTS = {
    "네",
    "예",
    "응",
    "음",
    "어",
    "맞아요",
    "맞습니다",
    "아니요",
    "확인",
    "오케이",
    "ok",
    "okay",
    "ㅋㅋ",
    "ㅎㅎ",
}
MIN_MATCH_SCORE = 0.55
MIN_CORRECTION_SCORE = 0.72
MIN_SPEAKER_VOTES = 2
MIN_SPEAKER_SHARE = 0.6
MIN_SPEAKER_AVG_SCORE = 0.6
MIN_DOMINANT_MEMBER_SHARE = 0.75
MIN_TEXT_SIMILARITY_FOR_MATCH = 0.6
TIME_WINDOW_SECONDS = 12.0
_live_messages_adapter = TypeAdapter(list[LiveTranscriptMessage])


@dataclass(frozen=True)
class _LiveEntry:
    index: int
    message: LiveTranscriptMessage
    seconds: float | None
    normalized_text: str
    is_ambiguous: bool


@dataclass(frozen=True)
class _SegmentMatch:
    segment_index: int
    live: _LiveEntry
    score: float
    text_similarity: float


def parse_live_messages(raw: str | None) -> list[LiveTranscriptMessage]:
    # multipart/form-data의 JSON 문자열을 WebSocket 발화 로그 DTO로 변환
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return _live_messages_adapter.validate_python(parsed)
    except json.JSONDecodeError as e:
        raise AppServiceError(
            "live_messages는 JSON 배열 문자열이어야 합니다.",
            status_code=422,
        ) from e
    except ValidationError as e:
        raise AppServiceError(
            f"live_messages 스키마 검증 실패: {e}",
            status_code=422,
        ) from e


def merge_live_transcript(
    segments: list[TranscribeSegment],
    live_messages: list[LiveTranscriptMessage],
) -> list[TranscribeSegment]:
    # STT 결과와 WebSocket 발화 로그를 매칭해 최종 전사를 보강
    live_entries = _prepare_live_entries(live_messages)
    if not segments or not live_entries:
        logger.info(
            "live transcript merge skipped: "
            "segments=%s live_messages=%s live_entries=%s",
            len(segments),
            len(live_messages),
            len(live_entries),
        )
        return segments

    matches = _match_segments_to_live_messages(segments, live_entries)
    speaker_mapping = _resolve_speaker_mapping(segments, live_entries, matches)
    _log_merge_result(
        segments=segments,
        live_entries=live_entries,
        matches=matches,
        speaker_mapping=speaker_mapping,
    )
    return _apply_matches(
        segments,
        live_entries,
        matches,
        speaker_mapping,
    )


def _prepare_live_entries(
    live_messages: list[LiveTranscriptMessage],
) -> list[_LiveEntry]:
    entries: list[_LiveEntry] = []
    skipped: Counter[str] = Counter()
    for index, message in enumerate(live_messages):
        if message.type.upper() != "TEXT":
            skipped["non_text_type"] += 1
            continue
        text = (message.text or "").strip()
        if not text:
            skipped["empty_text"] += 1
            continue
        if message.from_member_id is None or not message.from_name:
            skipped["missing_member"] += 1
            continue
        normalized_text = _normalize_text(text)
        if not normalized_text:
            skipped["empty_normalized_text"] += 1
            continue
        entries.append(
            _LiveEntry(
                index=index,
                message=message,
                seconds=_parse_time_to_seconds(message.timestamp),
                normalized_text=normalized_text,
                is_ambiguous=_is_ambiguous_text(normalized_text),
            )
        )
    logger.info(
        "live transcript messages prepared: raw=%s prepared=%s skipped=%s samples=%s",
        len(live_messages),
        len(entries),
        dict(skipped),
        [_live_entry_debug_sample(entry) for entry in entries[:5]],
    )
    return entries


def _match_segments_to_live_messages(
    segments: list[TranscribeSegment],
    live_entries: list[_LiveEntry],
) -> dict[int, _SegmentMatch]:
    candidates: list[_SegmentMatch] = []
    for segment_index, segment in enumerate(segments):
        segment_seconds = _parse_time_to_seconds(segment.start_time)
        segment_text = _normalize_text(segment.text)
        if not segment_text:
            continue

        for live in live_entries:
            score, text_similarity = _score_match(
                segment_index=segment_index,
                segment_seconds=segment_seconds,
                segment_text=segment_text,
                live=live,
            )
            if (
                not live.is_ambiguous
                and text_similarity < MIN_TEXT_SIMILARITY_FOR_MATCH
            ):
                continue
            if score < MIN_MATCH_SCORE:
                continue
            candidates.append(
                _SegmentMatch(
                    segment_index=segment_index,
                    live=live,
                    score=score,
                    text_similarity=text_similarity,
                )
            )

    matches: dict[int, _SegmentMatch] = {}
    used_segments: set[int] = set()
    used_live_indexes: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda match: (
            match.score,
            match.text_similarity,
            -abs(match.segment_index - match.live.index),
        ),
        reverse=True,
    ):
        if candidate.segment_index in used_segments:
            continue
        if candidate.live.index in used_live_indexes:
            continue
        matches[candidate.segment_index] = candidate
        used_segments.add(candidate.segment_index)
        used_live_indexes.add(candidate.live.index)

    logger.info(
        "live transcript segment matching completed: "
        "segments=%s live_entries=%s matches=%s",
        len(segments),
        len(live_entries),
        len(matches),
    )
    return matches


def _score_match(
    segment_index: int,
    segment_seconds: float | None,
    segment_text: str,
    live: _LiveEntry,
) -> tuple[float, float]:
    text_similarity = SequenceMatcher(None, segment_text, live.normalized_text).ratio()
    time_score = _time_score(segment_seconds, live.seconds)
    order_score = _order_score(segment_index, live.index)

    score = (time_score * 0.45) + (text_similarity * 0.45) + (order_score * 0.10)
    if live.is_ambiguous:
        score *= 0.45
    return score, text_similarity


def _resolve_speaker_mapping(
    segments: list[TranscribeSegment],
    live_entries: list[_LiveEntry],
    matches: dict[int, _SegmentMatch],
) -> dict[str, tuple[int, str, float]]:
    votes: dict[str, dict[tuple[int, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for segment_index, match in matches.items():
        if match.live.is_ambiguous:
            continue
        segment = segments[segment_index]
        member_key = (
            match.live.message.from_member_id,
            match.live.message.from_name or "",
        )
        if member_key[0] is None or not member_key[1]:
            continue
        votes[segment.speaker][member_key].append(match.score)

    resolved: dict[str, tuple[int, str, float]] = {}
    for speaker, member_scores in votes.items():
        flattened_scores = [
            score for scores in member_scores.values() for score in scores
        ]
        if len(flattened_scores) < MIN_SPEAKER_VOTES:
            continue

        best_member, best_scores = max(
            member_scores.items(),
            key=lambda item: sum(item[1]),
        )
        best_count = len(best_scores)
        share = best_count / len(flattened_scores)
        avg_score = sum(best_scores) / best_count
        if share < MIN_SPEAKER_SHARE or avg_score < MIN_SPEAKER_AVG_SCORE:
            continue

        member_id, member_name = best_member
        if member_id is None:
            continue
        resolved[speaker] = (member_id, member_name, avg_score)
        logger.info(
            "speaker mapped: %s -> %s(member_id=%s, confidence=%.2f)",
            speaker,
            member_name,
            member_id,
            avg_score,
        )

    _apply_single_speaker_dominant_member_fallback(
        resolved=resolved,
        segments=segments,
        live_entries=live_entries,
        matches=matches,
    )
    return resolved


def _apply_single_speaker_dominant_member_fallback(
    resolved: dict[str, tuple[int, str, float]],
    segments: list[TranscribeSegment],
    live_entries: list[_LiveEntry],
    matches: dict[int, _SegmentMatch],
) -> None:
    # 짧은 회의에서는 텍스트 유사도 투표가 부족할 수 있어
    # 단일 화자/단일 멤버 흐름을 보강한다.
    speakers = {segment.speaker for segment in segments if segment.speaker}
    if len(speakers) != 1:
        return

    speaker = next(iter(speakers))
    if speaker in resolved:
        logger.info(
            "dominant-member fallback skipped: speaker already resolved speaker=%s",
            speaker,
        )
        return
    if not live_entries:
        logger.info("dominant-member fallback skipped: no live entries")
        return
    if not any(not live.is_ambiguous for live in live_entries):
        logger.info("dominant-member fallback skipped: all live entries are ambiguous")
        return

    member_votes: dict[tuple[int, str], int] = defaultdict(int)
    for live in live_entries:
        member_id = live.message.from_member_id
        member_name = live.message.from_name or ""
        if member_id is None or not member_name:
            continue
        member_votes[(member_id, member_name)] += 1
    if not member_votes:
        logger.info("dominant-member fallback skipped: no member votes")
        return

    best_member, best_count = max(member_votes.items(), key=lambda item: item[1])
    share = best_count / len(live_entries)
    if share < MIN_DOMINANT_MEMBER_SHARE:
        logger.info(
            "dominant-member fallback skipped: dominant share too low "
            "best_member_id=%s share=%.2f threshold=%.2f votes=%s",
            best_member[0],
            share,
            MIN_DOMINANT_MEMBER_SHARE,
            {member_id: count for (member_id, _), count in member_votes.items()},
        )
        return

    match_scores = [
        match.score
        for segment_index, match in matches.items()
        if 0 <= segment_index < len(segments)
        and segments[segment_index].speaker == speaker
        and match.live.message.from_member_id == best_member[0]
    ]
    confidence = sum(match_scores) / len(match_scores) if match_scores else share
    resolved[speaker] = (best_member[0], best_member[1], confidence)
    logger.info(
        "speaker mapped by dominant-member fallback: "
        "%s -> %s(member_id=%s, confidence=%.2f)",
        speaker,
        best_member[1],
        best_member[0],
        confidence,
    )


def _log_merge_result(
    segments: list[TranscribeSegment],
    live_entries: list[_LiveEntry],
    matches: dict[int, _SegmentMatch],
    speaker_mapping: dict[str, tuple[int, str, float]],
) -> None:
    null_speakers = sorted(
        {
            segment.speaker
            for segment in segments
            if segment.speaker and segment.speaker not in speaker_mapping
        }
    )
    logger.info(
        "live transcript merge result: segments=%s live_entries=%s matches=%s "
        "mapped_speakers=%s unmapped_speakers=%s match_samples=%s",
        len(segments),
        len(live_entries),
        len(matches),
        {
            speaker: {"member_id": member_id, "member_name": member_name}
            for speaker, (member_id, member_name, _) in speaker_mapping.items()
        },
        null_speakers,
        [_match_debug_sample(segments, match) for match in list(matches.values())[:5]],
    )


def _live_entry_debug_sample(entry: _LiveEntry) -> dict[str, object]:
    return {
        "index": entry.index,
        "meeting_id": entry.message.meeting_id,
        "member_id": entry.message.from_member_id,
        "member_name": entry.message.from_name,
        "timestamp": entry.message.timestamp,
        "seconds": entry.seconds,
        "ambiguous": entry.is_ambiguous,
        "text": _text_preview(entry.message.text),
    }


def _match_debug_sample(
    segments: list[TranscribeSegment],
    match: _SegmentMatch,
) -> dict[str, object]:
    segment = segments[match.segment_index]
    return {
        "segment_index": match.segment_index,
        "message_id": segment.message_id,
        "speaker": segment.speaker,
        "segment_time": segment.start_time,
        "live_index": match.live.index,
        "live_member_id": match.live.message.from_member_id,
        "score": round(match.score, 3),
        "text_similarity": round(match.text_similarity, 3),
        "segment_text": _text_preview(segment.text),
        "live_text": _text_preview(match.live.message.text),
    }


def _text_preview(value: str | None, limit: int = 40) -> str:
    normalized = " ".join((value or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _apply_matches(
    segments: list[TranscribeSegment],
    live_entries: list[_LiveEntry],
    matches: dict[int, _SegmentMatch],
    speaker_mapping: dict[str, tuple[int, str, float]],
) -> list[TranscribeSegment]:
    if not matches:
        live_only_segments = _build_live_only_segments(live_entries)
        if live_only_segments:
            logger.info(
                "live transcript merge using WebSocket-only fallback: "
                "segments=%s live_entries=%s",
                len(segments),
                len(live_entries),
            )
            return _renumber_segments(live_only_segments)
        return segments

    corrected: list[tuple[float, int, TranscribeSegment]] = []
    used_live_indexes = {match.live.index for match in matches.values()}
    for segment_index, segment in enumerate(segments):
        update: dict[str, object] = {}

        match = matches.get(segment_index)
        if match:
            member_id = match.live.message.from_member_id
            member_name = match.live.message.from_name
            if member_id is not None and member_name:
                update["speaker"] = member_name
                update["member_id"] = member_id

        if "member_id" not in update:
            speaker_match = speaker_mapping.get(segment.speaker)
            if speaker_match:
                member_id, member_name, _ = speaker_match
                update["speaker"] = member_name
                update["member_id"] = member_id

        if match and match.score >= MIN_CORRECTION_SCORE:
            live_text = (match.live.message.text or "").strip()
            if live_text:
                update["text"] = live_text
                if live_text != segment.text:
                    logger.info(
                        "transcript corrected: message_id=%s score=%.2f",
                        segment.message_id,
                        match.score,
                    )

        sort_key = _segment_sort_key(segment, segment_index)
        corrected.append((sort_key, segment_index, segment.model_copy(update=update)))

    if matches:
        for live in live_entries:
            if live.index in used_live_indexes:
                continue
            if live.is_ambiguous:
                continue

            live_segment = _build_live_only_segment(live, live_entries)
            if live_segment is None:
                continue
            sort_key = _live_sort_key(
                live=live,
                segments=segments,
                matches=matches,
            )
            corrected.append((sort_key, len(segments) + live.index, live_segment))
            logger.info(
                "live transcript segment added from unmatched WebSocket message: "
                "live_index=%s member_id=%s",
                live.index,
                live.message.from_member_id,
            )

    sorted_segments = [
        item[2] for item in sorted(corrected, key=lambda item: (item[0], item[1]))
    ]
    return _renumber_segments(sorted_segments)


def _build_live_only_segments(
    live_entries: list[_LiveEntry],
) -> list[TranscribeSegment]:
    segments = []
    for live in live_entries:
        if live.is_ambiguous:
            continue
        live_segment = _build_live_only_segment(live, live_entries)
        if live_segment is not None:
            segments.append(live_segment)
    return segments


def _renumber_segments(
    segments: list[TranscribeSegment],
) -> list[TranscribeSegment]:
    return [
        segment.model_copy(update={"message_id": index})
        for index, segment in enumerate(segments, start=1)
    ]


def _segment_sort_key(segment: TranscribeSegment, fallback_index: int) -> float:
    seconds = _parse_time_to_seconds(segment.start_time)
    if seconds is not None:
        return seconds
    return float(fallback_index)


def _live_sort_key(
    *,
    live: _LiveEntry,
    segments: list[TranscribeSegment],
    matches: dict[int, _SegmentMatch],
) -> float:
    if live.seconds is not None:
        return live.seconds

    anchors = sorted(
        (
            (
                match.live.index,
                _segment_sort_key(segments[segment_index], segment_index),
            )
            for segment_index, match in matches.items()
            if 0 <= segment_index < len(segments)
        ),
        key=lambda item: item[0],
    )
    if not anchors:
        return float(live.index)

    previous_anchor = next(
        (anchor for anchor in reversed(anchors) if anchor[0] < live.index),
        None,
    )
    next_anchor = next((anchor for anchor in anchors if anchor[0] > live.index), None)

    if previous_anchor and next_anchor:
        previous_index, previous_sort_key = previous_anchor
        next_index, next_sort_key = next_anchor
        progress = (live.index - previous_index) / (next_index - previous_index)
        return previous_sort_key + ((next_sort_key - previous_sort_key) * progress)
    if previous_anchor:
        previous_index, previous_sort_key = previous_anchor
        return previous_sort_key + ((live.index - previous_index) * 0.001)
    if next_anchor:
        next_index, next_sort_key = next_anchor
        return max(0.0, next_sort_key - ((next_index - live.index) * 0.001))
    return float(live.index)


def _build_live_only_segment(
    live: _LiveEntry,
    live_entries: list[_LiveEntry],
) -> TranscribeSegment | None:
    text = (live.message.text or "").strip()
    member_id = live.message.from_member_id
    member_name = live.message.from_name
    if not text or member_id is None or not member_name:
        return None

    start_seconds = live.seconds
    next_seconds = next(
        (
            entry.seconds
            for entry in live_entries
            if entry.index > live.index
            and entry.seconds is not None
            and start_seconds is not None
            and entry.seconds > start_seconds
        ),
        None,
    )
    if start_seconds is None:
        start_time = "00:00:00"
        end_time = "00:00:00"
    else:
        start_time = _format_seconds(start_seconds)
        end_time = _format_seconds(next_seconds or start_seconds)

    return TranscribeSegment(
        message_id=live.index + 1,
        speaker=member_name,
        member_id=member_id,
        start_time=start_time,
        end_time=end_time,
        text=text,
        is_final=True,
    )


def _format_seconds(value: float) -> str:
    total = max(0, int(value))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_time_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None

    # ISO 절대시각은 STT 상대 시간과 기준이 달라서 시간 점수에는 사용하지 않는다.
    if "T" in raw or re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return None

    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None


def _time_score(segment_seconds: float | None, live_seconds: float | None) -> float:
    if segment_seconds is None or live_seconds is None:
        return 0.4
    diff = abs(segment_seconds - live_seconds)
    if diff >= TIME_WINDOW_SECONDS:
        return 0.0
    return max(0.0, 1.0 - (diff / TIME_WINDOW_SECONDS))


def _order_score(segment_index: int, live_index: int) -> float:
    diff = abs(segment_index - live_index)
    if diff >= 6:
        return 0.0
    return max(0.0, 1.0 - (diff / 6))


def _normalize_text(value: str) -> str:
    cleaned = re.sub(rf"[{re.escape(punctuation)}]", " ", value.lower())
    cleaned = re.sub(r"[^\w가-힣\s]", " ", cleaned)
    return " ".join(cleaned.split())


def _is_ambiguous_text(value: str) -> bool:
    if value in AMBIGUOUS_SHORT_TEXTS:
        return True
    return len(value.replace(" ", "")) <= 2
