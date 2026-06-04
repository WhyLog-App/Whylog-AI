from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.enums import CommitType, MatchStatus

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._/-]{1,}|[가-힣]{2,}")
_PATH_LIKE_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")
_COMMIT_TYPE_PATTERN = re.compile(r"^\s*([a-zA-Z]+)(?:\([^)]+\))?!?:")
_TYPE_BONUS = 3

_GENERIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "bug",
    "chore",
    "change",
    "changes",
    "code",
    "commit",
    "component",
    "components",
    "config",
    "data",
    "detail",
    "details",
    "feature",
    "feat",
    "field",
    "fields",
    "fix",
    "hotfix",
    "id",
    "improve",
    "issue",
    "item",
    "list",
    "minor",
    "page",
    "pages",
    "patch",
    "refactor",
    "request",
    "response",
    "result",
    "results",
    "screen",
    "state",
    "status",
    "test",
    "text",
    "title",
    "view",
    "views",
    "todo",
    "ui",
    "update",
    "user",
    "work",
    "summary",
    "변경요약",
    "변경방향",
    "파일맥락",
    "기술키워드",
    "근거",
    "적용사항",
    "공통",
    "관리",
    "관련",
    "구조",
    "기능",
    "기반",
    "데이터",
    "개선",
    "구현",
    "구체화",
    "결과",
    "수정",
    "방식",
    "사용자",
    "상세",
    "상태",
    "서비스",
    "요청",
    "응답",
    "작업",
    "적용",
    "정보",
    "정리",
    "테스트",
    "추가",
    "커밋",
    "컴포넌트",
    "클라이언트",
    "페이지",
    "필드",
    "항목",
    "화면",
    "회의",
    "변경",
    "도입",
    "삭제",
    "제거",
    "전환",
}

_MODULE_STOPWORDS = {
    "api",
    "app",
    "build",
    "config",
    "component",
    "components",
    "controller",
    "core",
    "commit",
    "data",
    "domain",
    "dto",
    "helper",
    "id",
    "impl",
    "lib",
    "main",
    "model",
    "page",
    "pages",
    "repo",
    "repository",
    "result",
    "results",
    "screen",
    "service",
    "src",
    "state",
    "status",
    "test",
    "tests",
    "text",
    "title",
    "utils",
    "ui",
    "view",
    "views",
    "변경요약",
    "변경방향",
    "파일맥락",
    "기술키워드",
    "근거",
    "적용사항",
    "공통",
    "관리",
    "관련",
    "구조",
    "기반",
    "데이터",
    "구체화",
    "결과",
    "방식",
    "사용자",
    "상세",
    "상태",
    "서비스",
    "요청",
    "응답",
    "정보",
    "커밋",
    "컴포넌트",
    "클라이언트",
    "페이지",
    "필드",
    "항목",
    "화면",
    "회의",
}

_ABSTRACT_COMMIT_WORDS = {
    "change",
    "chore",
    "cleanup",
    "doc",
    "docs",
    "feature",
    "feat",
    "fix",
    "build",
    "minor",
    "patch",
    "refactor",
    "test",
    "tests",
    "update",
    "수정",
    "정리",
    "리팩토링",
    "변경",
}

_POSITIVE_DIRECTION_WORDS = {
    "add",
    "adopt",
    "apply",
    "create",
    "enable",
    "introduce",
    "migrate",
    "onboard",
    "도입",
    "추가",
    "적용",
    "전환",
    "활성화",
    "확대",
}

_NEGATIVE_DIRECTION_WORDS = {
    "delete",
    "disable",
    "drop",
    "remove",
    "retire",
    "revert",
    "rollback",
    "삭제",
    "비활성화",
    "제거",
    "중단",
    "철회",
    "폐기",
}

_LABEL_ALIASES = {
    "positive": "positive",
    "negative": "negative",
    "add": "positive",
    "adopt": "positive",
    "apply": "positive",
    "create": "positive",
    "enable": "positive",
    "introduce": "positive",
    "migrate": "positive",
    "onboard": "positive",
    "remove": "negative",
    "delete": "negative",
    "disable": "negative",
    "drop": "negative",
    "revert": "negative",
    "rollback": "negative",
    "retire": "negative",
    "도입": "positive",
    "추가": "positive",
    "적용": "positive",
    "전환": "positive",
    "활성화": "positive",
    "확대": "positive",
    "제거": "negative",
    "삭제": "negative",
    "비활성화": "negative",
    "롤백": "negative",
    "중단": "negative",
    "철회": "negative",
    "폐기": "negative",
}

_COMMIT_TYPE_ALIASES: dict[str, CommitType] = {
    "feature": CommitType.FEAT,
    "feat": CommitType.FEAT,
    "fix": CommitType.FIX,
    "bugfix": CommitType.FIX,
    "docs": CommitType.DOCS,
    "doc": CommitType.DOCS,
    "documentation": CommitType.DOCS,
    "refactor": CommitType.REFACTOR,
    "test": CommitType.TEST,
    "tests": CommitType.TEST,
    "build": CommitType.BUILD,
    "chore": CommitType.CHORE,
}

_DOCS_APPLICATION_WORDS = {
    "docs",
    "documentation",
    "openapi",
    "swagger",
    "문서",
    "문서화",
    "명세",
    "스웨거",
}
_FIX_APPLICATION_WORDS = {
    "bug",
    "bugfix",
    "error",
    "exception",
    "fix",
    "issue",
    "문제",
    "버그",
    "에러",
    "예외",
    "오류",
}
_FIX_ACTION_WORDS = {
    "fix",
    "resolve",
    "고침",
    "수정",
    "해결",
}
_REFACTOR_APPLICATION_WORDS = {
    "refactor",
    "refactoring",
    "리팩토링",
}
_FEATURE_APPLICATION_WORDS = {
    "add",
    "create",
    "enable",
    "feature",
    "implement",
    "introduce",
    "support",
    "구현",
    "도입",
    "생성",
    "연동",
    "적용",
    "지원",
    "추가",
}


@dataclass(frozen=True)
class ScoreBreakdown:
    semantic: int
    keyword: int
    context: int
    type_bonus: int
    penalty: int
    total: int
    status: MatchStatus
    is_opposite_direction: bool
    is_goal_mismatch: bool


@dataclass(frozen=True)
class ScoringInput:
    semantic_distance: float | None
    application_text: str
    commit_text: str
    commit_message: str
    application_direction_labels: set[str]
    commit_direction_labels: set[str]
    application_keywords: set[str]
    commit_keywords: set[str]
    application_modules: set[str]
    commit_modules: set[str]


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _split_compound_token(token: str) -> set[str]:
    normalized = _normalize_token(token)
    if not normalized:
        return set()

    parts = re.split(r"[._/\-]+", normalized)
    return {part for part in parts if part}


def parse_csv_tokens(raw: str | None) -> set[str]:
    if not raw:
        return set()

    tokens: set[str] = set()
    for part in re.split(r"[,\n;|]+", raw):
        stripped = _normalize_token(part)
        if not stripped:
            continue
        tokens.add(stripped)
        tokens.update(_split_compound_token(stripped))
    return tokens


def extract_direction_labels_from_text(text: str) -> set[str]:
    labels: set[str] = set()
    for token in extract_text_tokens(text):
        if token in _POSITIVE_DIRECTION_WORDS:
            labels.add("positive")
        if token in _NEGATIVE_DIRECTION_WORDS:
            labels.add("negative")
    return labels


def normalize_direction_labels(*values: str | None) -> set[str]:
    labels: set[str] = set()
    for value in values:
        for token in parse_csv_tokens(value):
            label = _LABEL_ALIASES.get(token)
            if label:
                labels.add(label)
    return labels


def extract_commit_type(commit_message: str) -> CommitType | None:
    match = _COMMIT_TYPE_PATTERN.match(commit_message or "")
    if not match:
        return None
    raw_type = _normalize_token(match.group(1))
    return _COMMIT_TYPE_ALIASES.get(raw_type)


def infer_application_commit_types(application_text: str) -> set[CommitType]:
    tokens = extract_text_tokens(application_text)
    expected_types: set[CommitType] = set()

    if tokens & _DOCS_APPLICATION_WORDS:
        expected_types.update({CommitType.DOCS, CommitType.CHORE})
    if tokens & _FIX_APPLICATION_WORDS and (
        not expected_types or tokens & _FIX_ACTION_WORDS
    ):
        expected_types.add(CommitType.FIX)
    if tokens & _REFACTOR_APPLICATION_WORDS:
        expected_types.add(CommitType.REFACTOR)
    if tokens & _FEATURE_APPLICATION_WORDS:
        expected_types.add(CommitType.FEAT)
    return expected_types


def score_commit_type_match(
    application_text: str,
    commit_message: str,
    commit_text: str = "",
) -> int:
    commit_type = extract_commit_type(commit_message)
    if not commit_type:
        return 0
    expected_types = infer_application_commit_types(application_text)
    commit_tokens = extract_text_tokens(f"{commit_message} {commit_text}")
    if (
        commit_type == CommitType.CHORE
        and CommitType.DOCS in expected_types
        and not commit_tokens & _DOCS_APPLICATION_WORDS
    ):
        return 0
    if commit_type in expected_types:
        return _TYPE_BONUS
    return 0


def extract_text_tokens(text: str) -> set[str]:
    if not text:
        return set()

    tokens: set[str] = set()
    for raw in _TOKEN_PATTERN.findall(text):
        normalized = _normalize_token(raw)
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(_split_compound_token(normalized))
    return tokens


def extract_tech_keywords(text: str, csv_keywords: str | None = None) -> set[str]:
    candidate_tokens = extract_text_tokens(text)
    candidate_tokens.update(parse_csv_tokens(csv_keywords))

    return {
        token
        for token in candidate_tokens
        if len(token) >= 2 and token not in _GENERIC_STOPWORDS
    }


def extract_module_tokens(text: str, csv_modules: str | None = None) -> set[str]:
    candidate_tokens = parse_csv_tokens(csv_modules)
    for path_like in _PATH_LIKE_PATTERN.findall(text or ""):
        candidate_tokens.add(_normalize_token(path_like))
        candidate_tokens.update(_split_compound_token(path_like))

    return {
        token
        for token in candidate_tokens
        if len(token) >= 2 and token not in _MODULE_STOPWORDS
    }


def _scoreable_keyword_tokens(tokens: set[str]) -> set[str]:
    return {
        normalized
        for token in tokens
        if (normalized := _normalize_token(token))
        and len(normalized) >= 2
        and normalized not in _GENERIC_STOPWORDS
    }


def _scoreable_module_tokens(tokens: set[str]) -> set[str]:
    return {
        normalized
        for token in tokens
        if (normalized := _normalize_token(token))
        and len(normalized) >= 2
        and normalized not in _MODULE_STOPWORDS
    }


def is_opposite_direction(
    application_labels: set[str],
    commit_labels: set[str],
) -> bool:
    if not application_labels or not commit_labels:
        return False
    return (
        "positive" in application_labels
        and "negative" in commit_labels
        or "negative" in application_labels
        and "positive" in commit_labels
    )


def score_semantic(
    semantic_distance: float | None,
    *,
    opposite_direction: bool,
) -> int:
    if opposite_direction:
        return 0
    if semantic_distance is None:
        return 0

    normalized_distance = max(0.0, min(1.0, semantic_distance))
    return int(round((1.0 - normalized_distance) * 50))


def score_keyword_match(
    application_keywords: set[str],
    commit_keywords: set[str],
) -> int:
    overlap_count = len(
        _scoreable_keyword_tokens(application_keywords)
        & _scoreable_keyword_tokens(commit_keywords)
    )
    if overlap_count <= 0:
        return 0
    if overlap_count == 1:
        return 15
    if overlap_count == 2:
        return 25
    return 30


def score_context_match(
    application_modules: set[str],
    commit_modules: set[str],
) -> int:
    application_modules = _scoreable_module_tokens(application_modules)
    commit_modules = _scoreable_module_tokens(commit_modules)
    if not application_modules or not commit_modules:
        return 0

    overlap = application_modules & commit_modules
    if len(overlap) >= 2:
        return 20
    if len(overlap) == 1:
        return 10

    # 간접 연관: prefix/substring 기반 1회라도 발견되면 10점.
    for application_token in application_modules:
        for commit_token in commit_modules:
            if application_token in commit_token or commit_token in application_token:
                return 10
    return 0


def is_abstract_commit_message(commit_message: str) -> bool:
    tokens = extract_text_tokens(commit_message)
    if not tokens:
        return True

    meaningful_tokens = tokens - _ABSTRACT_COMMIT_WORDS
    if not meaningful_tokens:
        return True

    if len(tokens) <= 3 and len(meaningful_tokens) <= 1:
        return True
    return False


def is_ambiguous_application(
    application_text: str,
    application_keywords: set[str],
    application_modules: set[str],
) -> bool:
    application_tokens = extract_text_tokens(application_text)
    if len(application_tokens) < 3:
        return True
    if not application_keywords and not application_modules:
        return True
    return False


def resolve_match_status(total_score: int) -> MatchStatus:
    if total_score >= 70:
        return MatchStatus.APPLIED
    if total_score >= 50:
        return MatchStatus.PARTIAL
    return MatchStatus.UNAPPLIED


def _format_overlap(tokens: set[str], *, limit: int = 3) -> str:
    sorted_tokens = sorted(tokens)
    shown = sorted_tokens[:limit]
    suffix = (
        "" if len(sorted_tokens) <= limit else f" 외 {len(sorted_tokens) - limit}개"
    )
    return ", ".join(shown) + suffix


def build_connection_reason(
    score: ScoreBreakdown,
    *,
    keyword_overlap: set[str] | None = None,
    module_overlap: set[str] | None = None,
) -> str:
    if score.is_opposite_direction:
        return "의미 방향이 반대여서 자동 연결을 제한했습니다."
    if score.is_goal_mismatch:
        return "키워드는 유사하지만 변경 목적이 달라 자동 연결을 제한했습니다."

    reason_parts: list[str] = []

    if score.semantic >= 30:
        reason_parts.append("의미 유사성이 높습니다")
    elif score.semantic > 0:
        reason_parts.append("의미 유사성이 일부 확인됩니다")

    if score.keyword >= 25 and keyword_overlap:
        reason_parts.append(
            f"{_format_overlap(keyword_overlap)} 키워드가 다수 일치합니다"
        )
    elif score.keyword >= 15 and keyword_overlap:
        reason_parts.append(f"{_format_overlap(keyword_overlap)} 키워드가 일치합니다")
    elif score.keyword >= 25:
        reason_parts.append("핵심 기술 키워드가 다수 일치합니다")
    elif score.keyword >= 15:
        reason_parts.append("핵심 기술 키워드가 일부 일치합니다")

    if score.context >= 20 and module_overlap:
        reason_parts.append(f"{_format_overlap(module_overlap)} 모듈이 직접 일치합니다")
    elif score.context >= 10 and module_overlap:
        reason_parts.append(f"{_format_overlap(module_overlap)} 모듈이 간접 일치합니다")
    elif score.context >= 20:
        reason_parts.append("파일/모듈 맥락이 직접 일치합니다")
    elif score.context >= 10:
        reason_parts.append("파일/모듈 맥락이 간접 일치합니다")

    if not reason_parts:
        reason_parts.append("복합 신호 기반으로 부분 일치가 확인됩니다")
    return ", ".join(reason_parts) + "."


def calculate_match_score(payload: ScoringInput) -> ScoreBreakdown:
    opposite_direction = is_opposite_direction(
        payload.application_direction_labels,
        payload.commit_direction_labels,
    )
    semantic = score_semantic(
        payload.semantic_distance,
        opposite_direction=opposite_direction,
    )
    keyword = score_keyword_match(payload.application_keywords, payload.commit_keywords)
    context = score_context_match(payload.application_modules, payload.commit_modules)

    goal_mismatch = keyword >= 15 and semantic <= 10
    if goal_mismatch:
        return ScoreBreakdown(
            semantic=semantic,
            keyword=keyword,
            context=context,
            type_bonus=0,
            penalty=0,
            total=0,
            status=MatchStatus.UNAPPLIED,
            is_opposite_direction=opposite_direction,
            is_goal_mismatch=True,
        )

    base_score = semantic + keyword + context
    type_bonus = score_commit_type_match(
        payload.application_text,
        payload.commit_message,
        payload.commit_text,
    )
    penalty = 0
    if is_abstract_commit_message(payload.commit_message):
        penalty += 10
    if is_ambiguous_application(
        payload.application_text,
        payload.application_keywords,
        payload.application_modules,
    ):
        penalty += 10

    total = max(0, min(100, base_score + type_bonus - penalty))
    return ScoreBreakdown(
        semantic=semantic,
        keyword=keyword,
        context=context,
        type_bonus=type_bonus,
        penalty=penalty,
        total=total,
        status=resolve_match_status(total),
        is_opposite_direction=opposite_direction,
        is_goal_mismatch=False,
    )
