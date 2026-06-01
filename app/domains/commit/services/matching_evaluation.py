import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIDENCE_THRESHOLD = 70


@dataclass(frozen=True)
class GoldenMatchCase:
    case_id: str
    application_id: int | None
    application_title: str
    application_reasons: tuple[str, ...]
    expected_commit_hashes: tuple[str, ...]
    accepted_commit_hashes: tuple[str, ...]
    distractor_commit_hashes: tuple[str, ...]
    should_match: bool
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    application_title: str
    should_match: bool
    passed: bool
    expected_found: bool
    first_expected_rank: int | None
    recommended_count: int
    recommended_hashes: tuple[str, ...]
    false_positive_hashes: tuple[str, ...]
    high_confidence_false_positive_hashes: tuple[str, ...]
    distractor_hit_hashes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "application_title": self.application_title,
            "should_match": self.should_match,
            "passed": self.passed,
            "expected_found": self.expected_found,
            "first_expected_rank": self.first_expected_rank,
            "recommended_count": self.recommended_count,
            "recommended_hashes": list(self.recommended_hashes),
            "false_positive_hashes": list(self.false_positive_hashes),
            "high_confidence_false_positive_hashes": list(
                self.high_confidence_false_positive_hashes
            ),
            "distractor_hit_hashes": list(self.distractor_hit_hashes),
        }


@dataclass(frozen=True)
class MatchEvaluationSummary:
    total_cases: int
    passed_cases: int
    match_cases: int
    no_match_cases: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    no_match_accuracy: float
    false_positive_count: int
    high_confidence_false_positive_count: int
    distractor_hit_count: int
    cases: tuple[CaseEvaluation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "match_cases": self.match_cases,
            "no_match_cases": self.no_match_cases,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "no_match_accuracy": self.no_match_accuracy,
            "false_positive_count": self.false_positive_count,
            "high_confidence_false_positive_count": (
                self.high_confidence_false_positive_count
            ),
            "distractor_hit_count": self.distractor_hit_count,
            "cases": [case.as_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class _Recommendation:
    commit_hash: str
    confidence: int | None


def load_golden_cases(path: str | Path) -> list[GoldenMatchCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    parsed_cases = [_golden_case_from_mapping(case) for case in cases]
    _validate_unique_case_ids(parsed_cases)
    return parsed_cases


def evaluate_match_response(
    cases: list[GoldenMatchCase],
    response: dict[str, Any],
    *,
    top_k: int = 5,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    fail_on_false_positive: bool = False,
) -> MatchEvaluationSummary:
    result = _extract_result(response)
    application_items = result.get("applications") or []
    by_application_id = {
        item.get("application_id"): item
        for item in application_items
        if item.get("application_id") is not None
    }
    by_application_title = {
        item.get("application_title"): item
        for item in application_items
        if item.get("application_title")
    }

    evaluations: list[CaseEvaluation] = []
    total_relevant_recommendations = 0
    total_recommendations = 0
    reciprocal_ranks: list[float] = []

    for case in cases:
        item = _find_application_item(
            case,
            by_application_id=by_application_id,
            by_application_title=by_application_title,
        )
        raw_recommendations = (item or {}).get("recommended_commits") or []
        recommendations = _to_recommendations(raw_recommendations[:top_k])
        recommended_hashes = tuple(
            recommendation.commit_hash for recommendation in recommendations
        )

        relevant_hashes = case.expected_commit_hashes + case.accepted_commit_hashes
        first_expected_rank = _first_matching_rank(
            recommended_hashes,
            case.expected_commit_hashes,
        )
        expected_found = first_expected_rank is not None
        false_positive_hashes = tuple(
            recommendation.commit_hash
            for recommendation in recommendations
            if not _hash_in_set(recommendation.commit_hash, relevant_hashes)
        )
        high_confidence_false_positive_hashes = tuple(
            recommendation.commit_hash
            for recommendation in recommendations
            if _is_high_confidence(recommendation, confidence_threshold)
            and not _hash_in_set(recommendation.commit_hash, relevant_hashes)
        )
        high_confidence_hashes = tuple(
            recommendation.commit_hash
            for recommendation in recommendations
            if _is_high_confidence(recommendation, confidence_threshold)
        )
        distractor_hit_hashes = tuple(
            recommendation.commit_hash
            for recommendation in recommendations
            if _hash_in_set(recommendation.commit_hash, case.distractor_commit_hashes)
        )
        passed = _case_passed(
            case=case,
            expected_found=expected_found,
            high_confidence_hashes=high_confidence_hashes,
            high_confidence_false_positive_hashes=(
                high_confidence_false_positive_hashes
            ),
            fail_on_false_positive=fail_on_false_positive,
        )

        total_recommendations += len(recommended_hashes)
        total_relevant_recommendations += sum(
            1
            for commit_hash in recommended_hashes
            if _hash_in_set(commit_hash, relevant_hashes)
        )
        if case.should_match:
            reciprocal_ranks.append(
                0.0 if first_expected_rank is None else 1 / first_expected_rank
            )

        evaluations.append(
            CaseEvaluation(
                case_id=case.case_id,
                application_title=case.application_title,
                should_match=case.should_match,
                passed=passed,
                expected_found=expected_found,
                first_expected_rank=first_expected_rank,
                recommended_count=len(recommended_hashes),
                recommended_hashes=recommended_hashes,
                false_positive_hashes=false_positive_hashes,
                high_confidence_false_positive_hashes=(
                    high_confidence_false_positive_hashes
                ),
                distractor_hit_hashes=distractor_hit_hashes,
            )
        )

    match_cases = sum(1 for case in cases if case.should_match)
    no_match_cases = len(cases) - match_cases
    passed_cases = sum(1 for evaluation in evaluations if evaluation.passed)
    found_cases = sum(1 for evaluation in evaluations if evaluation.expected_found)
    no_match_passes = sum(
        1
        for evaluation in evaluations
        if not evaluation.should_match and evaluation.passed
    )
    false_positive_count = sum(
        len(evaluation.false_positive_hashes) for evaluation in evaluations
    )
    high_confidence_false_positive_count = sum(
        len(evaluation.high_confidence_false_positive_hashes)
        for evaluation in evaluations
    )
    distractor_hit_count = sum(
        len(evaluation.distractor_hit_hashes) for evaluation in evaluations
    )

    return MatchEvaluationSummary(
        total_cases=len(cases),
        passed_cases=passed_cases,
        match_cases=match_cases,
        no_match_cases=no_match_cases,
        recall_at_k=_safe_div(found_cases, match_cases),
        precision_at_k=_safe_div(total_relevant_recommendations, total_recommendations),
        mean_reciprocal_rank=_safe_div(sum(reciprocal_ranks), len(reciprocal_ranks)),
        no_match_accuracy=_safe_div(no_match_passes, no_match_cases),
        false_positive_count=false_positive_count,
        high_confidence_false_positive_count=high_confidence_false_positive_count,
        distractor_hit_count=distractor_hit_count,
        cases=tuple(evaluations),
    )


def _golden_case_from_mapping(raw: dict[str, Any]) -> GoldenMatchCase:
    expected_commit_hashes = tuple(
        _normalize_hash(value)
        for value in raw.get("expected_commit_hashes", [])
        if _normalize_hash(value)
    )
    should_match = bool(raw.get("should_match", True))
    if should_match and not expected_commit_hashes:
        raise ValueError(f"expected_commit_hashes is required: {raw.get('case_id')}")

    return GoldenMatchCase(
        case_id=str(raw["case_id"]),
        application_id=raw.get("application_id"),
        application_title=str(raw["application_title"]),
        application_reasons=tuple(
            str(reason) for reason in raw.get("application_reasons", [])
        ),
        expected_commit_hashes=expected_commit_hashes,
        accepted_commit_hashes=tuple(
            _normalize_hash(value)
            for value in raw.get("accepted_commit_hashes", [])
            if _normalize_hash(value)
        ),
        distractor_commit_hashes=tuple(
            _normalize_hash(value)
            for value in raw.get("distractor_commit_hashes", [])
            if _normalize_hash(value)
        ),
        should_match=should_match,
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
    )


def _validate_unique_case_ids(cases: list[GoldenMatchCase]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            duplicates.add(case.case_id)
        seen.add(case.case_id)
    if duplicates:
        raise ValueError(f"duplicate case_id values: {sorted(duplicates)}")


def _extract_result(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if isinstance(result, dict) and "applications" in result:
        return result
    return response


def _find_application_item(
    case: GoldenMatchCase,
    *,
    by_application_id: dict[int, dict[str, Any]],
    by_application_title: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if case.application_id is not None and case.application_id in by_application_id:
        return by_application_id[case.application_id]
    return by_application_title.get(case.application_title)


def _to_recommendations(raw_recommendations: list[Any]) -> tuple[_Recommendation, ...]:
    recommendations: list[_Recommendation] = []
    for raw in raw_recommendations:
        if not isinstance(raw, dict):
            continue
        commit_hash = _normalize_hash(raw.get("commit_hash"))
        if not commit_hash:
            continue
        recommendations.append(
            _Recommendation(
                commit_hash=commit_hash,
                confidence=_normalize_confidence(raw.get("confidence")),
            )
        )
    return tuple(recommendations)


def _normalize_confidence(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _is_high_confidence(
    recommendation: _Recommendation,
    confidence_threshold: int,
) -> bool:
    # Older response fixtures may not have confidence. Treat those recommendations as
    # above-threshold because /api/commit/match already filters recommendations.
    if recommendation.confidence is None:
        return True
    return recommendation.confidence >= confidence_threshold


def _case_passed(
    *,
    case: GoldenMatchCase,
    expected_found: bool,
    high_confidence_hashes: tuple[str, ...],
    high_confidence_false_positive_hashes: tuple[str, ...],
    fail_on_false_positive: bool,
) -> bool:
    if not case.should_match:
        return not high_confidence_hashes
    if not expected_found:
        return False
    if fail_on_false_positive and high_confidence_false_positive_hashes:
        return False
    return True


def _first_matching_rank(
    recommended_hashes: tuple[str, ...],
    expected_hashes: tuple[str, ...],
) -> int | None:
    for index, commit_hash in enumerate(recommended_hashes, start=1):
        if _hash_in_set(commit_hash, expected_hashes):
            return index
    return None


def _hash_in_set(commit_hash: str, expected_hashes: tuple[str, ...]) -> bool:
    return any(_hash_matches(commit_hash, expected) for expected in expected_hashes)


def _hash_matches(left: str, right: str) -> bool:
    if len(left) < 7 or len(right) < 7:
        return left == right
    return left.startswith(right) or right.startswith(left)


def _normalize_hash(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
