from app.domains.decision.services.matching_scoring import (
    ScoringInput,
    build_connection_reason,
    calculate_match_score,
    extract_direction_labels_from_text,
    extract_module_tokens,
    extract_tech_keywords,
    normalize_direction_labels,
    score_context_match,
    score_keyword_match,
)


def _make_payload(
    *,
    distance: float | None = 0.12,
    application_text: str = (
        "Redis 도입으로 notification module 처리 지연을 줄이고 queue 안정성을 확보한다."
    ),
    commit_text: str = (
        "introduce redis pubsub for notification queue and improve retry."
    ),
    commit_message: str = "feat: introduce redis pubsub for notification queue",
    application_direction: set[str] | None = None,
    commit_direction: set[str] | None = None,
    application_keywords: set[str] | None = None,
    commit_keywords: set[str] | None = None,
    application_modules: set[str] | None = None,
    commit_modules: set[str] | None = None,
) -> ScoringInput:
    resolved_application_direction = (
        application_direction
        if application_direction is not None
        else extract_direction_labels_from_text(application_text)
    )
    resolved_commit_direction = (
        commit_direction
        if commit_direction is not None
        else normalize_direction_labels("introduce")
        | extract_direction_labels_from_text(commit_text)
    )
    resolved_application_keywords = (
        application_keywords
        if application_keywords is not None
        else extract_tech_keywords(application_text)
    )
    resolved_commit_keywords = (
        commit_keywords
        if commit_keywords is not None
        else extract_tech_keywords(commit_text, "redis")
    )
    resolved_application_modules = (
        application_modules
        if application_modules is not None
        else extract_module_tokens(application_text)
    )
    resolved_commit_modules = (
        commit_modules
        if commit_modules is not None
        else extract_module_tokens(commit_text, "notification,queue")
    )

    return ScoringInput(
        semantic_distance=distance,
        application_text=application_text,
        commit_text=commit_text,
        commit_message=commit_message,
        application_direction_labels=resolved_application_direction,
        commit_direction_labels=resolved_commit_direction,
        application_keywords=resolved_application_keywords,
        commit_keywords=resolved_commit_keywords,
        application_modules=resolved_application_modules,
        commit_modules=resolved_commit_modules,
    )


class TestKeywordScorePolicy:
    def test_one_keyword_overlap_returns_15(self):
        score = score_keyword_match({"redis"}, {"redis", "kafka"})
        assert score == 15

    def test_two_keyword_overlap_returns_25(self):
        score = score_keyword_match({"redis", "kafka"}, {"redis", "kafka", "postgres"})
        assert score == 25

    def test_three_or_more_overlap_returns_30(self):
        score = score_keyword_match(
            {"redis", "kafka", "postgres"},
            {"redis", "kafka", "postgres", "spring"},
        )
        assert score == 30


class TestContextScorePolicy:
    def test_same_domain_folder_returns_20(self):
        score = score_context_match(
            {"notification", "queue"},
            {"notification", "queue", "producer"},
        )
        assert score == 20

    def test_indirect_context_returns_10(self):
        score = score_context_match({"notification"}, {"notificationservice"})
        assert score == 10

    def test_unrelated_context_returns_0(self):
        score = score_context_match({"billing"}, {"auth", "token"})
        assert score == 0


class TestTotalScorePolicy:
    def test_strong_match_becomes_applied(self):
        score = calculate_match_score(_make_payload())

        assert score.semantic >= 40
        assert score.keyword >= 15
        assert score.context >= 10
        assert score.total >= 70
        assert score.status == "APPLIED"

    def test_boundary_score_70_is_applied(self):
        payload = _make_payload(
            distance=0.10,  # semantic 45
            application_keywords={"redis", "kafka"},
            commit_keywords={"redis"},  # keyword 15
            application_modules={"notification"},
            commit_modules={"notification"},  # context 10
        )
        score = calculate_match_score(payload)

        assert score.total == 70
        assert score.status == "APPLIED"

    def test_boundary_score_69_is_partial(self):
        payload = _make_payload(
            distance=0.12,  # semantic 44
            application_keywords={"redis", "kafka"},
            commit_keywords={"redis"},  # keyword 15
            application_modules={"notification"},
            commit_modules={"notification"},  # context 10
        )
        score = calculate_match_score(payload)

        assert score.total == 69
        assert score.status == "PARTIAL"

    def test_boundary_score_50_is_partial(self):
        payload = _make_payload(
            distance=0.30,  # semantic 35
            application_keywords={"redis", "kafka"},
            commit_keywords={"redis"},  # keyword 15
            application_modules={"notification"},
            commit_modules={"billing"},  # context 0
        )
        score = calculate_match_score(payload)

        assert score.total == 50
        assert score.status == "PARTIAL"

    def test_boundary_score_49_is_unapplied(self):
        payload = _make_payload(
            distance=0.32,  # semantic 34
            application_keywords={"redis", "kafka"},
            commit_keywords={"redis"},  # keyword 15
            application_modules={"notification"},
            commit_modules={"billing"},  # context 0
        )
        score = calculate_match_score(payload)

        assert score.total == 49
        assert score.status == "UNAPPLIED"

    def test_opposite_direction_sets_semantic_to_zero(self):
        payload = _make_payload(
            distance=0.05,  # semantic high expected, but must be zero
            application_direction={"positive"},
            commit_direction={"negative"},
        )
        score = calculate_match_score(payload)

        assert score.semantic == 0

    def test_goal_mismatch_keyword_only_forces_zero(self):
        payload = _make_payload(
            distance=0.90,  # semantic 5
            application_keywords={"redis"},
            commit_keywords={"redis"},  # keyword 15
            application_modules={"notification"},
            commit_modules={"auth"},
        )
        score = calculate_match_score(payload)

        assert score.is_goal_mismatch is True
        assert score.total == 0
        assert score.status == "UNAPPLIED"

    def test_abstract_commit_message_penalty_minus_10(self):
        payload = _make_payload(
            distance=0.30,  # semantic 35
            commit_message="refactor",
            application_keywords={"redis", "kafka"},
            commit_keywords={"redis", "kafka"},  # keyword 25
            application_modules={"notification"},
            commit_modules={"notification"},  # context 10
        )
        score = calculate_match_score(payload)

        assert score.penalty >= 10
        assert score.total <= 60

    def test_ambiguous_application_penalty_minus_10(self):
        payload = _make_payload(
            distance=0.20,  # semantic 40
            application_text="개선 방향 논의",
            application_keywords=set(),
            application_modules=set(),
            commit_keywords={"redis", "kafka"},
            commit_modules={"notification"},
        )
        score = calculate_match_score(payload)

        assert score.penalty >= 10

    def test_connection_reason_is_readable(self):
        score = calculate_match_score(_make_payload())
        reason = build_connection_reason(score)

        assert reason.endswith(".")
        assert len(reason) > 10
