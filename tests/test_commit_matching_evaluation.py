from pathlib import Path

from app.domains.commit.services.matching_evaluation import (
    GoldenMatchCase,
    evaluate_match_response,
    load_golden_cases,
)

FIXTURE_PATH = Path("tests/fixtures/commit_matching_golden_cases.json")


def _commit(commit_hash: str) -> dict:
    return {
        "commit_hash": commit_hash,
        "commit_message": "test commit",
        "confidence": 90,
    }


def _application(
    application_id: int,
    application_title: str,
    commits: list[dict],
) -> dict:
    return {
        "application_id": application_id,
        "application_document_id": f"eval_application_{application_id}",
        "application_title": application_title,
        "recommended_commits": commits,
    }


def test_load_golden_cases_from_fixture():
    cases = load_golden_cases(FIXTURE_PATH)

    assert len(cases) == 5
    assert cases[0].case_id == "gemini-quota-retry"
    assert cases[0].should_match is True
    assert cases[0].application_reasons
    assert cases[0].distractor_commit_hashes == (
        "7c67758a4dfaca4cf18e264da011569ae868ba64",
    )
    assert cases[-1].case_id == "no-match-dinner-menu"
    assert cases[-1].should_match is False


def test_evaluate_match_response_passes_when_expected_commits_are_returned():
    cases = load_golden_cases(FIXTURE_PATH)
    response = {
        "result": {
            "applications": [
                _application(
                    9101,
                    "Gemini API 사용량 초과 시 재시도 로직 추가",
                    [_commit("5706bbf")],
                ),
                _application(
                    9102,
                    "WebSocket 발화 로그와 STT 전사 병합 기준 보강",
                    [_commit("cd139b0")],
                ),
                _application(
                    9103,
                    "적용사항 임베딩에 회의 타임라인 맥락 반영",
                    [_commit("3b50917")],
                ),
                _application(
                    9104,
                    "커밋 파일 경로 기반 모듈 토큰 보강",
                    [_commit("daafb71")],
                ),
                _application(9105, "저녁 메뉴로 치킨과 맥주를 선정", []),
            ]
        }
    }

    summary = evaluate_match_response(cases, response, top_k=5)

    assert summary.total_cases == 5
    assert summary.passed_cases == 5
    assert summary.recall_at_k == 1.0
    assert summary.precision_at_k == 1.0
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.no_match_accuracy == 1.0
    assert summary.false_positive_count == 0


def test_evaluate_match_response_counts_false_positives_and_missed_no_match():
    cases = load_golden_cases(FIXTURE_PATH)
    response = {
        "applications": [
            _application(
                9101,
                "Gemini API 사용량 초과 시 재시도 로직 추가",
                [_commit("7c67758"), _commit("5706bbf")],
            ),
            _application(
                9105,
                "저녁 메뉴로 치킨과 맥주를 선정",
                [_commit("8af78f0")],
            ),
        ]
    }

    summary = evaluate_match_response(cases[:1] + cases[-1:], response, top_k=5)

    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.recall_at_k == 1.0
    assert summary.precision_at_k == 1 / 3
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.no_match_accuracy == 0.0
    assert summary.false_positive_count == 2
    assert summary.cases[0].first_expected_rank == 2


def test_evaluate_match_response_uses_title_fallback_when_id_is_missing():
    cases = [
        GoldenMatchCase(
            case_id="title-fallback",
            application_id=None,
            application_title="회의 타임라인 맥락 임베딩",
            application_reasons=(),
            expected_commit_hashes=("3b50917d1ca2f9f811734ac02604bdae93dec86f",),
            accepted_commit_hashes=(),
            distractor_commit_hashes=(),
            should_match=True,
            tags=(),
        )
    ]
    response = {
        "applications": [
            _application(
                9103,
                "회의 타임라인 맥락 임베딩",
                [_commit("3b50917")],
            )
        ]
    }

    summary = evaluate_match_response(cases, response, top_k=5)

    assert summary.passed_cases == 1
    assert summary.cases[0].first_expected_rank == 1


def test_evaluate_match_response_treats_accepted_hash_as_relevant_not_expected():
    cases = [
        GoldenMatchCase(
            case_id="accepted-only",
            application_id=9101,
            application_title="Gemini 재시도 로직",
            application_reasons=(),
            expected_commit_hashes=("5706bbfddb71a374545106ee3c7fc83797925964",),
            accepted_commit_hashes=("7c67758a4dfaca4cf18e264da011569ae868ba64",),
            distractor_commit_hashes=(),
            should_match=True,
            tags=(),
        )
    ]
    response = {
        "applications": [
            _application(
                9101,
                "Gemini 재시도 로직",
                [_commit("7c67758")],
            )
        ]
    }

    summary = evaluate_match_response(cases, response, top_k=5)

    assert summary.passed_cases == 0
    assert summary.recall_at_k == 0.0
    assert summary.precision_at_k == 1.0
    assert summary.cases[0].expected_found is False
    assert summary.cases[0].false_positive_hashes == ()


def test_evaluate_match_response_fails_match_case_when_application_is_missing():
    cases = [
        GoldenMatchCase(
            case_id="missing-application",
            application_id=9999,
            application_title="없는 적용사항",
            application_reasons=(),
            expected_commit_hashes=("5706bbfddb71a374545106ee3c7fc83797925964",),
            accepted_commit_hashes=(),
            distractor_commit_hashes=(),
            should_match=True,
            tags=(),
        )
    ]

    summary = evaluate_match_response(cases, {"applications": []}, top_k=5)

    assert summary.passed_cases == 0
    assert summary.recall_at_k == 0.0
    assert summary.cases[0].recommended_count == 0


def test_evaluate_match_response_tracks_distractor_hits():
    cases = load_golden_cases(FIXTURE_PATH)
    response = {
        "applications": [
            _application(
                9101,
                "Gemini API 사용량 초과 시 재시도 로직 추가",
                [_commit("7c67758"), _commit("5706bbf")],
            )
        ]
    }

    summary = evaluate_match_response(cases[:1], response, top_k=5)

    assert summary.distractor_hit_count == 1
    assert summary.high_confidence_false_positive_count == 1
    assert summary.cases[0].distractor_hit_hashes == ("7c67758",)
    assert summary.cases[0].high_confidence_false_positive_hashes == ("7c67758",)


def test_evaluate_match_response_can_fail_match_case_on_false_positive():
    cases = load_golden_cases(FIXTURE_PATH)
    response = {
        "applications": [
            _application(
                9101,
                "Gemini API 사용량 초과 시 재시도 로직 추가",
                [_commit("7c67758"), _commit("5706bbf")],
            )
        ]
    }

    summary = evaluate_match_response(
        cases[:1],
        response,
        top_k=5,
        fail_on_false_positive=True,
    )

    assert summary.passed_cases == 0
    assert summary.cases[0].expected_found is True
    assert summary.cases[0].first_expected_rank == 2


def test_no_match_case_ignores_below_threshold_recommendation():
    cases = load_golden_cases(FIXTURE_PATH)
    low_confidence_commit = _commit("8af78f0") | {"confidence": 49}
    response = {
        "applications": [
            _application(
                9105,
                "저녁 메뉴로 치킨과 맥주를 선정",
                [low_confidence_commit],
            ),
        ]
    }

    summary = evaluate_match_response(cases[-1:], response, top_k=5)

    assert summary.passed_cases == 1
    assert summary.no_match_accuracy == 1.0
    assert summary.false_positive_count == 1
    assert summary.high_confidence_false_positive_count == 0
