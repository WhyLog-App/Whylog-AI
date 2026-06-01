import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.domains.commit.services.matching_evaluation import (  # noqa: E402
    evaluate_match_response,
    load_golden_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate /api/commit/match response with golden cases.",
    )
    parser.add_argument(
        "--cases",
        default="tests/fixtures/commit_matching_golden_cases.json",
        help="Golden case JSON path.",
    )
    parser.add_argument(
        "--response",
        required=True,
        help="Saved /api/commit/match response JSON path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Evaluation cutoff for recommended commits.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=70,
        help="Confidence cutoff for high-confidence false positive checks.",
    )
    parser.add_argument(
        "--fail-on-false-positive",
        action="store_true",
        help=(
            "Fail matched cases when a non-relevant commit is recommended at or "
            "above the confidence threshold."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Exit with code 1 when at least one golden case fails.",
    )
    args = parser.parse_args()

    cases = load_golden_cases(args.cases)
    response = json.loads(Path(args.response).read_text(encoding="utf-8"))
    summary = evaluate_match_response(
        cases,
        response,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
        fail_on_false_positive=args.fail_on_false_positive,
    )

    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_summary(summary.as_dict())

    if args.fail_on_failure and summary.passed_cases < summary.total_cases:
        return 1
    return 0


def _print_summary(summary: dict) -> None:
    print("Commit Matching Evaluation")
    print(f"- total_cases: {summary['total_cases']}")
    print(f"- passed_cases: {summary['passed_cases']}")
    print(f"- recall_at_k: {summary['recall_at_k']:.3f}")
    print(f"- precision_at_k: {summary['precision_at_k']:.3f}")
    print(f"- mean_reciprocal_rank: {summary['mean_reciprocal_rank']:.3f}")
    print(f"- no_match_accuracy: {summary['no_match_accuracy']:.3f}")
    print(f"- false_positive_count: {summary['false_positive_count']}")
    print(
        "- high_confidence_false_positive_count: "
        f"{summary['high_confidence_false_positive_count']}"
    )
    print(f"- distractor_hit_count: {summary['distractor_hit_count']}")
    print()

    for case in summary["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(
            f"[{status}] {case['case_id']} - "
            f"{case['application_title']} "
            f"(rank={case['first_expected_rank']}, "
            f"recommended={case['recommended_count']})"
        )
        if case["false_positive_hashes"]:
            print(f"  false_positive_hashes={case['false_positive_hashes']}")
        if case["high_confidence_false_positive_hashes"]:
            print(
                "  high_confidence_false_positive_hashes="
                f"{case['high_confidence_false_positive_hashes']}"
            )
        if case["distractor_hit_hashes"]:
            print(f"  distractor_hit_hashes={case['distractor_hit_hashes']}")


if __name__ == "__main__":
    sys.exit(main())
