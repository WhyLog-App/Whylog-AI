from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.background import BackgroundTasks

from app.core.errors import AppServiceError
from app.domains.commit.router import analyze_commit
from app.domains.commit.schemas import (
    ChangedFile,
    CommitAnalyzeRequest,
    DecisionCommitMatchRequest,
    DecisionCommitMatchResponse,
)
from app.domains.commit.services.matching import (
    _to_decision_entries,
    match_decisions_with_commits,
)
from app.domains.commit.services.summarize import generate_embedding_text
from app.main import app


def _build_match_payload() -> dict:
    return {
        "meeting_id": "meeting-123",
        "project_id": "project-abc",
        "repository": "whylog/web",
        "top_k": 5,
    }


class TestDecisionCommitMatchingService:
    def test_decision_entries_accept_chroma_numpy_embeddings(self):
        np = pytest.importorskip("numpy")
        raw = {
            "ids": ["meeting-123_card0_item0"],
            "documents": ["title: Redis 도입 | text: 적용사항: redis 도입"],
            "metadatas": [
                {
                    "decision_title": "Redis 도입",
                    "applied_item": "redis 도입",
                }
            ],
            "embeddings": np.array([[0.1, 0.2, 0.3]]),
        }

        entries = _to_decision_entries(raw)

        assert entries[0].embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_matches_applied_and_partial_commits(self):
        decision_collection = MagicMock()
        decision_collection.get.return_value = {
            "ids": ["meeting-123_card0_item0"],
            "documents": [
                (
                    "title: 알림 시스템 개선 | text: 적용사항: "
                    "notification queue redis "
                    "kafka rabbitmq 도입 | 근거: 처리 지연 감소"
                )
            ],
            "metadatas": [
                {
                    "decision_title": "알림 시스템 개선",
                    "applied_item": "notification queue redis kafka rabbitmq 도입",
                }
            ],
            "embeddings": [[0.11, 0.22, 0.33]],
        }

        commit_collection = MagicMock()
        commit_collection.query.return_value = {
            "ids": [["commit_1", "commit_2", "commit_3"]],
            "documents": [
                [
                    (
                        "title: whylog/web feat | text: 변경요약: redis queue 도입 "
                        "| 기술키워드: redis,kafka,rabbitmq | 변경방향: add "
                        "| 파일맥락: notification,queue"
                    ),
                    (
                        "title: whylog/web feat | text: 변경요약: "
                        "redis kafka 설정 추가 "
                        "| 기술키워드: redis,kafka | 변경방향: add "
                        "| 파일맥락: notification,queue"
                    ),
                    (
                        "title: whylog/web fix | text: 변경요약: billing 버그 수정 "
                        "| 기술키워드: billing | 변경방향: modify | 파일맥락: billing"
                    ),
                ]
            ],
            "metadatas": [
                [
                    {
                        "commit_ref": "c1",
                        "commit_id": 1,
                        "commit_hash": "h1",
                        "repository": "whylog/web",
                        "direction_primary": "add",
                        "direction_multi_csv": "add",
                        "tech_keywords_csv": "redis,kafka,rabbitmq",
                        "module_tags_csv": "notification,queue",
                        "commit_message": "feat: introduce redis queue",
                    },
                    {
                        "commit_ref": "c2",
                        "commit_hash": "h2",
                        "repository": "whylog/web",
                        "direction_primary": "add",
                        "direction_multi_csv": "add",
                        "tech_keywords_csv": "redis,kafka",
                        "module_tags_csv": "notification,queue",
                        "commit_message": "feat: add redis config",
                    },
                    {
                        "commit_ref": "c3",
                        "commit_hash": "h3",
                        "repository": "whylog/web",
                        "direction_primary": "modify",
                        "direction_multi_csv": "modify",
                        "tech_keywords_csv": "billing",
                        "module_tags_csv": "billing",
                        "commit_message": "fix billing",
                    },
                ]
            ],
            "distances": [[0.02, 0.78, 0.90]],
        }

        with (
            patch(
                "app.domains.commit.services.matching.get_decision_collection",
                return_value=decision_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_decisions_with_commits(
                DecisionCommitMatchRequest(**_build_match_payload())
            )

        assert result.total_decision_items == 1
        assert result.matched_decision_items == 1
        item = result.decisions[0]
        assert item.decision_status == "APPLIED"
        assert len(item.connected_commits) == 1
        assert len(item.recommended_commits) == 1
        assert item.connected_commits[0].commit_id == 1
        assert item.connected_commits[0].commit_hash == "h1"
        assert item.connected_commits[0].confidence >= 70
        assert item.recommended_commits[0].status == "PARTIAL"

    @pytest.mark.asyncio
    async def test_opposite_direction_candidate_is_not_matched(self):
        decision_collection = MagicMock()
        decision_collection.get.return_value = {
            "ids": ["meeting-123_card0_item0"],
            "documents": [
                "title: Redis 제거 | text: 적용사항: notification redis 제거"
            ],
            "metadatas": [
                {
                    "decision_title": "Redis 제거",
                    "applied_item": "notification redis 제거",
                }
            ],
            "embeddings": [[0.1, 0.2]],
        }

        commit_collection = MagicMock()
        commit_collection.query.return_value = {
            "ids": [["commit_1"]],
            "documents": [
                ["변경요약: redis 도입 | 기술키워드: redis | 파일맥락: notification"]
            ],
            "metadatas": [
                [
                    {
                        "commit_ref": "c1",
                        "commit_hash": "h1",
                        "repository": "whylog/web",
                        "direction_primary": "add",
                        "direction_multi_csv": "add",
                        "tech_keywords_csv": "redis",
                        "module_tags_csv": "notification",
                        "commit_message": "feat: introduce redis",
                    }
                ]
            ],
            "distances": [[0.01]],
        }

        with (
            patch(
                "app.domains.commit.services.matching.get_decision_collection",
                return_value=decision_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_decisions_with_commits(
                DecisionCommitMatchRequest(**_build_match_payload())
            )

        item = result.decisions[0]
        assert item.decision_status == "UNAPPLIED"
        assert item.connected_commits == []
        assert item.recommended_commits == []

    @pytest.mark.asyncio
    async def test_same_commit_is_removed_from_opposite_decision(self):
        decision_collection = MagicMock()
        decision_collection.get.return_value = {
            "ids": ["meeting-123_card0_item0", "meeting-123_card1_item0"],
            "documents": [
                (
                    "title: Redis 도입 | text: 적용사항: notification redis kafka "
                    "rabbitmq 도입"
                ),
                (
                    "title: Redis 제거 | text: 적용사항: notification redis kafka "
                    "rabbitmq 제거"
                ),
            ],
            "metadatas": [
                {
                    "decision_title": "Redis 도입",
                    "applied_item": "notification redis kafka rabbitmq 도입",
                },
                {
                    "decision_title": "Redis 제거",
                    "applied_item": "notification redis kafka rabbitmq 제거",
                },
            ],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        commit_collection = MagicMock()
        commit_collection.query.side_effect = [
            {
                "ids": [["commit_shared"]],
                "documents": [
                    [
                        (
                            "변경요약: redis queue 도입 | 기술키워드: redis,kafka,"
                            "rabbitmq | 파일맥락: notification,queue"
                        )
                    ]
                ],
                "metadatas": [
                    [
                        {
                            "commit_ref": "c-shared",
                            "commit_hash": "h-shared",
                            "repository": "whylog/web",
                            "direction_primary": "add",
                            "direction_multi_csv": "add",
                            "tech_keywords_csv": "redis,kafka,rabbitmq",
                            "module_tags_csv": "notification,queue",
                            "commit_message": "feat: introduce redis queue",
                        }
                    ]
                ],
                "distances": [[0.05]],
            },
            {
                "ids": [["commit_shared"]],
                "documents": [
                    [
                        (
                            "변경요약: redis queue 도입 | 기술키워드: redis,kafka,"
                            "rabbitmq | 파일맥락: notification,queue"
                        )
                    ]
                ],
                "metadatas": [
                    [
                        {
                            "commit_ref": "c-shared",
                            "commit_hash": "h-shared",
                            "repository": "whylog/web",
                            "direction_primary": "add",
                            "direction_multi_csv": "add",
                            "tech_keywords_csv": "redis,kafka,rabbitmq",
                            "module_tags_csv": "notification,queue",
                            "commit_message": "feat: introduce redis queue",
                        }
                    ]
                ],
                "distances": [[0.20]],
            },
        ]

        with (
            patch(
                "app.domains.commit.services.matching.get_decision_collection",
                return_value=decision_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_decisions_with_commits(
                DecisionCommitMatchRequest(**_build_match_payload())
            )

        first_item = result.decisions[0]
        second_item = result.decisions[1]
        assert first_item.decision_status == "APPLIED"
        assert len(first_item.connected_commits) == 1
        assert second_item.decision_status == "UNAPPLIED"
        assert second_item.connected_commits == []
        assert second_item.recommended_commits == []


class TestDecisionCommitMatchingEndpoint:
    client = TestClient(app)

    def test_match_endpoint_success(self):
        mock_result = DecisionCommitMatchResponse(
            meeting_id="meeting-123",
            project_id="project-abc",
            repository="whylog/web",
            total_decision_items=1,
            matched_decision_items=1,
            decisions=[],
        )

        with patch(
            "app.domains.commit.router.match_decisions_with_commits",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = self.client.post(
                "/api/commit/match",
                json=_build_match_payload(),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["isSuccess"] is True
        assert body["code"] == "COMMIT_MATCH_200"
        assert body["result"]["meeting_id"] == "meeting-123"

    def test_match_endpoint_validation_error(self):
        payload = _build_match_payload()
        payload["top_k"] = 0

        response = self.client.post("/api/commit/match", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_match_endpoint_service_failure(self):
        with patch(
            "app.domains.commit.router.match_decisions_with_commits",
            new_callable=AsyncMock,
            side_effect=AppServiceError("커밋 후보 조회 실패", status_code=502),
        ):
            response = self.client.post(
                "/api/commit/match",
                json=_build_match_payload(),
            )

        assert response.status_code == 502
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_502"


class TestCommitAnalyzeHashMetadata:
    @pytest.mark.asyncio
    async def test_analyze_commit_passes_commit_hash_to_background_embedding(self):
        background_tasks = BackgroundTasks()
        request = CommitAnalyzeRequest(
            commit_id=1,
            commit_hash="b8fd9ad",
            repository="whylog/web",
            message="feat: API 구현",
            changed_file_list=[
                ChangedFile(
                    file_name="app/domains/api.py",
                    changed_code="+def handler():\n+    return True",
                )
            ],
        )

        with patch(
            "app.domains.commit.router.summarize_commit",
            new_callable=AsyncMock,
            return_value="API를 구현했습니다.",
        ):
            response = await analyze_commit(request, background_tasks)

        assert response.result.commit_id == 1
        task = background_tasks.tasks[0]
        assert task.args[0] == 1
        assert task.args[1] == "b8fd9ad"

    @pytest.mark.asyncio
    async def test_generate_embedding_text_stores_commit_hash_metadata(self):
        collection = MagicMock()

        with (
            patch(
                "app.domains.commit.services.summarize._get_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.domains.commit.services.summarize._call_gemini",
                new_callable=AsyncMock,
                return_value=(
                    "변경요약: API 엔드포인트를 추가했습니다.\n"
                    "기술키워드: fastapi\n"
                    "변경방향: add\n"
                    "파일맥락: commit"
                ),
            ),
            patch(
                "app.domains.commit.services.summarize._generate_embedding",
                new_callable=AsyncMock,
                return_value=[0.1, 0.2, 0.3],
            ),
            patch(
                "app.domains.commit.services.summarize.get_commit_collection",
                return_value=collection,
            ),
        ):
            await generate_embedding_text(
                1,
                "b8fd9ad",
                "whylog/web",
                "feat: API 구현",
                [
                    ChangedFile(
                        file_name="app/domains/api.py",
                        changed_code="+def handler():\n+    return True",
                    )
                ],
            )

        _, kwargs = collection.upsert.call_args
        assert kwargs["metadatas"][0]["commit_id"] == 1
        assert kwargs["metadatas"][0]["commit_hash"] == "b8fd9ad"
