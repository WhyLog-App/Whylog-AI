from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.background import BackgroundTasks

from app.core.errors import AppServiceError
from app.domains.commit.router import analyze_commit
from app.domains.commit.schemas import (
    ApplicationCommitMatchItem,
    ApplicationCommitMatchRequest,
    ApplicationCommitMatchResponse,
    ChangedFile,
    CommitAnalyzeRequest,
)
from app.domains.commit.services.matching import (
    _to_application_entries,
    match_applications_with_commits,
)
from app.domains.commit.services.summarize import generate_embedding_text
from app.main import app


def _build_match_payload() -> dict:
    return {
        "meeting_id": "meeting-123",
        "repository_ids": [1],
        "top_k": 5,
    }


class TestApplicationCommitMatchingService:
    def test_application_entries_accept_chroma_numpy_embeddings(self):
        np = pytest.importorskip("numpy")
        raw = {
            "ids": ["meeting-123_application0"],
            "documents": ["title: Redis 도입 | text: 적용사항: redis 도입"],
            "metadatas": [
                {
                    "application_id": 101,
                    "application_title": "redis 도입",
                }
            ],
            "embeddings": np.array([[0.1, 0.2, 0.3]]),
        }

        entries = _to_application_entries(raw)

        assert entries[0].embedding == [0.1, 0.2, 0.3]
        assert entries[0].application_id == 101

    def test_application_entries_missing_application_id_returns_none(self):
        raw = {
            "ids": ["meeting-123_application0"],
            "documents": ["title: Redis 도입 | text: 적용사항: redis 도입"],
            "metadatas": [
                {
                    "application_id": "",
                    "application_title": "redis 도입",
                }
            ],
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        entries = _to_application_entries(raw)

        assert entries[0].application_id is None

    @pytest.mark.asyncio
    async def test_returns_recommended_commits_sorted_by_confidence(self):
        application_collection = MagicMock()
        application_collection.get.return_value = {
            "ids": ["meeting-123_application0"],
            "documents": [
                (
                    "title: 알림 시스템 개선 | text: 적용사항: "
                    "notification queue redis "
                    "kafka rabbitmq 도입 | 근거: 처리 지연 감소"
                )
            ],
            "metadatas": [
                {
                    "application_id": 101,
                    "application_title": "notification queue redis kafka rabbitmq 도입",
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
                        "title: repository-1 feat | text: 변경요약: redis queue 도입 "
                        "| 기술키워드: redis,kafka,rabbitmq | 변경방향: add "
                        "| 파일맥락: notification,queue"
                    ),
                    (
                        "title: repository-1 feat | text: 변경요약: "
                        "redis kafka 설정 추가 "
                        "| 기술키워드: redis,kafka | 변경방향: add "
                        "| 파일맥락: notification,queue"
                    ),
                    (
                        "title: repository-1 fix | text: 변경요약: billing 버그 수정 "
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
                        "repository_id": 1,
                        "direction_primary": "add",
                        "direction_multi_csv": "add",
                        "tech_keywords_csv": "redis,kafka,rabbitmq",
                        "module_tags_csv": "notification,queue",
                        "commit_message": "feat: introduce redis queue",
                    },
                    {
                        "commit_ref": "c2",
                        "commit_hash": "h2",
                        "repository_id": 1,
                        "direction_primary": "add",
                        "direction_multi_csv": "add",
                        "tech_keywords_csv": "redis,kafka",
                        "module_tags_csv": "notification,queue",
                        "commit_message": "feat: add redis config",
                    },
                    {
                        "commit_ref": "c3",
                        "commit_hash": "h3",
                        "repository_id": 1,
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
                "app.domains.commit.services.matching.get_application_collection",
                return_value=application_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_applications_with_commits(
                ApplicationCommitMatchRequest(**_build_match_payload())
            )

        assert result.total_applications == 1
        assert result.matched_applications == 1
        item = result.applications[0]
        assert item.application_id == 101
        assert len(item.recommended_commits) == 2
        assert item.recommended_commits[0].commit_id == 1
        assert item.recommended_commits[0].commit_hash == "h1"
        assert item.recommended_commits[0].commit_message == (
            "feat: introduce redis queue"
        )
        assert item.recommended_commits[0].confidence >= 70
        assert item.recommended_commits[0].reason.endswith(".")
        assert "총" in item.recommended_commits[0].score_detail
        assert "겹친 키워드" in item.recommended_commits[0].score_detail
        assert item.recommended_commits[1].commit_hash == "h2"
        assert item.recommended_commits[0].confidence >= (
            item.recommended_commits[1].confidence
        )
        commit_collection.query.assert_called_once()
        assert commit_collection.query.call_args.kwargs["where"] == {"repository_id": 1}

    @pytest.mark.asyncio
    async def test_multiple_repository_ids_query_with_in_filter(self):
        application_collection = MagicMock()
        application_collection.get.return_value = {
            "ids": ["meeting-123_application0"],
            "documents": ["title: Redis 도입 | text: 적용사항: redis 도입"],
            "metadatas": [{"application_title": "redis 도입"}],
            "embeddings": [[0.1, 0.2]],
        }

        commit_collection = MagicMock()
        commit_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        payload = _build_match_payload()
        payload["repository_ids"] = [1, 2, 3]

        with (
            patch(
                "app.domains.commit.services.matching.get_application_collection",
                return_value=application_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            await match_applications_with_commits(
                ApplicationCommitMatchRequest(**payload)
            )

        commit_collection.query.assert_called_once()
        assert commit_collection.query.call_args.kwargs["where"] == {
            "repository_id": {"$in": [1, 2, 3]}
        }

    @pytest.mark.asyncio
    async def test_filters_candidate_outside_requested_repository_ids(self):
        application_collection = MagicMock()
        application_collection.get.return_value = {
            "ids": ["meeting-123_application0"],
            "documents": [
                "title: Redis 도입 | text: 적용사항: notification redis 도입"
            ],
            "metadatas": [{"application_title": "notification redis 도입"}],
            "embeddings": [[0.1, 0.2]],
        }

        commit_collection = MagicMock()
        commit_collection.query.return_value = {
            "ids": [["commit_99"]],
            "documents": [
                [
                    (
                        "변경요약: redis queue 도입 | 기술키워드: redis "
                        "| 변경방향: add | 파일맥락: notification"
                    )
                ]
            ],
            "metadatas": [
                [
                    {
                        "commit_ref": "c99",
                        "commit_hash": "h99",
                        "repository_id": 99,
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
                "app.domains.commit.services.matching.get_application_collection",
                return_value=application_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_applications_with_commits(
                ApplicationCommitMatchRequest(**_build_match_payload())
            )

        assert result.matched_applications == 0
        assert result.applications[0].recommended_commits == []

    @pytest.mark.asyncio
    async def test_opposite_direction_candidate_is_not_matched(self):
        application_collection = MagicMock()
        application_collection.get.return_value = {
            "ids": ["meeting-123_application0"],
            "documents": [
                "title: Redis 제거 | text: 적용사항: notification redis 제거"
            ],
            "metadatas": [
                {
                    "application_title": "notification redis 제거",
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
                        "repository_id": 1,
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
                "app.domains.commit.services.matching.get_application_collection",
                return_value=application_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_applications_with_commits(
                ApplicationCommitMatchRequest(**_build_match_payload())
            )

        item = result.applications[0]
        assert item.recommended_commits == []

    @pytest.mark.asyncio
    async def test_same_commit_is_removed_from_opposite_application(self):
        application_collection = MagicMock()
        application_collection.get.return_value = {
            "ids": ["meeting-123_application0", "meeting-123_application1"],
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
                    "application_title": "notification redis kafka rabbitmq 도입",
                },
                {
                    "application_title": "notification redis kafka rabbitmq 제거",
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
                            "repository_id": 1,
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
                            "repository_id": 1,
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
                "app.domains.commit.services.matching.get_application_collection",
                return_value=application_collection,
            ),
            patch(
                "app.domains.commit.services.matching.get_commit_collection",
                return_value=commit_collection,
            ),
        ):
            result = await match_applications_with_commits(
                ApplicationCommitMatchRequest(**_build_match_payload())
            )

        first_item = result.applications[0]
        second_item = result.applications[1]
        assert len(first_item.recommended_commits) == 1
        assert second_item.recommended_commits == []


class TestApplicationCommitMatchingEndpoint:
    client = TestClient(app)

    def test_match_endpoint_success(self):
        mock_result = ApplicationCommitMatchResponse(
            meeting_id="meeting-123",
            repository_ids=[1],
            total_applications=1,
            matched_applications=1,
            applications=[
                ApplicationCommitMatchItem(
                    application_id=101,
                    application_document_id="meeting-123_application0",
                    application_title="세션 캐시 적용",
                    recommended_commits=[],
                )
            ],
        )

        with patch(
            "app.domains.commit.router.match_applications_with_commits",
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
        assert body["result"]["repository_ids"] == [1]
        assert body["result"]["applications"][0]["application_id"] == 101

    def test_match_endpoint_echoes_multiple_repository_ids(self):
        mock_result = ApplicationCommitMatchResponse(
            meeting_id="meeting-123",
            repository_ids=[1, 2, 3],
            total_applications=0,
            matched_applications=0,
            applications=[],
        )
        payload = _build_match_payload()
        payload["repository_ids"] = [1, 2, 3]

        with patch(
            "app.domains.commit.router.match_applications_with_commits",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = self.client.post("/api/commit/match", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["repository_ids"] == [1, 2, 3]

    def test_match_endpoint_validation_error(self):
        payload = _build_match_payload()
        payload["top_k"] = 0

        response = self.client.post("/api/commit/match", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_match_endpoint_requires_repository_ids(self):
        payload = _build_match_payload()
        payload["repository_ids"] = []

        response = self.client.post("/api/commit/match", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_match_endpoint_requires_repository_ids_key(self):
        payload = _build_match_payload()
        del payload["repository_ids"]

        response = self.client.post("/api/commit/match", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_match_endpoint_service_failure(self):
        with patch(
            "app.domains.commit.router.match_applications_with_commits",
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
            repository_id=1,
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
        assert task.kwargs["commit_hash"] == "b8fd9ad"
        assert task.kwargs["repository_id"] == 1
        assert task.kwargs["commit_id"] == 1

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
                commit_hash="b8fd9ad",
                repository_id=1,
                message="feat: API 구현",
                changed_file_list=[
                    ChangedFile(
                        file_name="app/domains/api.py",
                        changed_code="+def handler():\n+    return True",
                    )
                ],
                commit_id=1,
            )

        _, kwargs = collection.upsert.call_args
        assert kwargs["metadatas"][0]["commit_id"] == 1
        assert kwargs["metadatas"][0]["commit_hash"] == "b8fd9ad"
        assert kwargs["metadatas"][0]["commit_message"] == "feat: API 구현"
        assert kwargs["metadatas"][0]["repository_id"] == 1
