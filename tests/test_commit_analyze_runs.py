from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.domains.commit.schemas import ChangedFile, CommitAnalyzeRequest
from app.domains.commit.services import analyze_runs
from app.domains.commit.services.analyze_runs import (
    create_commit_analyze_run,
    get_commit_analyze_run_status,
    run_commit_analyze_pipeline,
)
from app.main import app


def _build_analyze_payload() -> dict:
    return {
        "commit_id": 4,
        "commit_hash": "cb2222fb915f9dfbd5b22eded57dadd57f225798",
        "repository_id": 1,
        "message": "chore: Swagger 에러 문서화를 위한 어노테이션 추가",
        "changed_file_list": [
            {
                "file_name": "src/main/java/com/whylog/ApiErrorCodeExample.java",
                "changed_code": "+public @interface ApiErrorCodeExample {}",
            }
        ],
    }


@pytest.fixture(autouse=True)
def clear_commit_analyze_runs():
    analyze_runs._runs.clear()
    yield
    analyze_runs._runs.clear()


class TestCommitAnalyzeRunsEndpoint:
    client = TestClient(app)

    def test_create_commit_analyze_run_returns_accepted(self):
        with patch(
            "app.domains.commit.router.run_commit_analyze_pipeline",
            new_callable=AsyncMock,
        ):
            response = self.client.post(
                "/api/commit/analyze/runs",
                json=_build_analyze_payload(),
            )

        assert response.status_code == 202
        body = response.json()
        assert body["isSuccess"] is True
        assert body["code"] == "COMMIT_ANALYZE_202"
        assert body["result"]["status"] == "queued"
        assert body["result"]["phase"] == "queued"
        assert body["result"]["commit_id"] == 4
        assert body["result"]["repository_id"] == 1

    def test_get_missing_commit_analyze_run_returns_404(self):
        response = self.client.get("/api/commit/analyze/runs/not-found")

        assert response.status_code == 404
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_404"


class TestCommitAnalyzeRunService:
    @pytest.mark.asyncio
    async def test_run_pipeline_completes_after_embedding_saved(self):
        request = CommitAnalyzeRequest(
            **_build_analyze_payload(),
        )
        accepted = await create_commit_analyze_run(request)

        with (
            patch(
                "app.domains.commit.services.analyze_runs.summarize_commit",
                new_callable=AsyncMock,
                return_value="Swagger 에러 응답 예시 문서화를 보강했습니다.",
            ) as summarize_mock,
            patch(
                "app.domains.commit.services.analyze_runs.store_commit_embedding",
                new_callable=AsyncMock,
            ) as embedding_mock,
        ):
            await run_commit_analyze_pipeline(accepted.run_id)

        status = await get_commit_analyze_run_status(accepted.run_id)

        assert status is not None
        assert status.status == "completed"
        assert status.phase == "embedding_ready"
        assert status.result is not None
        assert status.result.summary == "Swagger 에러 응답 예시 문서화를 보강했습니다."
        assert status.result.embedding_ready is True
        summarize_mock.assert_awaited_once()
        embedding_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_pipeline_fails_when_all_files_are_filtered(self):
        request = CommitAnalyzeRequest(
            commit_id=5,
            commit_hash="b8fd9ad",
            repository_id=1,
            message="chore: lock 파일 갱신",
            changed_file_list=[
                ChangedFile(
                    file_name="package-lock.json",
                    changed_code="+{}",
                )
            ],
        )
        accepted = await create_commit_analyze_run(request)

        await run_commit_analyze_pipeline(accepted.run_id)

        status = await get_commit_analyze_run_status(accepted.run_id)

        assert status is not None
        assert status.status == "failed"
        assert status.phase == "failed"
        assert status.error == "400: 분석할 수 있는 변경 파일이 없습니다."
