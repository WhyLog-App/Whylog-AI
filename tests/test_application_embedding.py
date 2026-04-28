"""적용사항 임베딩 파이프라인 테스트."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppServiceError
from app.domains.decision.schemas import (
    Application,
    EmbeddedDocument,
    MeetingAnalysis,
    MeetingAnalysisResult,
)
from app.domains.decision.services.embedding import (
    build_embedding_documents,
    embed_and_store_applications,
)
from app.main import app


def _make_result(
    applications: list[Application] | None = None,
) -> MeetingAnalysisResult:
    return MeetingAnalysisResult(
        overall_analysis=MeetingAnalysis(),
        applications=applications or [],
        other_mentions=[],
    )


SAMPLE_APPLICATION = Application(
    application_id=101,
    application_title="사용자 세션 캐싱 적용",
    application_reasons=["DB 부하를 줄여 응답 속도를 개선한다."],
    timeline=[],
)

APPLICATION_NO_REASONS = Application(
    application_title="PR 리뷰 2명 이상 필수",
    application_reasons=[],
    timeline=[],
)

APPLICATION_MULTI_REASONS = Application(
    application_title="캐시 만료 정책 적용",
    application_reasons=[
        "응답 지연을 줄인다.",
        "DB 부하를 줄인다.",
        "트래픽 급증 시 안정성을 높인다.",
    ],
    timeline=[],
)


class TestBuildEmbeddingDocuments:
    def test_applications_generate_separate_documents(self):
        """application마다 별도 문서가 생성된다."""
        result = _make_result(
            [
                SAMPLE_APPLICATION,
                Application(
                    application_title="API 응답 캐싱 적용",
                    application_reasons=["반복 조회 비용을 줄인다."],
                    timeline=[],
                ),
            ]
        )
        docs = build_embedding_documents("mtg-1", result)

        assert len(docs) == 2
        assert docs[0].document_id == "mtg-1_application0"
        assert docs[1].document_id == "mtg-1_application1"

    def test_document_text_contains_title_and_reason(self):
        """임베딩 텍스트에 적용사항 제목과 근거가 포함된다."""
        result = _make_result([SAMPLE_APPLICATION])
        docs = build_embedding_documents("mtg-1", result)

        assert "title: 사용자 세션 캐싱 적용 | text:" in docs[0].text
        assert "적용사항: 사용자 세션 캐싱 적용" in docs[0].text
        assert "근거: DB 부하를 줄여 응답 속도를 개선한다." in docs[0].text

    def test_application_without_reasons_omits_reason_tag(self):
        """application_reasons가 비어있으면 근거 태그가 없다."""
        result = _make_result([APPLICATION_NO_REASONS])
        docs = build_embedding_documents("mtg-3", result)

        assert len(docs) == 1
        assert "근거:" not in docs[0].text

    def test_empty_applications_returns_empty(self):
        """적용사항이 없으면 빈 리스트를 반환한다."""
        result = _make_result([])
        docs = build_embedding_documents("mtg-5", result)
        assert docs == []

    def test_whitespace_normalization(self):
        """텍스트의 연속 공백이 정규화된다."""
        application = Application(
            application_title="  공백   많은   제목  ",
            application_reasons=["  공백   많은   근거  "],
            timeline=[],
        )
        result = _make_result([application])
        docs = build_embedding_documents("mtg-6", result)

        assert "공백 많은 제목" in docs[0].text
        assert "공백 많은 근거" in docs[0].text

    def test_reasons_include_all_reasons(self):
        """근거는 전체를 중복 제거 후 반영한다."""
        result = _make_result([APPLICATION_MULTI_REASONS])
        docs = build_embedding_documents("mtg-all", result)

        assert len(docs) == 1
        assert "근거: 응답 지연을 줄인다." in docs[0].text
        assert "DB 부하를 줄인다." in docs[0].text
        assert "트래픽 급증 시 안정성을 높인다." in docs[0].text


def _mock_embedding(n: int, dim: int = 768) -> list[list[float]]:
    return [[0.1] * dim for _ in range(n)]


class TestEmbedAndStore:
    @pytest.mark.asyncio
    async def test_stores_documents_in_chroma(self):
        """임베딩 생성 후 ChromaDB에 저장된다."""
        result = _make_result([SAMPLE_APPLICATION])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(1),
            ),
            patch(
                "app.domains.decision.services.embedding.get_application_collection",
                return_value=mock_collection,
            ),
        ):
            docs = await embed_and_store_applications("mtg-1", None, result)

        assert len(docs) == 1
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs.kwargs["ids"] == ["mtg-1_application0"]
        assert call_kwargs.kwargs["metadatas"][0]["application_id"] == 101
        assert call_kwargs.kwargs["metadatas"][0]["application_title"] == (
            "사용자 세션 캐싱 적용"
        )

    @pytest.mark.asyncio
    async def test_missing_application_id_is_stored_as_empty_metadata(self, caplog):
        """application_id 미전달 시 Chroma metadata에는 빈 문자열로 저장한다."""
        application = Application(
            application_title="세션 캐시 적용",
            application_reasons=["응답 속도 개선"],
            timeline=[],
        )
        result = _make_result([application])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(1),
            ),
            patch(
                "app.domains.decision.services.embedding.get_application_collection",
                return_value=mock_collection,
            ),
            caplog.at_level("WARNING"),
        ):
            docs = await embed_and_store_applications("mtg-no-id", None, result)

        assert docs[0].application_id is None
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs.kwargs["metadatas"][0]["application_id"] == ""
        assert "Application IDs missing for 1 embedding documents" in caplog.text

    @pytest.mark.asyncio
    async def test_reprocessing_deletes_then_adds(self):
        """동일 meeting_id 재호출 시 기존 문서를 삭제 후 새로 저장한다."""
        result = _make_result([SAMPLE_APPLICATION])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["mtg-1_application0", "mtg-1_application1"]
        }

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(1),
            ),
            patch(
                "app.domains.decision.services.embedding.get_application_collection",
                return_value=mock_collection,
            ),
        ):
            docs = await embed_and_store_applications("mtg-1", None, result)

        mock_collection.delete.assert_called_once_with(
            ids=["mtg-1_application0", "mtg-1_application1"]
        )
        assert len(docs) == 1
        mock_collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_result_still_deletes_existing(self):
        """적용사항이 없어도 기존 문서를 삭제한다."""
        result = _make_result([])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["mtg-empty_application0"]}

        with patch(
            "app.domains.decision.services.embedding.get_application_collection",
            return_value=mock_collection,
        ):
            docs = await embed_and_store_applications("mtg-empty", None, result)

        assert docs == []
        mock_collection.delete.assert_called_once_with(ids=["mtg-empty_application0"])
        mock_collection.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_failure_raises_502(self):
        """기존 문서 조회 실패 시 502 예외를 발생시킨다."""
        result = _make_result([SAMPLE_APPLICATION])
        mock_collection = MagicMock()
        mock_collection.get.side_effect = RuntimeError("query failed")

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(1),
            ),
            patch(
                "app.domains.decision.services.embedding.get_application_collection",
                return_value=mock_collection,
            ),
        ):
            with pytest.raises(AppServiceError) as exc:
                await embed_and_store_applications("mtg-fail", None, result)

        assert exc.value.status_code == 502
        assert "기존 임베딩 문서 조회에 실패" in exc.value.message


def _make_embedding_payload() -> dict:
    return {
        "meeting_id": "meeting-123",
        "project_id": "proj-abc",
        "analysis_result": {
            "applications": [
                {
                    "application_id": 101,
                    "application_title": "세션 캐시 적용",
                    "application_reasons": ["응답 속도 개선"],
                    "timeline": [],
                }
            ],
            "other_mentions": [],
        },
    }


class TestApplicationEmbeddingEndpoint:
    client = TestClient(app)

    def test_success_response(self):
        """정상 요청 시 200 응답과 표준 성공 포맷을 반환한다."""
        mock_docs = [
            EmbeddedDocument(
                document_id="meeting-123_application0",
                text=(
                    "title: 세션 캐시 적용 | text: 적용사항: 세션 캐시 적용 "
                    "| 근거: 응답 속도 개선"
                ),
                application_id=101,
                application_title="세션 캐시 적용",
            )
        ]

        with patch(
            "app.domains.decision.router.embed_and_store_applications",
            new_callable=AsyncMock,
            return_value=mock_docs,
        ):
            response = self.client.post(
                "/api/meeting-analysis/embeddings",
                json=_make_embedding_payload(),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["isSuccess"] is True
        assert body["code"] == "APPLICATION_EMBEDDING_200"
        assert body["result"]["total_documents"] == 1
        assert body["result"]["document_ids"] == ["meeting-123_application0"]
        assert body["result"]["documents"][0]["application_id"] == 101
        assert body["result"]["documents"][0]["application_title"] == "세션 캐시 적용"

    def test_validation_error_response(self):
        """meeting_id 누락 시 422 표준 에러 포맷을 반환한다."""
        invalid_payload = _make_embedding_payload()
        invalid_payload.pop("meeting_id")

        response = self.client.post(
            "/api/meeting-analysis/embeddings",
            json=invalid_payload,
        )

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_service_failure_response(self):
        """서비스 레이어 예외 발생 시 502 표준 에러 포맷을 반환한다."""
        with patch(
            "app.domains.decision.router.embed_and_store_applications",
            new_callable=AsyncMock,
            side_effect=AppServiceError(
                "Gemini 임베딩 생성 실패",
                status_code=502,
            ),
        ):
            response = self.client.post(
                "/api/meeting-analysis/embeddings",
                json=_make_embedding_payload(),
            )

        assert response.status_code == 502
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_502"

    def test_empty_applications_response(self):
        """적용사항이 비어도 200과 문서 수 0을 반환한다."""
        with patch(
            "app.domains.decision.router.embed_and_store_applications",
            new_callable=AsyncMock,
            return_value=[],
        ):
            payload = _make_embedding_payload()
            payload["analysis_result"]["applications"] = []
            response = self.client.post(
                "/api/meeting-analysis/embeddings",
                json=payload,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["isSuccess"] is True
        assert body["result"]["total_documents"] == 0
