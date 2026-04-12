"""결정사항 임베딩 파이프라인 테스트.

- 텍스트 정규화 단위 테스트 (입력 → 문서 수/내용/ID)
- ChromaDB 저장 + 재호출 upsert 정책 테스트
- API 경로 테스트 (Gemini mock)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppServiceError
from app.domains.decision.schemas import (
    DecisionCard,
    DecisionExtractionResult,
    EmbeddedDocument,
    OverallAnalysis,
)
from app.domains.decision.services.embedding import (
    build_embedding_documents,
    embed_and_store_decisions,
)
from app.main import app

# ── 테스트 픽스처 ──


def _make_result(
    cards: list[DecisionCard] | None = None,
) -> DecisionExtractionResult:
    return DecisionExtractionResult(
        overall_analysis=OverallAnalysis(),
        decision_cards=cards or [],
        other_mentions=[],
    )


SAMPLE_CARD = DecisionCard(
    decision_title="Redis 캐시 도입",
    applied_items=["사용자 세션 캐싱 적용", "API 응답 캐싱 적용"],
    decision_reasons=["DB 부하를 줄여 응답 속도를 개선한다."],
    timeline=[],
)

CARD_NO_ITEMS = DecisionCard(
    decision_title="모니터링 강화",
    applied_items=[],
    decision_reasons=["장애 감지 시간을 단축한다."],
    timeline=[],
)

CARD_NO_REASONS = DecisionCard(
    decision_title="코드 리뷰 필수화",
    applied_items=["PR 리뷰 2명 이상 필수"],
    decision_reasons=[],
    timeline=[],
)


# ── 정규화 단위 테스트 ──


class TestBuildEmbeddingDocuments:
    def test_applied_items_generate_separate_documents(self):
        """applied_item마다 별도 문서가 생성된다."""
        result = _make_result([SAMPLE_CARD])
        docs = build_embedding_documents("mtg-1", result)

        assert len(docs) == 2
        assert docs[0].document_id == "mtg-1_card0_item0"
        assert docs[1].document_id == "mtg-1_card0_item1"

    def test_document_text_contains_title_item_reason(self):
        """임베딩 텍스트에 결정 제목, 적용사항, 근거가 포함된다."""
        result = _make_result([SAMPLE_CARD])
        docs = build_embedding_documents("mtg-1", result)

        assert "title: Redis 캐시 도입 | text:" in docs[0].text
        assert "적용사항: 사용자 세션 캐싱 적용" in docs[0].text
        assert "근거: DB 부하를 줄여 응답 속도를 개선한다." in docs[0].text

    def test_card_without_applied_items_generates_one_document(self):
        """applied_items가 비어있으면 카드당 1개 문서가 생성된다."""
        result = _make_result([CARD_NO_ITEMS])
        docs = build_embedding_documents("mtg-2", result)

        assert len(docs) == 1
        assert docs[0].document_id == "mtg-2_card0_item0"
        assert "text: 적용사항 없음" in docs[0].text
        assert "적용사항:" not in docs[0].text
        assert "근거:" in docs[0].text

    def test_card_without_reasons_omits_reason_tag(self):
        """decision_reasons가 비어있으면 근거 태그가 없다."""
        result = _make_result([CARD_NO_REASONS])
        docs = build_embedding_documents("mtg-3", result)

        assert len(docs) == 1
        assert "근거:" not in docs[0].text

    def test_multiple_cards_index_correctly(self):
        """여러 카드가 있을 때 card index가 올바르게 증가한다."""
        result = _make_result([SAMPLE_CARD, CARD_NO_ITEMS])
        docs = build_embedding_documents("mtg-4", result)

        assert len(docs) == 3
        assert docs[0].document_id == "mtg-4_card0_item0"
        assert docs[1].document_id == "mtg-4_card0_item1"
        assert docs[2].document_id == "mtg-4_card1_item0"

    def test_empty_cards_returns_empty(self):
        """카드가 없으면 빈 리스트를 반환한다."""
        result = _make_result([])
        docs = build_embedding_documents("mtg-5", result)
        assert docs == []

    def test_whitespace_normalization(self):
        """텍스트의 연속 공백이 정규화된다."""
        card = DecisionCard(
            decision_title="  공백   많은   제목  ",
            applied_items=["  공백   많은   항목  "],
            decision_reasons=["  공백   많은   근거  "],
            timeline=[],
        )
        result = _make_result([card])
        docs = build_embedding_documents("mtg-6", result)

        assert "공백 많은 제목" in docs[0].text
        assert "공백 많은 항목" in docs[0].text
        assert "공백 많은 근거" in docs[0].text


# ── ChromaDB 저장 + 재호출 정책 테스트 ──


def _mock_embedding(n: int, dim: int = 768) -> list[list[float]]:
    return [[0.1] * dim for _ in range(n)]


class TestEmbedAndStore:
    @pytest.mark.asyncio
    async def test_stores_documents_in_chroma(self):
        """임베딩 생성 후 ChromaDB에 저장된다."""
        result = _make_result([SAMPLE_CARD])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(2),
            ),
            patch(
                "app.domains.decision.services.embedding.get_decision_collection",
                return_value=mock_collection,
            ),
        ):
            docs = await embed_and_store_decisions("mtg-1", None, result)

        assert len(docs) == 2
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert len(call_kwargs.kwargs["ids"]) == 2

    @pytest.mark.asyncio
    async def test_reprocessing_deletes_then_adds(self):
        """동일 meeting_id 재호출 시 기존 문서를 삭제 후 새로 저장한다."""
        result = _make_result([SAMPLE_CARD])
        mock_collection = MagicMock()
        # 기존에 3개 문서가 있었던 상황
        mock_collection.get.return_value = {
            "ids": ["mtg-1_card0_item0", "mtg-1_card0_item1", "mtg-1_card1_item0"]
        }

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(2),
            ),
            patch(
                "app.domains.decision.services.embedding.get_decision_collection",
                return_value=mock_collection,
            ),
        ):
            docs = await embed_and_store_decisions("mtg-1", None, result)

        # 기존 3개 삭제 확인
        mock_collection.delete.assert_called_once_with(
            ids=["mtg-1_card0_item0", "mtg-1_card0_item1", "mtg-1_card1_item0"]
        )
        # 새로 2개 저장 확인
        assert len(docs) == 2
        mock_collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_result_still_deletes_existing(self):
        """카드가 없어도 기존 문서를 삭제한다 (좀비 방지)."""
        result = _make_result([])
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["mtg-empty_card0_item0"]}

        with patch(
            "app.domains.decision.services.embedding.get_decision_collection",
            return_value=mock_collection,
        ):
            docs = await embed_and_store_decisions("mtg-empty", None, result)

        assert docs == []
        mock_collection.delete.assert_called_once_with(ids=["mtg-empty_card0_item0"])
        mock_collection.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_failure_raises_502(self):
        """기존 문서 조회 실패 시 502 예외를 발생시킨다."""
        result = _make_result([SAMPLE_CARD])
        mock_collection = MagicMock()
        mock_collection.get.side_effect = RuntimeError("query failed")

        with (
            patch(
                "app.domains.decision.services.embedding._generate_embeddings",
                new_callable=AsyncMock,
                return_value=_mock_embedding(2),
            ),
            patch(
                "app.domains.decision.services.embedding.get_decision_collection",
                return_value=mock_collection,
            ),
        ):
            with pytest.raises(AppServiceError) as exc:
                await embed_and_store_decisions("mtg-fail", None, result)

        assert exc.value.status_code == 502
        assert "기존 임베딩 문서 조회에 실패" in exc.value.message


def _make_embedding_payload() -> dict:
    return {
        "meeting_id": "meeting-123",
        "project_id": "proj-abc",
        "decision_result": {
            "decision_cards": [
                {
                    "decision_title": "Redis 캐시 도입",
                    "applied_items": ["세션 캐시 적용"],
                    "decision_reasons": ["응답 속도 개선"],
                    "timeline": [],
                }
            ],
            "other_mentions": [],
        },
    }


class TestDecisionEmbeddingEndpoint:
    client = TestClient(app)

    def test_success_response(self):
        """정상 요청 시 200 응답과 표준 성공 포맷을 반환한다."""
        mock_docs = [
            EmbeddedDocument(
                document_id="meeting-123_card0_item0",
                text=(
                    "title: Redis 캐시 도입 | text: 적용사항: 세션 캐시 적용 "
                    "| 근거: 응답 속도 개선"
                ),
                decision_title="Redis 캐시 도입",
                applied_item="세션 캐시 적용",
            )
        ]

        with patch(
            "app.domains.decision.router.embed_and_store_decisions",
            new_callable=AsyncMock,
            return_value=mock_docs,
        ):
            response = self.client.post(
                "/api/decisions/embeddings",
                json=_make_embedding_payload(),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["isSuccess"] is True
        assert body["code"] == "DECISION_EMBEDDING_200"
        assert body["result"]["total_documents"] == 1
        assert body["result"]["document_ids"] == ["meeting-123_card0_item0"]

    def test_validation_error_response(self):
        """meeting_id 누락 시 422 표준 에러 포맷을 반환한다."""
        invalid_payload = _make_embedding_payload()
        invalid_payload.pop("meeting_id")

        response = self.client.post(
            "/api/decisions/embeddings",
            json=invalid_payload,
        )

        assert response.status_code == 422
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_422"

    def test_service_failure_response(self):
        """서비스 레이어 예외 발생 시 502 표준 에러 포맷을 반환한다."""
        with patch(
            "app.domains.decision.router.embed_and_store_decisions",
            new_callable=AsyncMock,
            side_effect=AppServiceError(
                "Gemini 임베딩 생성 실패",
                status_code=502,
            ),
        ):
            response = self.client.post(
                "/api/decisions/embeddings",
                json=_make_embedding_payload(),
            )

        assert response.status_code == 502
        body = response.json()
        assert body["isSuccess"] is False
        assert body["code"] == "COMMON_502"

    def test_empty_decision_cards_response(self):
        """결정 카드가 비어도 200과 문서 수 0을 반환한다."""
        with patch(
            "app.domains.decision.router.embed_and_store_decisions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            payload = _make_embedding_payload()
            payload["decision_result"]["decision_cards"] = []
            response = self.client.post(
                "/api/decisions/embeddings",
                json=payload,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["isSuccess"] is True
        assert body["result"]["total_documents"] == 0
