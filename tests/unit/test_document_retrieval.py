from __future__ import annotations

import pytest

from app.retrieval.documents import QdrantDocumentSearch
from app.tools.document_search import DocumentSearchBackendError, DocumentSearchTool
from app.tools.registry import ToolRegistry
from app.tools.schemas import DocumentSearchArguments
from app.vectorstores.documents import DocumentSearchHit


class FakeEmbeddings:
    deployment = "embedding-test"
    dimensions = 3

    def embed(self, texts: object, *, text_version: str) -> list[list[float]]:
        assert text_version == "document_markdown_v1"
        return [[1.0, 0.0, 0.0]]


class FakeStore:
    collection_name = "documents-test"

    def __init__(self, hits: list[DocumentSearchHit]) -> None:
        self.hits = hits

    def search_candidates(self, vector: list[float], *, candidate_limit: int) -> list[DocumentSearchHit]:
        assert vector == [1.0, 0.0, 0.0]
        assert candidate_limit == 20
        return self.hits


def hit(chunk_id: str, score: float) -> DocumentSearchHit:
    return DocumentSearchHit(
        chunk_id=chunk_id,
        score=score,
        payload={
            "document_id": "delivery",
            "filename": "delivery.md",
            "title": "Çatdırılma",
            "heading": "Çatdırılma > Bakı",
            "text": "Çatdırılma bir gün çəkir.",
        },
    )


def test_document_retrieval_applies_threshold_and_builds_trace() -> None:
    backend = QdrantDocumentSearch(
        FakeEmbeddings(),
        FakeStore([hit("delivery:0001", 0.81), hit("delivery:0002", 0.44)]),
        enabled=True,
        ready=True,
        min_score=0.6,
        source_checksum="abc",
        document_count=1,
        chunk_count=2,
    )
    execution = backend.search_with_trace(DocumentSearchArguments(query="çatdırılma"))

    assert execution.result.match_status == "found"
    assert [chunk.chunk_id for chunk in execution.result.chunks] == ["delivery:0001"]
    assert execution.debug_trace["selected_chunk_ids"] == ["delivery:0001"]
    assert execution.debug_trace["candidates"][1]["selected"] is False


def test_document_retrieval_returns_not_found_below_threshold() -> None:
    backend = QdrantDocumentSearch(
        FakeEmbeddings(),
        FakeStore([hit("delivery:0001", 0.4)]),
        enabled=True,
        ready=True,
        min_score=0.6,
        source_checksum="abc",
    )
    result = backend.search_with_trace(DocumentSearchArguments(query="xaricə çatdırılma")).result
    assert result.match_status == "not_found"
    assert result.chunks == []


def test_disabled_document_retrieval_returns_explicit_unavailable_error() -> None:
    backend = QdrantDocumentSearch(
        None,
        None,
        enabled=True,
        ready=False,
        min_score=None,
        source_checksum=None,
        unavailable_reason="baseline_missing",
    )
    with pytest.raises(DocumentSearchBackendError) as captured:
        backend.search_with_trace(DocumentSearchArguments(query="zəmanət"))
    assert captured.value.code == "document_search_unavailable"
    assert captured.value.debug_trace["semantic_state"] == "failed"


@pytest.mark.asyncio
async def test_registry_advertises_and_executes_document_search() -> None:
    backend = QdrantDocumentSearch(
        FakeEmbeddings(),
        FakeStore([hit("delivery:0001", 0.81)]),
        enabled=True,
        ready=True,
        min_score=0.6,
        source_checksum="abc",
    )

    class ProductToolStub:
        name = "product_search"

        def debug_source_state(self) -> dict[str, object]:
            return {"documents": {"configured": False}}

    registry = ToolRegistry(
        ProductToolStub(),  # type: ignore[arg-type]
        1,
        document_search=DocumentSearchTool(backend),
    )
    names = [spec["function"]["name"] for spec in registry.specs()]
    execution = await registry.execute_with_trace("document_search", {"query": "çatdırılma"})

    assert names == ["product_search", "document_search"]
    assert execution.result["match_status"] == "found"
    assert registry.debug_source_state()["documents"]["ready"] is True
