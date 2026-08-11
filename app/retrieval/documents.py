from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.documents.corpus import DOCUMENT_TEXT_VERSION
from app.tools.document_search import DocumentSearchBackendError
from app.tools.schemas import (
    DocumentSearchArguments,
    DocumentSearchChunk,
    DocumentSearchResult,
)
from app.vectorstores.documents import DocumentSearchHit

logger = logging.getLogger(__name__)
DOCUMENT_QUERY_CANDIDATES = 20


class DocumentEmbeddingBackend(Protocol):
    deployment: str
    dimensions: int

    def embed(self, texts: Sequence[str], *, text_version: str) -> list[list[float]]: ...


class DocumentVectorBackend(Protocol):
    collection_name: str

    def search_candidates(
        self,
        vector: list[float],
        *,
        candidate_limit: int,
    ) -> list[DocumentSearchHit]: ...


@dataclass(frozen=True)
class DocumentRetrievalExecution:
    result: DocumentSearchResult
    debug_trace: dict[str, Any]


class QdrantDocumentSearch:
    def __init__(
        self,
        embeddings: DocumentEmbeddingBackend | None,
        store: DocumentVectorBackend | None,
        *,
        enabled: bool,
        ready: bool,
        min_score: float | None,
        source_checksum: str | None,
        document_count: int = 0,
        chunk_count: int = 0,
        unavailable_reason: str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.enabled = enabled
        self.ready = ready
        self.min_score = min_score
        self.source_checksum = source_checksum
        self.document_count = document_count
        self.chunk_count = chunk_count
        self.unavailable_reason = unavailable_reason

    @property
    def semantic_enabled(self) -> bool:
        return bool(
            self.enabled
            and self.ready
            and self.embeddings is not None
            and self.store is not None
            and self.min_score is not None
        )

    def search_with_trace(
        self,
        arguments: DocumentSearchArguments,
    ) -> DocumentRetrievalExecution:
        if not self.semantic_enabled:
            raise self._unavailable(arguments, self.unavailable_reason or "not_configured")
        assert self.embeddings is not None
        assert self.store is not None
        assert self.min_score is not None
        try:
            vector = self.embeddings.embed(
                [arguments.query],
                text_version=DOCUMENT_TEXT_VERSION,
            )[0]
            candidates = self.store.search_candidates(
                vector,
                candidate_limit=DOCUMENT_QUERY_CANDIDATES,
            )
        except Exception as exc:
            logger.warning(
                "document_search.qdrant_failed",
                extra={"error_type": type(exc).__name__},
            )
            raise self._unavailable(arguments, type(exc).__name__) from exc

        selected = [hit for hit in candidates if hit.score >= self.min_score][: arguments.limit]
        chunks = [self._chunk(hit) for hit in selected]
        result = DocumentSearchResult(
            match_status="found" if chunks else "not_found",
            total=len(chunks),
            min_score=self.min_score,
            chunks=chunks,
        )
        return DocumentRetrievalExecution(
            result=result,
            debug_trace=self._trace(arguments, candidates, selected),
        )

    def debug_source_state(self) -> dict[str, Any]:
        return {
            "configured": self.enabled,
            "ready": self.semantic_enabled,
            "collection": getattr(self.store, "collection_name", None),
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "source_checksum": self.source_checksum,
            "min_score": self.min_score,
            "role": "markdown_policy_retrieval",
            "detail": (
                "Markdown sənədləri semantic axtarış üçün hazırdır."
                if self.semantic_enabled
                else self.unavailable_reason or "Document Search bu versiyada bağlıdır."
            ),
        }

    @staticmethod
    def _chunk(hit: DocumentSearchHit) -> DocumentSearchChunk:
        payload = hit.payload
        required = ("document_id", "title", "heading", "text")
        if any(not isinstance(payload.get(field), str) for field in required):
            raise DocumentSearchBackendError(
                "document_search_unavailable",
                "Sənəd axtarışı müvəqqəti əlçatan deyil.",
            )
        return DocumentSearchChunk(
            chunk_id=hit.chunk_id,
            document_id=payload["document_id"],
            title=payload["title"],
            heading=payload["heading"],
            text=payload["text"],
            score=round(hit.score, 6),
        )

    def _trace(
        self,
        arguments: DocumentSearchArguments,
        candidates: Sequence[DocumentSearchHit],
        selected: Sequence[DocumentSearchHit],
    ) -> dict[str, Any]:
        selected_ids = {hit.chunk_id for hit in selected}
        return {
            "mode": "document_qdrant_v1",
            "query": arguments.query,
            "qdrant_checked": True,
            "semantic_state": "active",
            "min_score": self.min_score,
            "source_checksum": self.source_checksum,
            "candidate_count": len(candidates),
            "selected_chunk_ids": [hit.chunk_id for hit in selected],
            "candidates": [
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.payload.get("document_id"),
                    "filename": hit.payload.get("filename"),
                    "title": hit.payload.get("title"),
                    "heading": hit.payload.get("heading"),
                    "score": round(hit.score, 6),
                    "selected": hit.chunk_id in selected_ids,
                    "text_preview": str(hit.payload.get("text") or "")[:500],
                }
                for hit in candidates
            ],
        }

    def _unavailable(
        self,
        arguments: DocumentSearchArguments,
        reason: str,
    ) -> DocumentSearchBackendError:
        return DocumentSearchBackendError(
            "document_search_unavailable",
            "Sənəd axtarışı müvəqqəti əlçatan deyil.",
            debug_trace={
                "mode": "document_qdrant_v1",
                "query": arguments.query,
                "qdrant_checked": False,
                "semantic_state": "not_configured" if not self.enabled else "failed",
                "min_score": self.min_score,
                "source_checksum": self.source_checksum,
                "candidate_count": 0,
                "selected_chunk_ids": [],
                "candidates": [],
                "error_type": reason,
            },
        )

