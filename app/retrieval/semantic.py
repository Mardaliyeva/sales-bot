from __future__ import annotations

from app.retrieval.qdrant import QdrantProductSearch


class SemanticProductSearch(QdrantProductSearch):
    """Offline evaluator alias for the same Qdrant-only runtime backend."""
