from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.tools.catalog import ProductCatalog, normalize_text
from app.tools.schemas import ProductSearchArguments, ProductSearchResult
from app.vectorstores.qdrant import VectorSearchHit

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_COOLDOWN_SECONDS = 60.0


class QueryEmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]: ...


class SemanticCandidateStore(Protocol):
    def search_candidates(
        self,
        vector: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[VectorSearchHit]: ...


@dataclass
class _RankedCandidate:
    product: dict[str, Any]
    exact: bool = False
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    def rrf_score(self, rrf_k: int) -> float:
        score = 0.0
        if self.lexical_rank is not None:
            score += 1.0 / (rrf_k + self.lexical_rank)
        if self.semantic_rank is not None:
            score += 1.0 / (rrf_k + self.semantic_rank)
        return score


class HybridProductSearch:
    """Runtime product search: exact identifiers + lexical + optional semantic."""

    def __init__(
        self,
        catalog: ProductCatalog,
        embeddings: QueryEmbeddingBackend | None = None,
        store: SemanticCandidateStore | None = None,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_cooldown_seconds: float = DEFAULT_SEMANTIC_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not catalog.ready:
            raise ValueError("Hybrid axtarış üçün kataloq əvvəlcədən yüklənməlidir")
        if (embeddings is None) != (store is None):
            raise ValueError("Embedding və Qdrant birlikdə konfiqurasiya edilməlidir")
        if candidate_limit <= 0 or rrf_k <= 0 or semantic_cooldown_seconds < 0:
            raise ValueError("Hybrid axtarış parametrləri etibarsızdır")

        self.catalog = catalog
        self.embeddings = embeddings
        self.store = store
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        self.semantic_cooldown_seconds = semantic_cooldown_seconds
        self._clock = clock
        self._cooldown_lock = threading.Lock()
        self._semantic_disabled_until = 0.0
        self._products_by_id = {product["product_id"]: product for product in catalog.products}
        self._identifier_index = self._build_identifier_index(catalog.products)

    @property
    def semantic_enabled(self) -> bool:
        return self.embeddings is not None and self.store is not None

    def search(self, args: ProductSearchArguments) -> ProductSearchResult:
        if args.sort != "relevance":
            return self.catalog.search(args)

        exact_ids = self._exact_product_ids(args.query)
        exact_products = [self._products_by_id[product_id] for product_id in exact_ids]
        matching_exact = [
            product for product in exact_products if ProductCatalog._matches(product, args)
        ]
        if exact_products and len(matching_exact) != len(exact_products):
            logger.info("product_search.exact_filter_conflict")
            return self._result(args, [], total=0)

        candidates: dict[str, _RankedCandidate] = {}
        for product in matching_exact:
            candidates[product["product_id"]] = _RankedCandidate(product=product, exact=True)

        lexical = self.catalog.rank_candidates(args, limit=self.candidate_limit)
        for rank, candidate in enumerate(lexical, start=1):
            product_id = candidate.product["product_id"]
            ranked = candidates.setdefault(product_id, _RankedCandidate(product=candidate.product))
            ranked.lexical_rank = rank

        semantic_hits, semantic_state = self._semantic_candidates(args)
        for rank, hit in enumerate(semantic_hits, start=1):
            product = self._products_by_id.get(hit.product_id)
            if product is None or not ProductCatalog._matches(product, args):
                continue
            ranked = candidates.setdefault(hit.product_id, _RankedCandidate(product=product))
            ranked.semantic_rank = rank

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                not candidate.exact,
                -candidate.rrf_score(self.rrf_k),
                candidate.lexical_rank or self.candidate_limit + 1,
                candidate.semantic_rank or self.candidate_limit + 1,
                candidate.product["product_id"],
            ),
        )
        if ProductCatalog._has_structured_filter(args):
            total = self.catalog.count_filtered(args)
        else:
            total = len(ranked_candidates)

        logger.info(
            "product_search.hybrid_completed",
            extra={
                "exact_count": len(matching_exact),
                "lexical_count": len(lexical),
                "semantic_count": len(semantic_hits),
                "merged_count": len(ranked_candidates),
                "semantic_state": semantic_state,
            },
        )
        return self._result(args, ranked_candidates[: args.limit], total=total)

    def _semantic_candidates(
        self,
        args: ProductSearchArguments,
    ) -> tuple[list[VectorSearchHit], str]:
        if self.embeddings is None or self.store is None:
            return [], "not_configured"
        with self._cooldown_lock:
            if self._clock() < self._semantic_disabled_until:
                return [], "cooldown"
        try:
            vectors = self.embeddings.embed([args.query], text_version=DEFAULT_TEXT_VERSION)
            if len(vectors) != 1:
                raise ValueError("Query üçün dəqiq bir embedding yaranmalıdır")
            hits = self.store.search_candidates(
                vectors[0],
                args,
                candidate_limit=self.candidate_limit,
            )
        except Exception as exc:
            with self._cooldown_lock:
                self._semantic_disabled_until = self._clock() + self.semantic_cooldown_seconds
            logger.warning(
                "product_search.semantic_fallback",
                extra={"error_type": type(exc).__name__},
            )
            return [], "failed"
        return hits, "active"

    def _exact_product_ids(self, query: str) -> list[str]:
        normalized_query = normalize_text(query)
        matched: set[str] = set()
        for identifier, product_ids in self._identifier_index.items():
            pattern = rf"(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])"
            if re.search(pattern, normalized_query):
                matched.update(product_ids)
        return sorted(matched)

    @staticmethod
    def _build_identifier_index(
        products: Sequence[dict[str, Any]],
    ) -> dict[str, tuple[str, ...]]:
        model_counts = Counter(normalize_text(str(product["model"])) for product in products)
        identifiers: dict[str, set[str]] = {}
        for product in products:
            product_id = product["product_id"]
            values = [product_id, product["sku"]]
            model = normalize_text(str(product["model"]))
            if model and model_counts[model] == 1:
                values.append(product["model"])
            for value in values:
                normalized = normalize_text(str(value)).strip()
                if normalized:
                    identifiers.setdefault(normalized, set()).add(product_id)
        return {key: tuple(sorted(value)) for key, value in identifiers.items()}

    @staticmethod
    def _result(
        args: ProductSearchArguments,
        candidates: Sequence[_RankedCandidate] | Sequence[dict[str, Any]],
        *,
        total: int,
    ) -> ProductSearchResult:
        products = [
            candidate.product if isinstance(candidate, _RankedCandidate) else candidate
            for candidate in candidates
        ]
        return ProductSearchResult(
            total=total,
            applied_filters=ProductCatalog._applied_filters(args),
            items=[ProductCatalog._to_result(product) for product in products],
        )
