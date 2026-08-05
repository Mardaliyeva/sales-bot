from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments, ProductSearchResult
from app.vectorstores.qdrant import QdrantProductStore


class QueryEmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]: ...


class SemanticProductSearch:
    """Offline semantic backend; intentionally not wired into product_search yet."""

    def __init__(
        self,
        catalog: ProductCatalog,
        embeddings: QueryEmbeddingBackend,
        store: QdrantProductStore,
    ) -> None:
        if not catalog.ready:
            raise ValueError("Semantic axtarış üçün kataloq əvvəlcədən yüklənməlidir")
        self.catalog = catalog
        self.embeddings = embeddings
        self.store = store
        self._products_by_id = {product["product_id"]: product for product in catalog.products}

    def search(self, args: ProductSearchArguments) -> ProductSearchResult:
        if args.sort != "relevance":
            raise ValueError("Semantic baseline yalnız relevance sıralamasını dəstəkləyir")
        total = self.store.count_candidates(args)
        if total == 0:
            return ProductSearchResult(
                total=0,
                applied_filters=self._applied_filters(args),
                items=[],
            )

        vectors = self.embeddings.embed([args.query], text_version=DEFAULT_TEXT_VERSION)
        if len(vectors) != 1:
            raise ValueError("Query üçün dəqiq bir embedding yaranmalıdır")
        hits = self.store.search(vectors[0], args)
        products = []
        for hit in hits:
            product = self._products_by_id.get(hit.product_id)
            if product is None:
                raise ValueError(f"Qdrant kataloqda olmayan məhsul qaytardı: {hit.product_id}")
            products.append(ProductCatalog._to_result(product))
        return ProductSearchResult(
            total=total,
            applied_filters=self._applied_filters(args),
            items=products,
        )

    @staticmethod
    def _applied_filters(args: ProductSearchArguments) -> dict[str, object]:
        return args.model_dump(exclude={"query", "sort", "limit"}, exclude_none=True)
