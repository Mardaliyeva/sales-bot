from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.retrieval.hybrid import HybridProductSearch
from app.tools.catalog import CatalogCandidate, ProductCatalog
from app.tools.product_search import ProductSearchTool
from app.tools.schemas import ProductSearchArguments
from app.vectorstores.qdrant import VectorSearchHit


class FakeEmbeddings:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]:
        del text_version, refresh
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, product_ids: Sequence[str] = ()) -> None:
        self.product_ids = list(product_ids)
        self.calls = 0

    def search_candidates(
        self,
        vector: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = 20,
    ) -> list[VectorSearchHit]:
        del vector, args
        self.calls += 1
        return [
            VectorSearchHit(product_id=product_id, score=1.0 - rank / 100, payload={})
            for rank, product_id in enumerate(self.product_ids[:candidate_limit])
        ]


@pytest.fixture
def catalog(catalog_path: Path) -> ProductCatalog:
    instance = ProductCatalog(catalog_path)
    instance.load()
    return instance


def _different_category(catalog: ProductCatalog, category_id: str) -> str:
    return next(
        product["category"]["id"]
        for product in catalog.products
        if product["category"]["id"] != category_id
    )


@pytest.mark.parametrize("identifier_field", ["product_id", "sku", "model"])
def test_exact_identifier_is_case_insensitive_and_ranked_first(
    catalog: ProductCatalog,
    identifier_field: str,
) -> None:
    product = catalog.products[0]
    store = FakeStore([catalog.products[1]["product_id"]])
    search = HybridProductSearch(catalog, FakeEmbeddings(), store)

    result = search.search(
        ProductSearchArguments(query=f"MƏNƏ {product[identifier_field].upper()} LAZIMDIR")
    )

    assert result.items[0].product_id == product["product_id"]


def test_model_family_is_not_treated_as_exact_identifier(catalog: ProductCatalog) -> None:
    product = catalog.products[0]
    search = HybridProductSearch(catalog)

    matched = search._exact_product_ids(product["model_family"])

    assert matched == []


def test_fusion_deduplicates_products_and_supports_semantic_only_hits(
    catalog: ProductCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_product = catalog.products[0]
    semantic_product = catalog.products[-1]
    monkeypatch.setattr(
        catalog,
        "rank_candidates",
        lambda args, limit: [CatalogCandidate(product=lexical_product, score=1.0)],
    )
    store = FakeStore([lexical_product["product_id"], semantic_product["product_id"]])
    search = HybridProductSearch(catalog, FakeEmbeddings(), store)

    result = search.search(ProductSearchArguments(query="tamamilə naməlum ifadə zzzx"))
    returned = [item.product_id for item in result.items]

    assert returned.count(lexical_product["product_id"]) == 1
    assert semantic_product["product_id"] in returned
    assert result.total == 2


def test_rrf_uses_lexical_rank_as_deterministic_tie_break(
    catalog: ProductCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = catalog.products[:2]
    monkeypatch.setattr(
        catalog,
        "rank_candidates",
        lambda args, limit: [
            CatalogCandidate(product=first, score=1.0),
            CatalogCandidate(product=second, score=0.9),
        ],
    )
    store = FakeStore([second["product_id"], first["product_id"]])
    search = HybridProductSearch(catalog, FakeEmbeddings(), store)

    result = search.search(ProductSearchArguments(query="rrf tie query"))

    assert [item.product_id for item in result.items[:2]] == [
        first["product_id"],
        second["product_id"],
    ]


def test_qdrant_hit_must_exist_locally_and_pass_local_filters(catalog: ProductCatalog) -> None:
    laptop = next(product for product in catalog.products if product["category"]["id"] == "laptops")
    phone = next(product for product in catalog.products if product["category"]["id"] == "smartphones")
    store = FakeStore(["prd_missing_999", phone["product_id"], laptop["product_id"]])
    search = HybridProductSearch(catalog, FakeEmbeddings(), store)

    result = search.search(ProductSearchArguments(query="zzzx", category_id="laptops"))

    assert result.items
    assert all(item.category_id == "laptops" for item in result.items)
    assert phone["product_id"] not in {item.product_id for item in result.items}


def test_exact_identifier_with_conflicting_filter_returns_empty(catalog: ProductCatalog) -> None:
    product = catalog.products[0]
    conflicting_category = _different_category(catalog, product["category"]["id"])
    embeddings = FakeEmbeddings()
    store = FakeStore()
    search = HybridProductSearch(catalog, embeddings, store)

    result = search.search(
        ProductSearchArguments(
            query=product["sku"],
            category_id=conflicting_category,  # type: ignore[arg-type]
        )
    )

    assert result.total == 0
    assert result.items == []
    assert embeddings.calls == 0
    assert store.calls == 0


@pytest.mark.parametrize("sort", ["price_asc", "price_desc", "rating_desc"])
def test_non_relevance_sort_bypasses_semantic(
    catalog: ProductCatalog,
    sort: str,
) -> None:
    embeddings = FakeEmbeddings()
    store = FakeStore([catalog.products[0]["product_id"]])
    search = HybridProductSearch(catalog, embeddings, store)
    args = ProductSearchArguments(query="telefon", category_id="smartphones", sort=sort)  # type: ignore[arg-type]

    result = search.search(args)
    lexical_result = catalog.search(args)

    assert result == lexical_result
    assert embeddings.calls == 0
    assert store.calls == 0


def test_semantic_failure_falls_back_and_enters_cooldown(catalog: ProductCatalog) -> None:
    embeddings = FakeEmbeddings(RuntimeError("provider unavailable"))
    store = FakeStore()
    now = [100.0]
    search = HybridProductSearch(
        catalog,
        embeddings,
        store,
        semantic_cooldown_seconds=60,
        clock=lambda: now[0],
    )
    args = ProductSearchArguments(query="telefon", category_id="smartphones")

    first = search.search(args)
    second = search.search(args)
    now[0] = 161.0
    third = search.search(args)

    assert first == catalog.search(args)
    assert second == first
    assert third == first
    assert embeddings.calls == 2
    assert store.calls == 0


@pytest.mark.asyncio
async def test_product_search_tool_runs_sync_backend_in_worker_thread() -> None:
    class Backend:
        def search(self, arguments: ProductSearchArguments) -> Any:
            return {"query": arguments.query}

    result = await ProductSearchTool(Backend()).execute(ProductSearchArguments(query="laptop"))

    assert result == {"query": "laptop"}
