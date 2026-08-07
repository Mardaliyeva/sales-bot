from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.retrieval.qdrant import QdrantProductSearch
from app.tools.catalog import ProductCatalog
from app.tools.product_search import ProductSearchBackendError, ProductSearchTool
from app.tools.registry import ToolRegistry
from app.tools.schemas import ProductSearchArguments
from app.vectorstores.qdrant import VectorSearchHit


class FakeEmbeddings:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]:
        del text_version, refresh
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    collection_name = "products_v2_test"

    def __init__(
        self,
        products: Sequence[dict[str, Any]],
        *,
        exact_unfiltered: Sequence[str] = (),
        exact_filtered: Sequence[str] = (),
        filtered_count: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.products = {product["product_id"]: product for product in products}
        self.semantic_ids = [product["product_id"] for product in products]
        self.exact_unfiltered = list(exact_unfiltered)
        self.exact_filtered = list(exact_filtered)
        self.filtered_count = filtered_count
        self.error = error
        self.candidate_limits: list[int] = []
        self.semantic_scores: dict[str, float] = {}

    def count_candidates(self, args: ProductSearchArguments) -> int:
        if self.error is not None:
            raise self.error
        if self.filtered_count is not None:
            return self.filtered_count
        return len(self._matching_ids(args))

    def exact_candidates(
        self,
        _: ProductSearchArguments,
        *,
        include_structured_filters: bool,
    ) -> list[VectorSearchHit]:
        ids = self.exact_filtered if include_structured_filters else self.exact_unfiltered
        return [self._hit(product_id, 1.0) for product_id in ids]

    def search_candidates(
        self,
        _: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = 20,
    ) -> list[VectorSearchHit]:
        self.candidate_limits.append(candidate_limit)
        matching = set(self._matching_ids(args))
        return [
            self._hit(product_id, self.semantic_scores.get(product_id, 1.0 - rank / 100))
            for rank, product_id in enumerate(self.semantic_ids)
            if product_id in matching
        ][:candidate_limit]

    def _matching_ids(self, args: ProductSearchArguments) -> list[str]:
        matched: list[str] = []
        for product_id in self.semantic_ids:
            product = self.products.get(product_id, {})
            attributes = product.get("attributes", {})
            if args.category_id is not None and product.get("category", {}).get("id") != args.category_id:
                continue
            if args.brand is not None and product.get("brand") != args.brand:
                continue
            if args.model_family is not None and product.get("model_family") != args.model_family:
                continue
            if args.color_code is not None and product.get("color", {}).get("code") != args.color_code:
                continue
            if args.in_stock is not None:
                in_stock = product.get("stock", {}).get("status") == "in_stock"
                if in_stock != args.in_stock:
                    continue
            sale_price = product.get("price", {}).get("sale", 0)
            if args.max_price is not None and sale_price > args.max_price:
                continue
            if args.min_price is not None and sale_price < args.min_price:
                continue
            simple_fields = {
                "storage_gb": args.storage_gb,
                "ram_gb": args.ram_gb,
                "btu": args.btu,
                "screen_size_in": args.screen_size_in,
                "connectivity": args.connectivity,
                "active_noise_cancellation": args.active_noise_cancellation,
            }
            if any(
                expected is not None and attributes.get(field) != expected
                for field, expected in simple_fields.items()
            ):
                continue
            matched.append(product_id)
        return matched[:]

    def _hit(self, product_id: str, score: float) -> VectorSearchHit:
        product = self.products.get(product_id, {})
        return VectorSearchHit(
            product_id=product_id,
            score=score,
            payload={
                "product_id": product_id,
                "name": product.get("name"),
                "sale_price": product.get("price", {}).get("sale", 0),
                "rating": product.get("rating", 0),
            },
        )


@pytest.fixture
def catalog(catalog_path: Path) -> ProductCatalog:
    instance = ProductCatalog(catalog_path)
    instance.load()
    return instance


def test_semantic_search_hydrates_qdrant_ids_without_lexical_fields(catalog: ProductCatalog) -> None:
    selected = catalog.products[:2]
    embeddings = FakeEmbeddings()
    search = QdrantProductSearch(catalog, embeddings, FakeStore(selected))

    execution = search.search_with_trace(ProductSearchArguments(query="universitet üçün laptop"))

    assert [item.product_id for item in execution.result.items] == [
        product["product_id"] for product in selected
    ]
    assert embeddings.calls == [["universitet üçün laptop"]]
    assert execution.debug_trace["mode"] == "qdrant_only_v2"
    assert "lexical_candidates" not in execution.debug_trace
    assert "merged_candidates" not in execution.debug_trace
    assert execution.debug_trace["hydrated_product_ids"] == [
        product["product_id"] for product in selected
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [("product_id", "prd_smartphones_001"), ("sku", "SYN-PH-APL-001"), ("model", "iPhone 17 Pro")],
)
def test_exact_qdrant_lookup_precedes_embedding(
    catalog: ProductCatalog,
    field: str,
    value: str,
) -> None:
    product = catalog.products[0]
    embeddings = FakeEmbeddings()
    store = FakeStore(
        [product],
        exact_unfiltered=[product["product_id"]],
        exact_filtered=[product["product_id"]],
    )
    search = QdrantProductSearch(catalog, embeddings, store)

    args = ProductSearchArguments(query=value, **{field: value})
    execution = search.search_with_trace(args)

    assert execution.result.items[0].product_id == product["product_id"]
    assert execution.result.match_status == "exact_match"
    assert execution.result.strict_total == 1
    assert embeddings.calls == []
    assert execution.debug_trace["semantic_state"] == "not_run_exact_match"


def test_exact_filter_conflict_returns_empty_without_embedding(catalog: ProductCatalog) -> None:
    product = catalog.products[0]
    embeddings = FakeEmbeddings()
    store = FakeStore(
        [product],
        exact_unfiltered=[product["product_id"]],
        exact_filtered=[],
    )
    search = QdrantProductSearch(catalog, embeddings, store)

    execution = search.search_with_trace(
        ProductSearchArguments(
            query=product["sku"],
            sku=product["sku"],
            category_id="laptops",
        )
    )

    assert execution.result.total == 0
    assert execution.result.match_status == "not_found"
    assert execution.result.items == []
    assert execution.debug_trace["exact_filter_conflict"] is True
    assert embeddings.calls == []


def test_no_filter_candidates_skips_embedding(catalog: ProductCatalog) -> None:
    embeddings = FakeEmbeddings()
    search = QdrantProductSearch(
        catalog,
        embeddings,
        FakeStore([], filtered_count=0),
    )

    result = search.search(ProductSearchArguments(query="olmayan məhsul", brand="Missing"))

    assert result.total == 0
    assert result.items == []
    assert embeddings.calls == []


def test_missing_exact_model_returns_at_most_three_family_alternatives(
    catalog: ProductCatalog,
) -> None:
    iphones = [
        product for product in catalog.products if product["model_family"] == "iPhone"
    ]
    embeddings = FakeEmbeddings()
    search = QdrantProductSearch(catalog, embeddings, FakeStore(iphones))

    execution = search.search_with_trace(
        ProductSearchArguments(
            query="Apple iPhone 19",
            model="iPhone 19",
            category_id="smartphones",
        )
    )

    assert execution.result.match_status == "alternatives"
    assert execution.result.requested_label == "iPhone 19"
    assert execution.result.strict_total == 0
    assert 1 <= len(execution.result.items) <= 3
    assert all(item.model_family == "iPhone" for item in execution.result.items)
    assert all(item.differences[0].startswith("Model fərqlidir") for item in execution.result.items)
    assert execution.debug_trace["alternative_stages"][0]["filters"]["model_family"] == "iPhone"
    assert embeddings.calls == [["Apple iPhone 19"]]


@pytest.mark.parametrize(
    ("category_id", "model_family", "query"),
    [
        ("smartphones", "iPhone", "iPhone Future 99"),
        ("tablets", "iPad", "iPad Future 99"),
        ("laptops", "IdeaPad", "IdeaPad Future 99"),
        ("air_conditioners", "Inverter Split", "Inverter Split Future 99"),
        ("televisions", "Samsung TV", "Samsung TV Future 99"),
        ("headphones", "Sony Audio", "Sony Audio Future 99"),
    ],
)
def test_catalog_facets_drive_alternatives_across_categories(
    catalog: ProductCatalog,
    category_id: str,
    model_family: str,
    query: str,
) -> None:
    products = [
        product for product in catalog.products if product["model_family"] == model_family
    ]
    search = QdrantProductSearch(catalog, FakeEmbeddings(), FakeStore(products))

    result = search.search(
        ProductSearchArguments(
            query=query,
            model=query,
            category_id=category_id,  # type: ignore[arg-type]
        )
    )

    assert result.match_status == "alternatives"
    assert 1 <= len(result.items) <= 3
    assert all(item.model_family == model_family for item in result.items)


def test_missing_color_is_relaxed_and_explained(catalog: ProductCatalog) -> None:
    iphones = [
        product
        for product in catalog.products
        if product["model_family"] == "iPhone" and product["color"]["code"] != "green"
    ][:4]
    search = QdrantProductSearch(catalog, FakeEmbeddings(), FakeStore(iphones))

    result = search.search(
        ProductSearchArguments(
            query="yaşıl iPhone 19",
            model="iPhone 19",
            category_id="smartphones",
            color_code="green",
        )
    )

    assert result.match_status == "alternatives"
    assert "color_code" in result.relaxed_fields
    assert all(any("Rəng fərqlidir" in value for value in item.differences) for item in result.items)


def test_required_brand_is_never_relaxed_for_alternatives(catalog: ProductCatalog) -> None:
    iphones = [
        product for product in catalog.products if product["model_family"] == "iPhone"
    ]
    embeddings = FakeEmbeddings()
    search = QdrantProductSearch(catalog, embeddings, FakeStore(iphones))

    result = search.search(
        ProductSearchArguments(
            query="yalnız Samsung Future Phone",
            model="Future Phone",
            category_id="smartphones",
            brand="Samsung",
            required_filter_fields=["brand"],
        )
    )

    assert result.match_status == "not_found"
    assert result.items == []
    assert embeddings.calls == []


def test_alternatives_keep_budget_and_stock_hard(catalog: ProductCatalog) -> None:
    smartphones = [
        product
        for product in catalog.products
        if product["category"]["id"] == "smartphones"
    ]
    search = QdrantProductSearch(catalog, FakeEmbeddings(), FakeStore(smartphones))

    result = search.search(
        ProductSearchArguments(
            query="iPhone 19 stokda 1000 AZN-dək",
            model="iPhone 19",
            category_id="smartphones",
            max_price=1000,
            in_stock=True,
        )
    )

    assert result.match_status in {"alternatives", "not_found"}
    assert all(item.sale_price <= 1000 for item in result.items)
    assert all(item.stock_status == "in_stock" for item in result.items)


def test_alternative_score_gate_rejects_unrelated_candidates(catalog: ProductCatalog) -> None:
    iphones = [
        product for product in catalog.products if product["model_family"] == "iPhone"
    ][:3]
    store = FakeStore(iphones)
    store.semantic_scores = {product["product_id"]: 0.49 for product in iphones}
    search = QdrantProductSearch(
        catalog,
        FakeEmbeddings(),
        store,
        alternative_min_score=0.5,
    )

    result = search.search(
        ProductSearchArguments(
            query="iPhone 99",
            model="iPhone 99",
            category_id="smartphones",
        )
    )

    assert result.match_status == "not_found"
    assert result.items == []


@pytest.mark.parametrize("sort", ["price_asc", "price_desc", "rating_desc"])
def test_sort_uses_top_fifty_semantic_candidates(catalog: ProductCatalog, sort: str) -> None:
    products = catalog.products[:60]
    store = FakeStore(products)
    search = QdrantProductSearch(catalog, FakeEmbeddings(), store)

    result = search.search(ProductSearchArguments(query="məhsul", sort=sort))  # type: ignore[arg-type]

    assert len(result.items) == 5
    assert store.candidate_limits == [50]


def test_embedding_failure_is_explicit_and_has_no_fallback(catalog: ProductCatalog) -> None:
    search = QdrantProductSearch(
        catalog,
        FakeEmbeddings(RuntimeError("provider unavailable")),
        FakeStore(catalog.products[:2]),
    )

    with pytest.raises(ProductSearchBackendError) as captured:
        search.search(ProductSearchArguments(query="telefon"))

    assert captured.value.code == "product_search_unavailable"
    assert captured.value.debug_trace is not None
    assert captured.value.debug_trace["semantic_state"] == "failed"


def test_unknown_qdrant_product_id_fails_instead_of_silently_hydrating(
    catalog: ProductCatalog,
) -> None:
    store = FakeStore([])
    store.filtered_count = 1
    store.semantic_ids = ["prd_missing_999"]
    search = QdrantProductSearch(catalog, FakeEmbeddings(), store)

    with pytest.raises(ProductSearchBackendError):
        search.search(ProductSearchArguments(query="məhsul"))


@pytest.mark.asyncio
async def test_product_search_tool_runs_sync_backend_in_worker_thread() -> None:
    class Backend:
        def search(self, arguments: ProductSearchArguments) -> Any:
            return {"query": arguments.query}

    result = await ProductSearchTool(Backend()).execute(ProductSearchArguments(query="laptop"))

    assert result == {"query": "laptop"}


@pytest.mark.asyncio
async def test_tool_registry_returns_explicit_unavailable_error_with_trace(
    catalog: ProductCatalog,
) -> None:
    search = QdrantProductSearch(catalog)
    registry = ToolRegistry(ProductSearchTool(search), timeout_seconds=1)

    execution = await registry.execute_with_trace("product_search", {"query": "telefon"})

    assert execution.result["code"] == "product_search_unavailable"
    assert execution.debug_trace is not None
    assert execution.debug_trace["semantic_state"] == "not_configured"
