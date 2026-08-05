from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments


@pytest.fixture(scope="module")
def catalog() -> ProductCatalog:
    path = Path(__file__).resolve().parents[2] / "data" / "catalog" / "products.jsonl"
    instance = ProductCatalog(path)
    instance.load()
    return instance


def test_catalog_loads_canonical_dataset(catalog: ProductCatalog) -> None:
    assert catalog.ready is True
    assert len(catalog.products) == 300
    assert catalog.manifest["validation"]["status"] == "passed"


def test_reference_iphone_query(catalog: ProductCatalog) -> None:
    result = catalog.search(
        ProductSearchArguments(
            query="qara rəngdə 128 GB 2000-3000 AZN arası iPhone",
            category_id="smartphones",
            brand="Apple",
            model_family="iPhone",
            color_code="black",
            storage_gb=128,
            min_price=2000,
            max_price=3000,
            in_stock=True,
        )
    )
    assert result.total == 3
    assert [item.product_id for item in result.items] == [
        "prd_smartphones_001",
        "prd_smartphones_002",
        "prd_smartphones_003",
    ]


def test_all_metadata_golden_queries_respect_strict_filters(catalog: ProductCatalog) -> None:
    golden_path = catalog.catalog_dir / "golden_queries.json"
    queries = json.loads(golden_path.read_text(encoding="utf-8-sig"))
    for query in queries:
        if query["type"] != "metadata":
            continue
        result = catalog.search(ProductSearchArguments(query=query["query"], limit=5, **query["filters"]))
        expected = set(query["expected_product_ids"])
        returned = {item.product_id for item in result.items}
        assert result.total == len(expected), query["query_id"]
        assert returned <= expected, query["query_id"]
        assert len(returned) == min(5, len(expected)), query["query_id"]


def test_semantic_queries_have_a_lexical_baseline(catalog: ProductCatalog) -> None:
    queries = json.loads((catalog.catalog_dir / "golden_queries.json").read_text(encoding="utf-8-sig"))
    for query in queries:
        if query["type"] != "semantic":
            continue
        result = catalog.search(ProductSearchArguments(query=query["query"], limit=5, **query["filters"]))
        returned = {item.product_id for item in result.items}
        assert returned & set(query["expected_product_ids"]), query["query_id"]


def test_unknown_brand_returns_empty_result(catalog: ProductCatalog) -> None:
    result = catalog.search(
        ProductSearchArguments(query="qırmızı Nokia", category_id="smartphones", brand="Nokia")
    )
    assert result.total == 0
    assert result.items == []
