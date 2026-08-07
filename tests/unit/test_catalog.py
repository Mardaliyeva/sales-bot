from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.catalog import CatalogLoadError, ProductCatalog
from app.tools.schemas import AttributeFilter, ProductSearchArguments


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


def test_catalog_hydrates_qdrant_ids_in_the_same_order(catalog: ProductCatalog) -> None:
    ids = [catalog.products[5]["product_id"], catalog.products[0]["product_id"]]

    hydrated = catalog.hydrate(ids)

    assert [product["product_id"] for product in hydrated] == ids


def test_catalog_rejects_unknown_qdrant_id(catalog: ProductCatalog) -> None:
    with pytest.raises(CatalogLoadError):
        catalog.hydrate(["prd_missing_999"])


def test_applied_filters_are_json_serializable(catalog: ProductCatalog) -> None:
    args = ProductSearchArguments(
        query="Intel laptop",
        category_id="laptops",
        attribute_filters=[AttributeFilter(field="cpu_brand", operator="eq", value="Intel")],
    )

    assert catalog.applied_filters(args) == {
        "attribute_filters": [{"field": "cpu_brand", "operator": "eq", "value": "Intel"}],
        "category_id": "laptops",
    }


def test_catalog_has_no_runtime_search_or_ranking_methods(catalog: ProductCatalog) -> None:
    assert not hasattr(catalog, "search")
    assert not hasattr(catalog, "rank_candidates")
