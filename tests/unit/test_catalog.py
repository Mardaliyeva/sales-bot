from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog_generation import check_catalog
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


def test_catalog_builds_stable_bounded_field_capability_registry(
    catalog: ProductCatalog,
) -> None:
    capabilities = catalog.field_capabilities()
    display = catalog.field_capability("display_size_in")

    assert capabilities == tuple(sorted(capabilities, key=lambda item: item.field))
    assert len(catalog.field_capability_checksum) == 64
    assert display is not None
    assert display.value_type == "number"
    assert display.sortable is True
    assert "smartphones" in display.categories
    assert display.coverage_count <= len(catalog.products)
    assert all(not hasattr(item, "product_ids") for item in capabilities)


def test_all_generated_records_have_consistent_encoded_attributes() -> None:
    assert check_catalog() == []


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


def test_catalog_canonicalizes_unique_model_prefix_and_text_attribute(
    catalog: ProductCatalog,
) -> None:
    args = ProductSearchArguments(
        query="Samsung QN900D Neo QLED 8K televizoru",
        category_id="televisions",
        model="QN900D",
        attribute_filters=[
            AttributeFilter(field="resolution", operator="eq", value="4K")
        ],
    )

    canonical = catalog.canonicalize_search_arguments(args)

    assert canonical.arguments.model == "QN900D Neo QLED 8K"
    assert canonical.arguments.attribute_filters[0].value == "4K UHD"
    assert {item["field"] for item in canonical.corrections} == {
        "model",
        "attributes.resolution",
    }


def test_query_facet_does_not_turn_discovery_into_exact_lookup(catalog: ProductCatalog) -> None:
    args = ProductSearchArguments(
        query="Apple iPhone 17 Pro modeli",
        category_id="smartphones",
        brand="Apple",
        model_family="iPhone",
    )

    canonical = catalog.canonicalize_search_arguments(args)

    assert canonical.arguments.model is None


def test_legacy_catalog_does_not_infer_model_from_natural_language(
    catalog: ProductCatalog,
) -> None:
    args = ProductSearchArguments(
        query="Samsung QN900D Neo QLED 8K 500 AZN-dən ucuz olmalıdır",
        category_id="televisions",
        max_price=500,
    )

    canonical = catalog.canonicalize_search_arguments(args)

    assert canonical.arguments.model is None


def test_legacy_lookup_drops_unmapped_preference_without_inventing_model(
    catalog: ProductCatalog,
) -> None:
    args = ProductSearchArguments(
        query="Samsung WindFree Elite 12K BTU və Wi-Fi məlumatını de",
        search_intent="lookup",
        requested_fields=["btu", "wifi"],
        category_id="air_conditioners",
        brand="Samsung",
        model_family="WindFree Elite",
    )

    canonical = catalog.canonicalize_search_arguments(args)

    assert canonical.arguments.model is None
    assert canonical.arguments.model_family is None
    assert any(
        item["action"] == "removed_unmapped_preference_filter"
        for item in canonical.corrections
    )


def test_legacy_catalog_does_not_parse_currency_language_into_constraint(
    catalog: ProductCatalog,
) -> None:
    args = ProductSearchArguments(
        query="Samsung QN900D Neo QLED 8K 500 AZN-dən ucuz olmalıdır",
        category_id="televisions",
    )

    canonical = catalog.canonicalize_search_arguments(args)

    assert canonical.arguments.max_price is None
    assert canonical.arguments.model is None
