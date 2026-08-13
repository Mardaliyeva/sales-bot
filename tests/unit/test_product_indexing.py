from __future__ import annotations

import warnings
from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient, models

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.indexing.products import index_catalog
from app.tools.catalog import ProductCatalog
from app.tools.schemas import AttributeFilter, ProductSearchArguments
from app.vectorstores.qdrant import (
    QdrantProductStore,
    VectorStoreError,
    build_embedding_text,
    build_product_payload,
    product_point_id,
)


class FakeEmbeddings:
    deployment = "text-embedding-3-large"
    dimensions = 2

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]:
        del text_version, refresh
        return [[1.0, float(index % 10) / 10] for index, _ in enumerate(texts)]


def _local_store(*, vector_size: int = 2) -> QdrantProductStore:
    return QdrantProductStore(
        QdrantClient(":memory:"),
        collection_name="products_test",
        vector_size=vector_size,
    )


def test_product_point_id_is_deterministic() -> None:
    first = product_point_id("prd_laptops_001")
    second = product_point_id("prd_laptops_001")

    assert first == second
    assert first.version == 5
    assert first != product_point_id("prd_laptops_002")


def test_product_payload_contains_filter_and_version_fields(catalog_path: object) -> None:
    catalog = ProductCatalog(catalog_path)  # type: ignore[arg-type]
    catalog.load()
    product = catalog.products[0]
    payload = build_product_payload(
        product,
        catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
        embedding_text_version=DEFAULT_TEXT_VERSION,
        embedding_deployment="text-embedding-3-large",
        embedding_dimensions=3072,
    )

    assert payload["product_id"] == product["product_id"]
    assert payload["category_id"] == product["category"]["id"]
    assert payload["brand_normalized"]
    assert payload["description"] == product["description"]
    assert set(product["attributes"]) <= set(payload)
    assert payload["attribute_fields"] == sorted(product["attributes"])
    assert payload["dataset_version"] == catalog.manifest["dataset_version"]
    assert payload["embedding_dimensions"] == 3072


def test_embedding_text_contains_only_name_and_description(catalog_path: object) -> None:
    catalog = ProductCatalog(catalog_path)  # type: ignore[arg-type]
    catalog.load()
    product = catalog.products[0]

    text = build_embedding_text(product)

    assert text == f"{product['name']}\n{product['description']}"
    assert product["short_description"] not in text
    assert product["embedding_text"] not in text


def test_indexing_is_idempotent_and_status_is_ready(catalog_path: object) -> None:
    catalog = ProductCatalog(catalog_path)  # type: ignore[arg-type]
    store = _local_store()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            first = index_catalog(catalog, FakeEmbeddings(), store)
            second = index_catalog(catalog, FakeEmbeddings(), store)
        mismatched = store.status(
            [product["product_id"] for product in catalog.products],
            expected_dataset_version="1.0.0",
            expected_catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
            expected_embedding_text_version=DEFAULT_TEXT_VERSION,
            expected_embedding_deployment="different-deployment",
            expected_embedding_dimensions=2,
        )
    finally:
        store.close()

    assert first.product_count == 300
    assert first.status.ready is True
    assert second.status.indexed_count == 300
    assert second.status.missing_product_ids == ()
    assert second.status.extra_product_ids == ()
    assert second.status.payload_fields_match is True
    assert mismatched.metadata_matches is False
    assert mismatched.ready is False


def test_qdrant_filter_supports_all_product_search_fields() -> None:
    args = ProductSearchArguments(
        query="məhsul",
        category_id="laptops",
        brand="Lenovo",
        model_family="IdeaPad",
        color_code="silver",
        min_price=500,
        max_price=1500,
        in_stock=True,
        storage_gb=512,
        ram_gb=16,
        btu=12000,
        screen_size_in=55,
        connectivity="Bluetooth",
        active_noise_cancellation=True,
        attribute_filters=[
            AttributeFilter(field="cpu_brand", operator="eq", value="Intel"),
            AttributeFilter(field="battery_mah", operator="gte", value=5000),
            AttributeFilter(field="modes", operator="contains_any", value=["soyutma"]),
            AttributeFilter(
                field="indoor_unit_dimensions_mm",
                operator="eq",
                value="900 x 300 x 220 mm",
            ),
        ],
    )

    query_filter = QdrantProductStore.build_filter(args)

    assert query_filter is not None
    assert query_filter.must is not None
    keys = {condition.key for condition in query_filter.must if isinstance(condition, models.FieldCondition)}
    assert keys == {
        "category_id",
        "brand_normalized",
        "model_family_normalized",
        "color_code",
        "in_stock",
        "connectivity_normalized",
        "active_noise_cancellation",
        "storage_gb",
        "ram_gb",
        "btu",
        "screen_size_in",
        "sale_price",
        "cpu_brand_normalized",
        "battery_mah",
        "modes_normalized",
        "indoor_unit_dimensions_mm_normalized",
    }


def test_qdrant_exact_identifier_filter_uses_normalized_payload_fields() -> None:
    args = ProductSearchArguments(
        query="SYN-PH-APL-001",
        sku="syn-ph-apl-001",
        category_id="smartphones",
    )

    query_filter = QdrantProductStore.build_filter(args, include_identifiers=True)

    assert query_filter is not None
    assert query_filter.must is not None
    keys = {condition.key for condition in query_filter.must if isinstance(condition, models.FieldCondition)}
    assert "sku_normalized" in keys
    assert "category_id" in keys


def test_existing_incompatible_collection_is_not_recreated() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="products_test",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    store = QdrantProductStore(client, collection_name="products_test", vector_size=2)
    try:
        with pytest.raises(VectorStoreError):
            store.ensure_collection()
        info = client.get_collection("products_test")
    finally:
        store.close()

    assert isinstance(info.config.params.vectors, models.VectorParams)
    assert info.config.params.vectors.size == 3


def test_filtered_search_only_returns_matching_products(catalog_path: object) -> None:
    catalog = ProductCatalog(catalog_path)  # type: ignore[arg-type]
    store = _local_store()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            index_catalog(catalog, FakeEmbeddings(), store)
        args = ProductSearchArguments(query="telefon", category_id="smartphones", brand="Apple")
        hits = store.search([1.0, 0.0], args)
        candidates = store.search_candidates([1.0, 0.0], args, candidate_limit=20)
        count = store.count_candidates(args)
    finally:
        store.close()

    assert count == 10
    assert len(hits) == 5
    assert len(candidates) == 10
    assert all(hit.payload["category_id"] == "smartphones" for hit in hits)
    assert all(hit.payload["brand"] == "Apple" for hit in hits)


def test_ordered_candidates_use_numeric_payload_order_without_catalog_scan(
    catalog_path: object,
) -> None:
    catalog = ProductCatalog(catalog_path)  # type: ignore[arg-type]
    catalog.load()
    store = _local_store()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            index_catalog(catalog, FakeEmbeddings(), store)
        args = ProductSearchArguments(query="smartfon", category_id="smartphones")

        hits = store.ordered_candidates(
            args,
            field="display_size_in",
            direction="maximize",
            candidate_limit=20,
        )
    finally:
        store.close()

    values = [float(hit.payload["display_size_in"]) for hit in hits]
    assert len(hits) == 20
    assert values == sorted(values, reverse=True)
