from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.tools.catalog import normalize_text
from app.tools.schemas import (
    BOOLEAN_ATTRIBUTE_FIELDS,
    DIMENSION_ATTRIBUTE_FIELDS,
    LIST_ATTRIBUTE_FIELDS,
    NUMERIC_ATTRIBUTE_FIELDS,
    TEXT_ATTRIBUTE_FIELDS,
    AttributeFilter,
    ProductSearchArguments,
)

DEFAULT_COLLECTION_NAME = "sales_bot_products_semantic_v2"
DEFAULT_VECTOR_SIZE = 3072
DEFAULT_UPSERT_BATCH_SIZE = 32
DEFAULT_QUERY_CANDIDATES = 20
DEFAULT_SORT_CANDIDATES = 50

INTEGER_ATTRIBUTE_FIELDS = frozenset(
    {
        "battery_hours",
        "battery_mah",
        "btu",
        "coverage_max_m2",
        "coverage_min_m2",
        "hdmi_count",
        "main_camera_mp",
        "noise_level_db",
        "ram_gb",
        "refresh_rate_hz",
        "screen_size_in",
        "storage_gb",
    }
)
FLOAT_ATTRIBUTE_FIELDS = NUMERIC_ATTRIBUTE_FIELDS - INTEGER_ATTRIBUTE_FIELDS

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "product_id": models.PayloadSchemaType.KEYWORD,
    "product_id_normalized": models.PayloadSchemaType.KEYWORD,
    "sku": models.PayloadSchemaType.KEYWORD,
    "sku_normalized": models.PayloadSchemaType.KEYWORD,
    "model_normalized": models.PayloadSchemaType.KEYWORD,
    "category_id": models.PayloadSchemaType.KEYWORD,
    "brand_normalized": models.PayloadSchemaType.KEYWORD,
    "model_family_normalized": models.PayloadSchemaType.KEYWORD,
    "color_code": models.PayloadSchemaType.KEYWORD,
    "currency": models.PayloadSchemaType.KEYWORD,
    "dataset_version": models.PayloadSchemaType.KEYWORD,
    "catalog_checksum": models.PayloadSchemaType.KEYWORD,
    "embedding_text_version": models.PayloadSchemaType.KEYWORD,
    "embedding_deployment": models.PayloadSchemaType.KEYWORD,
    "attribute_fields": models.PayloadSchemaType.KEYWORD,
    "list_price": models.PayloadSchemaType.FLOAT,
    "sale_price": models.PayloadSchemaType.FLOAT,
    "rating": models.PayloadSchemaType.FLOAT,
    "discount_percent": models.PayloadSchemaType.INTEGER,
    "stock_quantity": models.PayloadSchemaType.INTEGER,
    "warranty_months": models.PayloadSchemaType.INTEGER,
    "review_count": models.PayloadSchemaType.INTEGER,
    "embedding_dimensions": models.PayloadSchemaType.INTEGER,
    "in_stock": models.PayloadSchemaType.BOOL,
}
PAYLOAD_INDEXES.update(
    {field: models.PayloadSchemaType.INTEGER for field in INTEGER_ATTRIBUTE_FIELDS}
)
PAYLOAD_INDEXES.update(
    {field: models.PayloadSchemaType.FLOAT for field in FLOAT_ATTRIBUTE_FIELDS}
)
PAYLOAD_INDEXES.update(
    {field: models.PayloadSchemaType.BOOL for field in BOOLEAN_ATTRIBUTE_FIELDS}
)
PAYLOAD_INDEXES.update(
    {f"{field}_normalized": models.PayloadSchemaType.KEYWORD for field in TEXT_ATTRIBUTE_FIELDS}
)
PAYLOAD_INDEXES.update(
    {f"{field}_normalized": models.PayloadSchemaType.KEYWORD for field in LIST_ATTRIBUTE_FIELDS}
)
PAYLOAD_INDEXES.update(
    {f"{field}_normalized": models.PayloadSchemaType.KEYWORD for field in DIMENSION_ATTRIBUTE_FIELDS}
)

REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "product_id",
        "sku",
        "model",
        "name",
        "description",
        "category_id",
        "brand",
        "model_family",
        "color_code",
        "sale_price",
        "currency",
        "in_stock",
        "rating",
        "dataset_version",
        "catalog_checksum",
        "embedding_text_version",
        "embedding_deployment",
        "embedding_dimensions",
        "attribute_fields",
    }
)


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorSearchHit:
    product_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class CollectionStatus:
    exists: bool
    collection_name: str
    expected_count: int
    indexed_count: int
    vector_size: int | None
    distance: str | None
    missing_product_ids: tuple[str, ...]
    extra_product_ids: tuple[str, ...]
    dataset_versions: tuple[str, ...]
    catalog_checksums: tuple[str, ...]
    embedding_text_versions: tuple[str, ...]
    embedding_deployments: tuple[str, ...]
    embedding_dimensions: tuple[int, ...]
    metadata_matches: bool
    payload_fields_match: bool = True

    @property
    def ready(self) -> bool:
        return (
            self.exists
            and self.indexed_count == self.expected_count
            and self.vector_size is not None
            and self.distance == models.Distance.COSINE.value
            and not self.missing_product_ids
            and not self.extra_product_ids
            and len(self.dataset_versions) == 1
            and len(self.embedding_text_versions) == 1
            and len(self.embedding_deployments) == 1
            and self.embedding_dimensions == (self.vector_size,)
            and self.metadata_matches
            and self.payload_fields_match
        )


def product_point_id(product_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"sales-bot/product/{product_id}")


def build_embedding_text(product: dict[str, Any]) -> str:
    name = str(product.get("name", "")).strip()
    description = str(product.get("description", "")).strip()
    if not name or not description:
        raise VectorStoreError("Embedding üçün məhsul adı və description boş ola bilməz")
    return f"{name}\n{description}"


def normalize_dimensions(value: object) -> str:
    if isinstance(value, dict):
        parts = [value.get("width"), value.get("height"), value.get("depth")]
        if all(isinstance(part, int) and part > 0 for part in parts):
            return "x".join(str(part) for part in parts)
    if isinstance(value, str):
        parts = re.findall(r"\d+", value)
        if len(parts) == 3:
            return "x".join(str(int(part)) for part in parts)
    raise ValueError("Ölçü 'en x hündürlük x dərinlik' formatında olmalıdır")


def build_product_payload(
    product: dict[str, Any],
    *,
    catalog_checksum: str,
    embedding_text_version: str,
    embedding_deployment: str,
    embedding_dimensions: int,
) -> dict[str, Any]:
    attributes = dict(product["attributes"])
    payload: dict[str, Any] = {
        "product_id": product["product_id"],
        "product_id_normalized": normalize_text(product["product_id"]),
        "sku": product["sku"],
        "sku_normalized": normalize_text(product["sku"]),
        "name": product["name"],
        "description": product["description"],
        "category_id": product["category"]["id"],
        "category_name": product["category"]["name"],
        "brand": product["brand"],
        "brand_normalized": normalize_text(product["brand"]),
        "model": product["model"],
        "model_normalized": normalize_text(product["model"]),
        "model_family": product["model_family"],
        "model_family_normalized": normalize_text(product["model_family"]),
        "color_code": product["color"]["code"],
        "color_name": product["color"]["name"],
        "list_price": float(product["price"]["list"]),
        "sale_price": float(product["price"]["sale"]),
        "discount_percent": int(product["price"]["discount_percent"]),
        "currency": product["price"]["currency"],
        "in_stock": product["stock"]["status"] == "in_stock",
        "stock_quantity": int(product["stock"]["quantity"]),
        "warranty_months": int(product["warranty_months"]),
        "rating": float(product["rating"]),
        "review_count": int(product["review_count"]),
        "attribute_fields": sorted(attributes),
        "dataset_version": product["dataset_version"],
        "catalog_checksum": catalog_checksum,
        "embedding_text_version": embedding_text_version,
        "embedding_deployment": embedding_deployment,
        "embedding_dimensions": embedding_dimensions,
    }
    for field, value in attributes.items():
        payload[field] = value
        if field in TEXT_ATTRIBUTE_FIELDS and isinstance(value, str):
            payload[f"{field}_normalized"] = normalize_text(value)
        elif field in LIST_ATTRIBUTE_FIELDS and isinstance(value, list):
            payload[f"{field}_normalized"] = [normalize_text(str(item)) for item in value]
        elif field in DIMENSION_ATTRIBUTE_FIELDS:
            payload[f"{field}_normalized"] = normalize_dimensions(value)
    return payload


class QdrantProductStore:
    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        timeout_seconds: float = 30.0,
    ) -> QdrantProductStore:
        if settings.qdrant_url is None or settings.qdrant_url == "YOUR_QDRANT_CLOUD_URL":
            raise VectorStoreError("QDRANT_URL konfiqurasiya edilməyib")
        if settings.qdrant_api_key is None:
            raise VectorStoreError("QDRANT_API_KEY konfiqurasiya edilməyib")
        api_key = settings.qdrant_api_key.get_secret_value()
        if not api_key or api_key in {"YOUR_QDRANT_API_KEY", "CHANGE_ME"}:
            raise VectorStoreError("QDRANT_API_KEY konfiqurasiya edilməyib")
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
            timeout=timeout_seconds,
            prefer_grpc=False,
        )
        return cls(client, collection_name=settings.qdrant_collection_name)

    def close(self) -> None:
        self.client.close()

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        self._validate_collection()
        self._ensure_payload_indexes()

    def upsert_products(
        self,
        products: Sequence[dict[str, Any]],
        vectors: Sequence[list[float]],
        *,
        catalog_checksum: str,
        embedding_text_version: str,
        embedding_deployment: str,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> None:
        if len(products) != len(vectors):
            raise VectorStoreError("Məhsul və embedding sayı uyğun deyil")
        if any(len(vector) != self.vector_size for vector in vectors):
            raise VectorStoreError("Qdrant-a yazılan embedding ölçüsü uyğun deyil")
        points = [
            models.PointStruct(
                id=product_point_id(product["product_id"]),
                vector=vector,
                payload=build_product_payload(
                    product,
                    catalog_checksum=catalog_checksum,
                    embedding_text_version=embedding_text_version,
                    embedding_deployment=embedding_deployment,
                    embedding_dimensions=self.vector_size,
                ),
            )
            for product, vector in zip(products, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

    def count_candidates(self, args: ProductSearchArguments) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self.build_filter(args, include_identifiers=False),
            exact=True,
        )
        return result.count

    def exact_candidates(
        self,
        args: ProductSearchArguments,
        *,
        include_structured_filters: bool,
    ) -> list[VectorSearchHit]:
        if not args.has_exact_identifier:
            return []
        query_filter = self.build_filter(
            args,
            include_identifiers=True,
            include_structured_filters=include_structured_filters,
        )
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=50,
            with_payload=True,
            with_vectors=False,
        )
        hits = [self._record_to_hit(record, score=1.0) for record in records]
        hits.sort(key=lambda hit: hit.product_id)
        return hits

    def search(
        self,
        vector: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = DEFAULT_QUERY_CANDIDATES,
    ) -> list[VectorSearchHit]:
        return self.search_candidates(vector, args, candidate_limit=candidate_limit)[: args.limit]

    def search_candidates(
        self,
        vector: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = DEFAULT_QUERY_CANDIDATES,
    ) -> list[VectorSearchHit]:
        if len(vector) != self.vector_size:
            raise VectorStoreError("Query embedding ölçüsü collection ilə uyğun deyil")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=self.build_filter(args, include_identifiers=False),
            limit=max(candidate_limit, args.limit),
            with_payload=True,
            with_vectors=False,
        )
        hits = [self._record_to_hit(point, score=float(point.score)) for point in response.points]
        hits.sort(key=lambda hit: (-hit.score, hit.product_id))
        return hits[:candidate_limit]

    def status(
        self,
        expected_product_ids: Sequence[str],
        *,
        expected_dataset_version: str | None = None,
        expected_catalog_checksum: str | None = None,
        expected_embedding_text_version: str | None = None,
        expected_embedding_deployment: str | None = None,
        expected_embedding_dimensions: int | None = None,
    ) -> CollectionStatus:
        expected = set(expected_product_ids)
        if not self.client.collection_exists(self.collection_name):
            return CollectionStatus(
                exists=False,
                collection_name=self.collection_name,
                expected_count=len(expected),
                indexed_count=0,
                vector_size=None,
                distance=None,
                missing_product_ids=tuple(sorted(expected)),
                extra_product_ids=(),
                dataset_versions=(),
                catalog_checksums=(),
                embedding_text_versions=(),
                embedding_deployments=(),
                embedding_dimensions=(),
                metadata_matches=False,
                payload_fields_match=False,
            )
        vector_size, distance = self._collection_vector_config()
        payloads = self._all_payloads()
        actual_ids = {
            payload["product_id"]
            for payload in payloads
            if isinstance(payload.get("product_id"), str)
        }
        dataset_versions = self._unique_strings(payloads, "dataset_version")
        catalog_checksums = self._unique_strings(payloads, "catalog_checksum")
        text_versions = self._unique_strings(payloads, "embedding_text_version")
        deployments = self._unique_strings(payloads, "embedding_deployment")
        dimensions = tuple(
            sorted(
                {
                    value
                    for payload in payloads
                    if isinstance((value := payload.get("embedding_dimensions")), int)
                }
            )
        )
        expected_checks = (
            expected_dataset_version is None or dataset_versions == (expected_dataset_version,),
            expected_catalog_checksum is None or catalog_checksums == (expected_catalog_checksum,),
            expected_embedding_text_version is None
            or text_versions == (expected_embedding_text_version,),
            expected_embedding_deployment is None or deployments == (expected_embedding_deployment,),
            expected_embedding_dimensions is None or dimensions == (expected_embedding_dimensions,),
        )
        payload_fields_match = all(self._payload_fields_valid(payload) for payload in payloads)
        return CollectionStatus(
            exists=True,
            collection_name=self.collection_name,
            expected_count=len(expected),
            indexed_count=len(payloads),
            vector_size=vector_size,
            distance=distance,
            missing_product_ids=tuple(sorted(expected - actual_ids)),
            extra_product_ids=tuple(sorted(actual_ids - expected)),
            dataset_versions=dataset_versions,
            catalog_checksums=catalog_checksums,
            embedding_text_versions=text_versions,
            embedding_deployments=deployments,
            embedding_dimensions=dimensions,
            metadata_matches=all(expected_checks),
            payload_fields_match=payload_fields_match,
        )

    @staticmethod
    def build_filter(
        args: ProductSearchArguments,
        *,
        include_identifiers: bool = False,
        include_structured_filters: bool = True,
    ) -> models.Filter | None:
        conditions: list[models.Condition] = []
        if include_identifiers:
            identifier_fields = {
                "product_id_normalized": args.product_id,
                "sku_normalized": args.sku,
                "model_normalized": args.model,
            }
            for key, value in identifier_fields.items():
                if value is not None:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=normalize_text(value)),
                        )
                    )
        if include_structured_filters:
            conditions.extend(QdrantProductStore._structured_conditions(args))
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _structured_conditions(args: ProductSearchArguments) -> list[models.Condition]:
        conditions: list[models.Condition] = []
        exact_fields: dict[str, str | bool | None] = {
            "category_id": args.category_id,
            "brand_normalized": normalize_text(args.brand) if args.brand is not None else None,
            "model_family_normalized": (
                normalize_text(args.model_family) if args.model_family is not None else None
            ),
            "color_code": args.color_code,
            "in_stock": args.in_stock,
            "connectivity_normalized": (
                normalize_text(args.connectivity) if args.connectivity is not None else None
            ),
            "active_noise_cancellation": args.active_noise_cancellation,
        }
        for key, value in exact_fields.items():
            if value is not None:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

        integer_fields = {
            "storage_gb": args.storage_gb,
            "ram_gb": args.ram_gb,
            "btu": args.btu,
        }
        for key, value in integer_fields.items():
            if value is not None:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        if args.screen_size_in is not None:
            conditions.append(
                models.FieldCondition(
                    key="screen_size_in",
                    range=models.Range(gte=args.screen_size_in, lte=args.screen_size_in),
                )
            )
        if args.min_price is not None or args.max_price is not None:
            conditions.append(
                models.FieldCondition(
                    key="sale_price",
                    range=models.Range(gte=args.min_price, lte=args.max_price),
                )
            )
        conditions.extend(
            QdrantProductStore._attribute_condition(attribute_filter)
            for attribute_filter in args.attribute_filters
        )
        return conditions

    @staticmethod
    def _attribute_condition(attribute_filter: AttributeFilter) -> models.FieldCondition:
        field = attribute_filter.field
        operator = attribute_filter.operator
        value = attribute_filter.value
        if field in NUMERIC_ATTRIBUTE_FIELDS:
            numeric_value = float(value) if isinstance(value, (int, float)) else 0.0
            if operator == "eq":
                value_range = models.Range(gte=numeric_value, lte=numeric_value)
            elif operator == "gte":
                value_range = models.Range(gte=numeric_value)
            else:
                value_range = models.Range(lte=numeric_value)
            return models.FieldCondition(key=field, range=value_range)
        if field in BOOLEAN_ATTRIBUTE_FIELDS:
            return models.FieldCondition(key=field, match=models.MatchValue(value=bool(value)))
        if field in TEXT_ATTRIBUTE_FIELDS:
            key = f"{field}_normalized"
            if operator == "in" and isinstance(value, list):
                normalized = [normalize_text(str(item)) for item in value]
                return models.FieldCondition(key=key, match=models.MatchAny(any=normalized))
            return models.FieldCondition(
                key=key,
                match=models.MatchValue(value=normalize_text(str(value))),
            )
        if field in LIST_ATTRIBUTE_FIELDS:
            normalized = [normalize_text(str(item)) for item in value] if isinstance(value, list) else []
            return models.FieldCondition(
                key=f"{field}_normalized",
                match=models.MatchAny(any=normalized),
            )
        if field in DIMENSION_ATTRIBUTE_FIELDS:
            return models.FieldCondition(
                key=f"{field}_normalized",
                match=models.MatchValue(value=normalize_dimensions(value)),
            )
        raise ValueError(f"Dəstəklənməyən attribute filter: {field}")

    def _validate_collection(self) -> None:
        size, distance = self._collection_vector_config()
        if size != self.vector_size or distance != models.Distance.COSINE.value:
            raise VectorStoreError(
                "Qdrant collection vector konfiqurasiyası uyğun deyil; yeni collection adı istifadə edin"
            )

    def _collection_vector_config(self) -> tuple[int, str]:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError("Named-vector Qdrant collection bu mərhələdə dəstəklənmir")
        return vectors.size, vectors.distance.value

    def _ensure_payload_indexes(self) -> None:
        info = self.client.get_collection(self.collection_name)
        existing = set(info.payload_schema)
        for field_name, schema in PAYLOAD_INDEXES.items():
            if field_name not in existing:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    def _all_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        offset: int | str | UUID | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            payloads.extend(dict(record.payload or {}) for record in records)
            if offset is None:
                return payloads

    @staticmethod
    def _payload_fields_valid(payload: dict[str, Any]) -> bool:
        if not REQUIRED_PAYLOAD_FIELDS.issubset(payload):
            return False
        attribute_fields = payload.get("attribute_fields")
        return isinstance(attribute_fields, list) and all(
            isinstance(field, str) and field in payload for field in attribute_fields
        )

    @staticmethod
    def _unique_strings(payloads: Sequence[dict[str, Any]], key: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    value
                    for payload in payloads
                    if isinstance((value := payload.get(key)), str)
                }
            )
        )

    @staticmethod
    def _record_to_hit(record: Any, *, score: float) -> VectorSearchHit:
        payload = dict(record.payload or {})
        product_id = payload.get("product_id")
        if not isinstance(product_id, str):
            raise VectorStoreError("Qdrant nəticəsində product_id yoxdur")
        return VectorSearchHit(product_id=product_id, score=score, payload=payload)
