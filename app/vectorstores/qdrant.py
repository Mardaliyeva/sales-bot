from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.tools.catalog import normalize_text
from app.tools.schemas import ProductSearchArguments

DEFAULT_COLLECTION_NAME = "sales_bot_products_v1"
DEFAULT_VECTOR_SIZE = 3072
DEFAULT_UPSERT_BATCH_SIZE = 32
DEFAULT_QUERY_CANDIDATES = 20

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "product_id": models.PayloadSchemaType.KEYWORD,
    "sku": models.PayloadSchemaType.KEYWORD,
    "category_id": models.PayloadSchemaType.KEYWORD,
    "brand_normalized": models.PayloadSchemaType.KEYWORD,
    "model_family_normalized": models.PayloadSchemaType.KEYWORD,
    "color_code": models.PayloadSchemaType.KEYWORD,
    "currency": models.PayloadSchemaType.KEYWORD,
    "connectivity_normalized": models.PayloadSchemaType.KEYWORD,
    "dataset_version": models.PayloadSchemaType.KEYWORD,
    "catalog_checksum": models.PayloadSchemaType.KEYWORD,
    "embedding_text_version": models.PayloadSchemaType.KEYWORD,
    "embedding_deployment": models.PayloadSchemaType.KEYWORD,
    "sale_price": models.PayloadSchemaType.FLOAT,
    "screen_size_in": models.PayloadSchemaType.FLOAT,
    "storage_gb": models.PayloadSchemaType.INTEGER,
    "ram_gb": models.PayloadSchemaType.INTEGER,
    "btu": models.PayloadSchemaType.INTEGER,
    "embedding_dimensions": models.PayloadSchemaType.INTEGER,
    "in_stock": models.PayloadSchemaType.BOOL,
    "active_noise_cancellation": models.PayloadSchemaType.BOOL,
}


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
        )


def product_point_id(product_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"sales-bot/product/{product_id}")


def build_product_payload(
    product: dict[str, Any],
    *,
    catalog_checksum: str,
    embedding_text_version: str,
    embedding_deployment: str,
    embedding_dimensions: int,
) -> dict[str, Any]:
    filters = product["filter_payload"]
    attributes = product["attributes"]
    payload: dict[str, Any] = {
        "product_id": product["product_id"],
        "sku": product["sku"],
        "name": product["name"],
        "category_id": product["category"]["id"],
        "category_name": product["category"]["name"],
        "brand": product["brand"],
        "brand_normalized": normalize_text(product["brand"]),
        "model": product["model"],
        "model_family": product["model_family"],
        "model_family_normalized": normalize_text(product["model_family"]),
        "color_code": product["color"]["code"],
        "sale_price": float(product["price"]["sale"]),
        "currency": product["price"]["currency"],
        "in_stock": bool(filters["in_stock"]),
        "dataset_version": product["dataset_version"],
        "catalog_checksum": catalog_checksum,
        "embedding_text_version": embedding_text_version,
        "embedding_deployment": embedding_deployment,
        "embedding_dimensions": embedding_dimensions,
    }
    optional_values = {
        "storage_gb": filters.get("storage_gb", attributes.get("storage_gb")),
        "ram_gb": filters.get("ram_gb", attributes.get("ram_gb")),
        "btu": filters.get("btu", attributes.get("btu")),
        "screen_size_in": filters.get("screen_size_in", attributes.get("screen_size_in")),
        "active_noise_cancellation": filters.get(
            "active_noise_cancellation",
            attributes.get("active_noise_cancellation"),
        ),
    }
    connectivity = filters.get("connectivity", attributes.get("connectivity"))
    if connectivity is not None:
        payload["connectivity"] = str(connectivity)
        payload["connectivity_normalized"] = normalize_text(str(connectivity))
    for key, value in optional_values.items():
        if value is not None:
            payload[key] = value
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
    def from_settings(cls, settings: Settings) -> QdrantProductStore:
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
            timeout=30,
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
            count_filter=self.build_filter(args),
            exact=True,
        )
        return result.count

    def search(
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
            query_filter=self.build_filter(args),
            limit=max(candidate_limit, args.limit),
            with_payload=True,
            with_vectors=False,
        )
        hits: list[VectorSearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            product_id = payload.get("product_id")
            if not isinstance(product_id, str):
                raise VectorStoreError("Qdrant nəticəsində product_id yoxdur")
            hits.append(VectorSearchHit(product_id=product_id, score=float(point.score), payload=payload))
        hits.sort(key=lambda hit: (-hit.score, hit.product_id))
        return hits[: args.limit]

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
            )
        vector_size, distance = self._collection_vector_config()
        payloads = self._all_metadata_payloads()
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
        )

    @staticmethod
    def build_filter(args: ProductSearchArguments) -> models.Filter | None:
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
        return models.Filter(must=conditions) if conditions else None

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

    def _all_metadata_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        offset: int | str | UUID | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=[
                    "product_id",
                    "dataset_version",
                    "catalog_checksum",
                    "embedding_text_version",
                    "embedding_deployment",
                    "embedding_dimensions",
                ],
                with_vectors=False,
            )
            payloads.extend(dict(record.payload or {}) for record in records)
            if offset is None:
                return payloads

    @staticmethod
    def _unique_strings(payloads: Sequence[dict[str, Any]], key: str) -> tuple[str, ...]:
        return tuple(
            sorted({value for payload in payloads if isinstance((value := payload.get(key)), str)})
        )
