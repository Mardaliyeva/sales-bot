from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.documents.corpus import DocumentChunk
from app.vectorstores.qdrant import DEFAULT_UPSERT_BATCH_SIZE, DEFAULT_VECTOR_SIZE, VectorStoreError

DEFAULT_DOCUMENT_COLLECTION_NAME = "sales_bot_documents_v1"
DEFAULT_DOCUMENT_QUERY_CANDIDATES = 20

DOCUMENT_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "chunk_id": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "filename": models.PayloadSchemaType.KEYWORD,
    "source_checksum": models.PayloadSchemaType.KEYWORD,
    "document_checksum": models.PayloadSchemaType.KEYWORD,
    "embedding_text_version": models.PayloadSchemaType.KEYWORD,
    "embedding_deployment": models.PayloadSchemaType.KEYWORD,
    "embedding_dimensions": models.PayloadSchemaType.INTEGER,
    "chunk_index": models.PayloadSchemaType.INTEGER,
}

REQUIRED_DOCUMENT_PAYLOAD_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "filename",
        "title",
        "heading",
        "chunk_index",
        "text",
        "source_checksum",
        "document_checksum",
        "embedding_text_version",
        "embedding_deployment",
        "embedding_dimensions",
    }
)


@dataclass(frozen=True)
class DocumentSearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class DocumentCollectionStatus:
    exists: bool
    collection_name: str
    expected_count: int
    indexed_count: int
    vector_size: int | None
    distance: str | None
    missing_chunk_ids: tuple[str, ...]
    extra_chunk_ids: tuple[str, ...]
    source_checksums: tuple[str, ...]
    embedding_text_versions: tuple[str, ...]
    embedding_deployments: tuple[str, ...]
    embedding_dimensions: tuple[int, ...]
    metadata_matches: bool
    payload_fields_match: bool

    @property
    def ready(self) -> bool:
        return (
            self.exists
            and self.indexed_count == self.expected_count
            and self.vector_size is not None
            and self.distance == models.Distance.COSINE.value
            and not self.missing_chunk_ids
            and not self.extra_chunk_ids
            and len(self.source_checksums) == 1
            and len(self.embedding_text_versions) == 1
            and len(self.embedding_deployments) == 1
            and self.embedding_dimensions == (self.vector_size,)
            and self.metadata_matches
            and self.payload_fields_match
        )


def document_point_id(chunk_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"sales-bot/document/{chunk_id}")


def build_document_payload(
    chunk: DocumentChunk,
    *,
    source_checksum: str,
    embedding_text_version: str,
    embedding_deployment: str,
    embedding_dimensions: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "title": chunk.title,
        "heading": chunk.heading,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "source_checksum": source_checksum,
        "document_checksum": chunk.document_checksum,
        "embedding_text_version": embedding_text_version,
        "embedding_deployment": embedding_deployment,
        "embedding_dimensions": embedding_dimensions,
    }


class QdrantDocumentStore:
    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_DOCUMENT_COLLECTION_NAME,
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
    ) -> QdrantDocumentStore:
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
        return cls(client, collection_name=settings.qdrant_document_collection_name)

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

    def sync_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[list[float]],
        *,
        source_checksum: str,
        embedding_text_version: str,
        embedding_deployment: str,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Sənəd chunk və embedding sayı uyğun deyil")
        if any(len(vector) != self.vector_size for vector in vectors):
            raise VectorStoreError("Qdrant-a yazılan document embedding ölçüsü uyğun deyil")
        points = [
            models.PointStruct(
                id=document_point_id(chunk.chunk_id),
                vector=vector,
                payload=build_document_payload(
                    chunk,
                    source_checksum=source_checksum,
                    embedding_text_version=embedding_text_version,
                    embedding_deployment=embedding_deployment,
                    embedding_dimensions=self.vector_size,
                ),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

        # Qdrant Cloud and the local client can expose UUID point IDs using
        # different Python types (UUID vs string). Compare their canonical
        # string forms so a successful upsert is never mistaken for stale data.
        expected_point_ids = {str(document_point_id(chunk.chunk_id)) for chunk in chunks}
        actual_point_ids = {str(record_id) for record_id, _ in self._all_records()}
        stale = sorted(actual_point_ids - expected_point_ids, key=str)
        if stale:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=stale),
                wait=True,
            )

    def search_candidates(
        self,
        vector: list[float],
        *,
        candidate_limit: int = DEFAULT_DOCUMENT_QUERY_CANDIDATES,
    ) -> list[DocumentSearchHit]:
        if len(vector) != self.vector_size:
            raise VectorStoreError("Document query embedding ölçüsü collection ilə uyğun deyil")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=candidate_limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = [self._record_to_hit(point, score=float(point.score)) for point in response.points]
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:candidate_limit]

    def status(
        self,
        expected_chunk_ids: Sequence[str],
        *,
        expected_source_checksum: str | None = None,
        expected_embedding_text_version: str | None = None,
        expected_embedding_deployment: str | None = None,
        expected_embedding_dimensions: int | None = None,
    ) -> DocumentCollectionStatus:
        expected = set(expected_chunk_ids)
        if not self.client.collection_exists(self.collection_name):
            return DocumentCollectionStatus(
                exists=False,
                collection_name=self.collection_name,
                expected_count=len(expected),
                indexed_count=0,
                vector_size=None,
                distance=None,
                missing_chunk_ids=tuple(sorted(expected)),
                extra_chunk_ids=(),
                source_checksums=(),
                embedding_text_versions=(),
                embedding_deployments=(),
                embedding_dimensions=(),
                metadata_matches=False,
                payload_fields_match=False,
            )
        vector_size, distance = self._collection_vector_config()
        records = self._all_records()
        payloads = [payload for _, payload in records]
        actual = {
            value
            for payload in payloads
            if isinstance((value := payload.get("chunk_id")), str)
        }
        source_checksums = self._unique_strings(payloads, "source_checksum")
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
            expected_source_checksum is None or source_checksums == (expected_source_checksum,),
            expected_embedding_text_version is None
            or text_versions == (expected_embedding_text_version,),
            expected_embedding_deployment is None or deployments == (expected_embedding_deployment,),
            expected_embedding_dimensions is None or dimensions == (expected_embedding_dimensions,),
        )
        return DocumentCollectionStatus(
            exists=True,
            collection_name=self.collection_name,
            expected_count=len(expected),
            indexed_count=len(records),
            vector_size=vector_size,
            distance=distance,
            missing_chunk_ids=tuple(sorted(expected - actual)),
            extra_chunk_ids=tuple(sorted(actual - expected)),
            source_checksums=source_checksums,
            embedding_text_versions=text_versions,
            embedding_deployments=deployments,
            embedding_dimensions=dimensions,
            metadata_matches=all(expected_checks),
            payload_fields_match=all(
                REQUIRED_DOCUMENT_PAYLOAD_FIELDS.issubset(payload) for payload in payloads
            ),
        )

    def _validate_collection(self) -> None:
        size, distance = self._collection_vector_config()
        if size != self.vector_size or distance != models.Distance.COSINE.value:
            raise VectorStoreError(
                "Document Qdrant collection vector konfiqurasiyası uyğun deyil; "
                "yeni collection adı istifadə edin"
            )

    def _collection_vector_config(self) -> tuple[int, str]:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError("Named-vector document collection dəstəklənmir")
        return vectors.size, vectors.distance.value

    def _ensure_payload_indexes(self) -> None:
        info = self.client.get_collection(self.collection_name)
        existing = set(info.payload_schema)
        for field_name, schema in DOCUMENT_PAYLOAD_INDEXES.items():
            if field_name not in existing:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    def _all_records(self) -> list[tuple[int | str | UUID, dict[str, Any]]]:
        records_out: list[tuple[int | str | UUID, dict[str, Any]]] = []
        offset: int | str | UUID | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                records_out.append((record.id, dict(record.payload or {})))
            if offset is None:
                return records_out

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
    def _record_to_hit(record: Any, *, score: float) -> DocumentSearchHit:
        payload = dict(record.payload or {})
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise VectorStoreError("Qdrant document nəticəsində chunk_id yoxdur")
        return DocumentSearchHit(chunk_id=chunk_id, score=score, payload=payload)
