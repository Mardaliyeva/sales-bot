from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from app.documents.corpus import DOCUMENT_TEXT_VERSION, DocumentCorpus
from app.vectorstores.documents import QdrantDocumentStore


def make_corpus(tmp_path: Path) -> DocumentCorpus:
    source = tmp_path / "source"
    source.mkdir()
    (source / "delivery.md").write_text(
        "# Çatdırılma\n\n## Bakı\n\nÇatdırılma bir gün çəkir.\n\n## Region\n\nÜç gün çəkir.",
        encoding="utf-8",
    )
    corpus = DocumentCorpus(source)
    corpus.load()
    return corpus


def test_document_collection_is_idempotent_and_removes_stale_chunks(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    client = QdrantClient(":memory:")
    store = QdrantDocumentStore(client, collection_name="documents_test", vector_size=4)
    store.ensure_collection()
    vectors = [[1.0, 0.0, 0.0, float(index)] for index, _ in enumerate(corpus.chunks)]
    store.sync_chunks(
        corpus.chunks,
        vectors,
        source_checksum=corpus.manifest["source_checksum"],
        embedding_text_version=DOCUMENT_TEXT_VERSION,
        embedding_deployment="embedding-test",
    )
    store.sync_chunks(
        corpus.chunks[:1],
        vectors[:1],
        source_checksum=corpus.manifest["source_checksum"],
        embedding_text_version=DOCUMENT_TEXT_VERSION,
        embedding_deployment="embedding-test",
    )

    status = store.status(
        [corpus.chunks[0].chunk_id],
        expected_source_checksum=corpus.manifest["source_checksum"],
        expected_embedding_text_version=DOCUMENT_TEXT_VERSION,
        expected_embedding_deployment="embedding-test",
        expected_embedding_dimensions=4,
    )
    assert status.ready
    assert status.indexed_count == 1
    assert not status.extra_chunk_ids


def test_document_search_returns_payload_and_score(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    client = QdrantClient(":memory:")
    store = QdrantDocumentStore(client, collection_name="documents_search", vector_size=4)
    store.ensure_collection()
    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in corpus.chunks]
    store.sync_chunks(
        corpus.chunks,
        vectors,
        source_checksum=corpus.manifest["source_checksum"],
        embedding_text_version=DOCUMENT_TEXT_VERSION,
        embedding_deployment="embedding-test",
    )

    hits = store.search_candidates([1.0, 0.0, 0.0, 0.0], candidate_limit=5)
    assert hits
    assert hits[0].payload["title"] == "Çatdırılma"
    assert hits[0].payload["text"]

