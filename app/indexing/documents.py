from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from app.config import Settings
from app.documents.corpus import DOCUMENT_TEXT_VERSION, DocumentCorpus, DocumentCorpusError
from app.embeddings.azure import DEFAULT_DIMENSIONS, AzureEmbeddingClient, EmbeddingError
from app.vectorstores.documents import DocumentCollectionStatus, QdrantDocumentStore
from app.vectorstores.qdrant import VectorStoreError


def _load_corpus(settings: Settings) -> DocumentCorpus:
    corpus = DocumentCorpus(settings.documents_path)
    corpus.load()
    return corpus


def _status(
    store: QdrantDocumentStore,
    corpus: DocumentCorpus,
    settings: Settings,
) -> DocumentCollectionStatus:
    return store.status(
        [chunk.chunk_id for chunk in corpus.chunks],
        expected_source_checksum=corpus.manifest["source_checksum"],
        expected_embedding_text_version=DOCUMENT_TEXT_VERSION,
        expected_embedding_deployment=settings.azure_embedding_model,
        expected_embedding_dimensions=DEFAULT_DIMENSIONS,
    )


def _print_status(status: DocumentCollectionStatus, corpus: DocumentCorpus) -> None:
    print(f"Collection: {status.collection_name}")
    print(f"Mövcuddur: {'bəli' if status.exists else 'xeyr'}")
    print(f"Sənəd sayı: {len(corpus.documents)}")
    print(f"Chunk sayı: {status.indexed_count}/{status.expected_count}")
    print(f"Vector: {status.vector_size or '-'} / {status.distance or '-'}")
    print(f"Source checksum: {corpus.manifest['source_checksum']}")
    print(f"Embedding text versiyası: {', '.join(status.embedding_text_versions) or '-'}")
    print(f"Embedding deployment: {', '.join(status.embedding_deployments) or '-'}")
    print(f"Metadata uyğundur: {'bəli' if status.metadata_matches else 'xeyr'}")
    print(f"Payload field-ləri uyğundur: {'bəli' if status.payload_fields_match else 'xeyr'}")
    print(f"Çatışmayan chunk sayı: {len(status.missing_chunk_ids)}")
    print(f"Artıq chunk sayı: {len(status.extra_chunk_ids)}")
    print(f"Hazırdır: {'bəli' if status.ready else 'xeyr'}")


def index_documents(settings: Settings, *, refresh_embeddings: bool = False) -> int:
    corpus = _load_corpus(settings)
    embeddings = AzureEmbeddingClient.from_settings(settings)
    store = QdrantDocumentStore.from_settings(settings)
    try:
        store.ensure_collection()
        vectors = embeddings.embed(
            [chunk.embedding_text for chunk in corpus.chunks],
            text_version=DOCUMENT_TEXT_VERSION,
            refresh=refresh_embeddings,
        )
        store.sync_chunks(
            corpus.chunks,
            vectors,
            source_checksum=corpus.manifest["source_checksum"],
            embedding_text_version=DOCUMENT_TEXT_VERSION,
            embedding_deployment=embeddings.deployment,
        )
        status = _status(store, corpus, settings)
        if not status.ready:
            _print_status(status, corpus)
            return 1
        manifest_path = corpus.write_manifest()
        print(f"{len(corpus.documents)} sənəd və {len(corpus.chunks)} chunk Qdrant-a indeksləndi.")
        print(f"Manifest: {manifest_path}")
        _print_status(status, corpus)
        return 0
    finally:
        embeddings.close()
        store.close()


def document_status(settings: Settings) -> int:
    corpus = _load_corpus(settings)
    store = QdrantDocumentStore.from_settings(settings)
    try:
        status = _status(store, corpus, settings)
        _print_status(status, corpus)
        return 0 if status.ready else 1
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Markdown sənədləri Qdrant-da idarə edir")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Sənədləri embedding edib Qdrant-a yazır")
    index_parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="Lokal embedding cache-ni nəzərə almır",
    )
    subparsers.add_parser("status", help="Document collection vəziyyətini yoxlayır")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()  # type: ignore[call-arg]
        if args.command == "index":
            return index_documents(settings, refresh_embeddings=args.refresh_embeddings)
        return document_status(settings)
    except (DocumentCorpusError, EmbeddingError, VectorStoreError, ValidationError) as exc:
        print(f"Xəta: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Xəta: uzaq xidmət əməliyyatı uğursuz oldu", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

