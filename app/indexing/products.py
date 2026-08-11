from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError
from qdrant_client.http import models

from app.config import get_settings
from app.embeddings.azure import DEFAULT_TEXT_VERSION, AzureEmbeddingClient, EmbeddingError
from app.tools.catalog import CatalogLoadError, ProductCatalog
from app.vectorstores.qdrant import (
    CollectionStatus,
    QdrantProductStore,
    VectorStoreError,
    build_embedding_text,
)


class ProductEmbeddingBackend(Protocol):
    deployment: str
    dimensions: int

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]: ...


@dataclass(frozen=True)
class IndexResult:
    product_count: int
    status: CollectionStatus


def index_catalog(
    catalog: ProductCatalog,
    embeddings: ProductEmbeddingBackend,
    store: QdrantProductStore,
    *,
    refresh_embeddings: bool = False,
) -> IndexResult:
    if not catalog.ready:
        catalog.load()
    checksum = catalog.manifest["checksums"]["products_sha256"]
    store.ensure_collection()
    vectors = embeddings.embed(
        [build_embedding_text(product) for product in catalog.products],
        text_version=DEFAULT_TEXT_VERSION,
        refresh=refresh_embeddings,
    )
    store.upsert_products(
        catalog.products,
        vectors,
        catalog_checksum=checksum,
        embedding_text_version=DEFAULT_TEXT_VERSION,
        embedding_deployment=embeddings.deployment,
    )
    status = store.status(
        [product["product_id"] for product in catalog.products],
        expected_dataset_version=catalog.manifest["dataset_version"],
        expected_catalog_checksum=checksum,
        expected_embedding_text_version=DEFAULT_TEXT_VERSION,
        expected_embedding_deployment=embeddings.deployment,
        expected_embedding_dimensions=embeddings.dimensions,
    )
    if not status.ready:
        raise VectorStoreError("Qdrant indeksləmə yoxlaması uğursuz oldu")
    return IndexResult(product_count=len(catalog.products), status=status)


def _print_status(status: CollectionStatus) -> None:
    print(f"Collection: {status.collection_name}")
    print(f"Mövcuddur: {'bəli' if status.exists else 'xeyr'}")
    print(f"Məhsul sayı: {status.indexed_count}/{status.expected_count}")
    print(f"Vector: {status.vector_size or '-'} / {status.distance or '-'}")
    print(f"Dataset versiyası: {', '.join(status.dataset_versions) or '-'}")
    print(f"Embedding text versiyası: {', '.join(status.embedding_text_versions) or '-'}")
    print(f"Embedding deployment: {', '.join(status.embedding_deployments) or '-'}")
    print(f"Metadata uyğundur: {'bəli' if status.metadata_matches else 'xeyr'}")
    print(f"Payload field-ləri uyğundur: {'bəli' if status.payload_fields_match else 'xeyr'}")
    print(f"Çatışmayan ID sayı: {len(status.missing_product_ids)}")
    print(f"Artıq ID sayı: {len(status.extra_product_ids)}")
    print(f"Hazırdır: {'bəli' if status.ready else 'xeyr'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Məhsul embedding-lərini Qdrant-a indekslə")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Kataloqu Azure və Qdrant ilə indekslə")
    index_parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="Lokal embedding cache-ni nəzərə alma",
    )
    index_parser.add_argument("--collection-name")
    status_parser = subparsers.add_parser("status", help="Qdrant indeksinin vəziyyətini göstər")
    status_parser.add_argument("--collection-name")
    promote_parser = subparsers.add_parser(
        "promote", help="Yoxlanmış collection-u active alias-a atomik keçir"
    )
    promote_parser.add_argument("--collection-name", required=True)
    promote_parser.add_argument("--alias", default="sales_bot_products_active")
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    embeddings: AzureEmbeddingClient | None = None
    store: QdrantProductStore | None = None
    try:
        settings = get_settings()
        catalog = ProductCatalog(settings.product_catalog_path)
        catalog.load()
        collection_name = getattr(args, "collection_name", None)
        store = QdrantProductStore.from_settings(settings, collection_name=collection_name)
        if args.command == "status":
            status = store.status(
                [product["product_id"] for product in catalog.products],
                expected_dataset_version=catalog.manifest["dataset_version"],
                expected_catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
                expected_embedding_text_version=DEFAULT_TEXT_VERSION,
                expected_embedding_deployment=settings.azure_embedding_model,
                expected_embedding_dimensions=3072,
            )
            _print_status(status)
            return 0 if status.ready else 1

        if args.command == "promote":
            status = store.status(
                [product["product_id"] for product in catalog.products],
                expected_dataset_version=catalog.manifest["dataset_version"],
                expected_catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
                expected_embedding_text_version=DEFAULT_TEXT_VERSION,
                expected_embedding_deployment=settings.azure_embedding_model,
                expected_embedding_dimensions=3072,
            )
            if not status.ready:
                raise VectorStoreError("Hazır olmayan collection active alias-a keçirilə bilməz")
            aliases = store.client.get_aliases().aliases
            operations = []
            if any(alias.alias_name == args.alias for alias in aliases):
                operations.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=args.alias)
                    )
                )
            operations.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=args.collection_name,
                        alias_name=args.alias,
                    )
                )
            )
            store.client.update_collection_aliases(operations)
            print(f"Active alias atomik yeniləndi: {args.alias} -> {args.collection_name}")
            return 0

        embeddings = AzureEmbeddingClient.from_settings(settings)
        result = index_catalog(
            catalog,
            embeddings,
            store,
            refresh_embeddings=args.refresh_embeddings,
        )
        print(f"{result.product_count} məhsul Qdrant-a indeksləndi.")
        _print_status(result.status)
        return 0
    except (CatalogLoadError, EmbeddingError, VectorStoreError, ValidationError) as exc:
        print(f"Xəta: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Xəta: uzaq xidmət əməliyyatı uğursuz oldu", file=sys.stderr)
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
