from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent.locks import SessionLockManager
from app.agent.runtime import AgentRuntime
from app.api.errors import ApiError, api_error_handler, validation_error_handler
from app.api.routes_chat import router as chat_router
from app.api.routes_debug import router as debug_router
from app.api.routes_health import router as health_router
from app.api.routes_sessions import router as sessions_router
from app.config import PROJECT_ROOT, Settings, get_settings
from app.db.repositories import ConversationRepository
from app.db.session import Database
from app.documents.baseline import load_document_baseline
from app.documents.corpus import DOCUMENT_TEXT_VERSION, DocumentCorpus
from app.embeddings.azure import DEFAULT_TEXT_VERSION, AzureEmbeddingClient
from app.llm.azure_client import AzureChatClient
from app.logging_config import configure_logging
from app.retrieval.documents import QdrantDocumentSearch
from app.retrieval.qdrant import QdrantProductSearch
from app.tools.catalog import JsonlCatalogBackend
from app.tools.document_search import DocumentSearchTool
from app.tools.product_search import ProductSearchTool
from app.tools.registry import ToolRegistry
from app.vectorstores.documents import QdrantDocumentStore
from app.vectorstores.qdrant import QdrantProductStore

MAX_REQUEST_BYTES = 16 * 1024
RUNTIME_VECTOR_TIMEOUT_SECONDS = 4.0
STARTUP_VECTOR_CHECK_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)
API_SCHEMA_VERSION = "2.6"


async def _scrub_expired_session_contexts(
    repository: ConversationRepository,
    *,
    interval_seconds: int,
) -> None:
    while True:
        try:
            scrubbed = await repository.scrub_expired_session_contexts()
            if scrubbed:
                logger.info("session_context.scrubbed", extra={"session_count": scrubbed})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("session_context.scrub_failed")
        await asyncio.sleep(interval_seconds)


def _git_revision() -> str | None:
    configured = os.getenv("GIT_REVISION")
    if configured:
        return configured.strip() or None
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _vector_status_payload(status: object | None) -> dict[str, object] | None:
    if status is None:
        return None
    if is_dataclass(status):
        payload = asdict(status)
    else:
        payload = {
            key: getattr(status, key)
            for key in (
                "exists",
                "expected_count",
                "indexed_count",
                "metadata_matches",
                "payload_fields_match",
                "missing_product_ids",
                "extra_product_ids",
            )
            if hasattr(status, key)
        }
    payload["ready"] = bool(getattr(status, "ready", False))
    return payload


def _document_status_payload(status: object | None) -> dict[str, object] | None:
    if status is None:
        return None
    payload = asdict(status) if is_dataclass(status) else {}
    payload["ready"] = bool(getattr(status, "ready", False))
    return payload


def _vector_runtime_configured(settings: Settings) -> bool:
    if not settings.customer_azure_openai_endpoint or not settings.qdrant_url:
        return False
    if settings.customer_azure_openai_endpoint == "YOUR_AZURE_ENDPOINT":
        return False
    if settings.qdrant_url == "YOUR_QDRANT_CLOUD_URL":
        return False
    secrets = (settings.customer_azure_openai_api_key, settings.qdrant_api_key)
    return all(
        secret is not None
        and bool(secret.get_secret_value())
        and secret.get_secret_value() not in {"YOUR_AZURE_API_KEY", "YOUR_QDRANT_API_KEY", "CHANGE_ME"}
        for secret in secrets
    )


def create_app(settings_override: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_override or get_settings()
        configure_logging(settings.log_level)
        catalog = JsonlCatalogBackend(settings.product_catalog_path)
        catalog.load()
        database = Database(settings.database_url)
        repository = ConversationRepository(database, settings)
        llm = AzureChatClient(settings)
        embeddings: AzureEmbeddingClient | None = None
        vector_store: QdrantProductStore | None = None
        document_store: QdrantDocumentStore | None = None
        vector_status = None
        vector_error: str | None = None
        document_corpus: DocumentCorpus | None = None
        document_status = None
        document_error: str | None = None
        document_min_score: float | None = None
        context_scrub_task: asyncio.Task[None] | None = None
        if _vector_runtime_configured(settings):
            try:
                embeddings = AzureEmbeddingClient.from_settings(
                    settings,
                    timeout_seconds=RUNTIME_VECTOR_TIMEOUT_SECONDS,
                    max_attempts=1,
                )
                vector_store = QdrantProductStore.from_settings(
                    settings,
                    timeout_seconds=RUNTIME_VECTOR_TIMEOUT_SECONDS,
                )
                vector_status = await asyncio.wait_for(
                    asyncio.to_thread(
                        vector_store.status,
                        [product["product_id"] for product in catalog.products],
                        expected_dataset_version=catalog.manifest["dataset_version"],
                        expected_catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
                        expected_embedding_text_version=DEFAULT_TEXT_VERSION,
                        expected_embedding_deployment=settings.azure_embedding_model,
                        expected_embedding_dimensions=embeddings.dimensions,
                    ),
                    timeout=STARTUP_VECTOR_CHECK_TIMEOUT_SECONDS,
                )
                if not vector_status.ready:
                    raise RuntimeError("Qdrant product collection runtime üçün hazır deyil")
                logger.info(
                    "product_search.semantic_enabled",
                    extra={"indexed_count": vector_status.indexed_count},
                )
            except Exception as exc:
                vector_error = type(exc).__name__
                logger.warning(
                    "product_search.semantic_disabled",
                    extra={"error_type": type(exc).__name__},
                )
                if embeddings is not None:
                    embeddings.close()
                if vector_store is not None:
                    vector_store.close()
                embeddings = None
                vector_store = None

        if settings.document_search_enabled:
            try:
                if embeddings is None:
                    raise RuntimeError("Azure embedding runtime hazır deyil")
                document_corpus = DocumentCorpus(settings.documents_path)
                document_corpus.load()
                document_store = QdrantDocumentStore.from_settings(
                    settings,
                    timeout_seconds=RUNTIME_VECTOR_TIMEOUT_SECONDS,
                )
                document_status = await asyncio.wait_for(
                    asyncio.to_thread(
                        document_store.status,
                        [chunk.chunk_id for chunk in document_corpus.chunks],
                        expected_source_checksum=document_corpus.manifest["source_checksum"],
                        expected_embedding_text_version=DOCUMENT_TEXT_VERSION,
                        expected_embedding_deployment=settings.azure_embedding_model,
                        expected_embedding_dimensions=embeddings.dimensions,
                    ),
                    timeout=STARTUP_VECTOR_CHECK_TIMEOUT_SECONDS,
                )
                if not document_status.ready:
                    raise RuntimeError("Qdrant document collection runtime üçün hazır deyil")
                baseline = load_document_baseline(
                    settings.document_baseline_path,
                    expected_source_checksum=document_corpus.manifest["source_checksum"],
                    expected_collection_name=settings.qdrant_document_collection_name,
                    expected_embedding_deployment=settings.azure_embedding_model,
                    expected_embedding_dimensions=embeddings.dimensions,
                )
                document_min_score = baseline.min_score
                logger.info(
                    "document_search.semantic_enabled",
                    extra={
                        "document_count": len(document_corpus.documents),
                        "chunk_count": len(document_corpus.chunks),
                    },
                )
            except Exception as exc:
                document_error = type(exc).__name__
                logger.warning(
                    "document_search.semantic_disabled",
                    extra={"error_type": type(exc).__name__},
                )
                if document_store is not None:
                    document_store.close()
                document_store = None

        started_at = datetime.now(UTC).isoformat()
        runtime_metadata = {
            "runtime_instance_id": str(uuid4()),
            "startup_time": started_at,
            "git_revision": _git_revision(),
            "api_schema_version": API_SCHEMA_VERSION,
            "catalog_checksum": catalog.manifest["checksums"]["products_sha256"],
            "catalog_dataset_version": catalog.manifest["dataset_version"],
            "active_collection": settings.qdrant_collection_name,
            "embedding_deployment": settings.azure_embedding_model,
            "document_collection": settings.qdrant_document_collection_name,
            "document_source_checksum": (
                document_corpus.manifest.get("source_checksum") if document_corpus else None
            ),
        }

        product_search = QdrantProductSearch(
            catalog,
            embeddings,
            vector_store,
            alternative_min_score=settings.alternative_min_score,
            entity_resolution_min_score=settings.entity_resolution_min_score,
            entity_resolution_margin=settings.entity_resolution_margin,
        )
        document_search = QdrantDocumentSearch(
            embeddings,
            document_store,
            enabled=settings.document_search_enabled,
            ready=bool(document_status and document_status.ready and document_min_score is not None),
            min_score=document_min_score,
            source_checksum=(
                document_corpus.manifest.get("source_checksum") if document_corpus else None
            ),
            document_count=len(document_corpus.documents) if document_corpus else 0,
            chunk_count=len(document_corpus.chunks) if document_corpus else 0,
            unavailable_reason=document_error,
        )
        document_tool = (
            DocumentSearchTool(document_search) if settings.document_search_enabled else None
        )
        tools = ToolRegistry(
            ProductSearchTool(product_search),
            settings.tool_timeout_seconds,
            document_search=document_tool,
        )

        app.state.settings = settings
        app.state.catalog = catalog
        app.state.database = database
        app.state.repository = repository
        app.state.llm = llm
        app.state.tools = tools
        app.state.product_search = product_search
        app.state.document_search = document_search
        app.state.runtime_metadata = runtime_metadata
        app.state.vector_runtime_required = _vector_runtime_configured(settings)
        app.state.vector_status = _vector_status_payload(vector_status)
        app.state.vector_error = vector_error
        app.state.document_runtime_required = settings.document_search_enabled
        app.state.document_status = _document_status_payload(document_status)
        app.state.document_error = document_error
        app.state.lock_manager = SessionLockManager()
        app.state.agent_runtime = AgentRuntime(
            settings=settings,
            repository=repository,
            llm=llm,
            tools=tools,
            runtime_metadata=runtime_metadata,
        )
        await repository.scrub_expired_session_contexts()
        context_scrub_task = asyncio.create_task(
            _scrub_expired_session_contexts(
                repository,
                interval_seconds=settings.session_context_scrub_interval_seconds,
            ),
            name="session-context-scrubber",
        )
        try:
            yield
        finally:
            if context_scrub_task is not None:
                context_scrub_task.cancel()
                with suppress(asyncio.CancelledError):
                    await context_scrub_task
            if embeddings is not None:
                embeddings.close()
            if vector_store is not None:
                vector_store.close()
            if document_store is not None:
                document_store.close()
            await llm.close()
            await database.dispose()

    app = FastAPI(
        title="Sales Bot API",
        version=API_SCHEMA_VERSION,
        lifespan=lifespan,
    )
    app.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]

    @app.middleware("http")
    async def enforce_request_size(request: Request, call_next: object) -> JSONResponse:
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                too_large = int(raw_length) > MAX_REQUEST_BYTES
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={
                        "request_id": None,
                        "error": {"code": "request_too_large", "message": "Sorğu çox böyükdür."},
                    },
                )
        return await call_next(request)  # type: ignore[operator]

    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(debug_router)
    return app


app = create_app()
