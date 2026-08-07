from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
from app.config import Settings, get_settings
from app.db.repositories import ConversationRepository
from app.db.session import Database
from app.embeddings.azure import DEFAULT_TEXT_VERSION, AzureEmbeddingClient
from app.llm.azure_client import AzureChatClient
from app.logging_config import configure_logging
from app.retrieval.qdrant import QdrantProductSearch
from app.tools.catalog import ProductCatalog
from app.tools.product_search import ProductSearchTool
from app.tools.registry import ToolRegistry
from app.vectorstores.qdrant import QdrantProductStore

MAX_REQUEST_BYTES = 16 * 1024
RUNTIME_VECTOR_TIMEOUT_SECONDS = 4.0
STARTUP_VECTOR_CHECK_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


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
        catalog = ProductCatalog(settings.product_catalog_path)
        catalog.load()
        database = Database(settings.database_url)
        repository = ConversationRepository(database, settings)
        llm = AzureChatClient(settings)
        embeddings: AzureEmbeddingClient | None = None
        vector_store: QdrantProductStore | None = None
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
                status = await asyncio.wait_for(
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
                if not status.ready:
                    raise RuntimeError("Qdrant product collection runtime üçün hazır deyil")
                logger.info(
                    "product_search.semantic_enabled",
                    extra={"indexed_count": status.indexed_count},
                )
            except Exception as exc:
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

        product_search = QdrantProductSearch(
            catalog,
            embeddings,
            vector_store,
            alternative_min_score=settings.alternative_min_score,
        )
        tools = ToolRegistry(ProductSearchTool(product_search), settings.tool_timeout_seconds)

        app.state.settings = settings
        app.state.catalog = catalog
        app.state.database = database
        app.state.repository = repository
        app.state.llm = llm
        app.state.tools = tools
        app.state.product_search = product_search
        app.state.lock_manager = SessionLockManager()
        app.state.agent_runtime = AgentRuntime(
            settings=settings,
            repository=repository,
            llm=llm,
            tools=tools,
        )
        try:
            yield
        finally:
            if embeddings is not None:
                embeddings.close()
            if vector_store is not None:
                vector_store.close()
            await llm.close()
            await database.dispose()

    app = FastAPI(
        title="Sales Bot API",
        version="0.1.0",
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
