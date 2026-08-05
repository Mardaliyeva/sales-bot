from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent.locks import SessionLockManager
from app.agent.runtime import AgentRuntime
from app.api.errors import ApiError, api_error_handler, validation_error_handler
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_sessions import router as sessions_router
from app.config import Settings, get_settings
from app.db.repositories import ConversationRepository
from app.db.session import Database
from app.llm.openrouter_client import OpenRouterClient
from app.logging_config import configure_logging
from app.tools.catalog import ProductCatalog
from app.tools.product_search import ProductSearchTool
from app.tools.registry import ToolRegistry

MAX_REQUEST_BYTES = 16 * 1024


def create_app(settings_override: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_override or get_settings()
        configure_logging(settings.log_level)
        catalog = ProductCatalog(settings.product_catalog_path)
        catalog.load()
        database = Database(settings.database_url)
        repository = ConversationRepository(database, settings)
        llm = OpenRouterClient(settings)
        tools = ToolRegistry(ProductSearchTool(catalog), settings.tool_timeout_seconds)

        app.state.settings = settings
        app.state.catalog = catalog
        app.state.database = database
        app.state.repository = repository
        app.state.llm = llm
        app.state.tools = tools
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
    return app


app = create_app()
