from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_catalog, get_database
from app.db.session import Database
from app.tools.catalog import ProductCatalog

router = APIRouter(tags=["health"])
DatabaseDep = Annotated[Database, Depends(get_database)]
CatalogDep = Annotated[ProductCatalog, Depends(get_catalog)]


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready")
async def ready(
    request: Request,
    database: DatabaseDep,
    catalog: CatalogDep,
) -> JSONResponse:
    database_ready = await database.is_ready()
    vector_status = getattr(request.app.state, "vector_status", None)
    vector_required = bool(getattr(request.app.state, "vector_runtime_required", False))
    vector_ready = bool(
        vector_status
        and (vector_status.get("metadata_matches") or vector_status.get("ready"))
    )
    payload_ready = bool(
        vector_status
        and (vector_status.get("payload_fields_match") or vector_status.get("ready"))
    )
    indexed_ready = bool(
        vector_status
        and (
            vector_status.get("ready")
            or (
                vector_status.get("exists")
                and vector_status.get("indexed_count") == vector_status.get("expected_count")
                and not vector_status.get("missing_product_ids")
                and not vector_status.get("extra_product_ids")
            )
        )
    )
    checks = {
        "database": database_ready,
        "catalog": catalog.ready,
        "qdrant_index": vector_ready and indexed_ready,
        "qdrant_payload_schema": payload_ready,
        "embedding_deployment": vector_ready,
    }
    if not vector_required:
        checks["qdrant_index"] = False
        checks["qdrant_payload_schema"] = False
        checks["embedding_deployment"] = False
    document_required = bool(getattr(request.app.state, "document_runtime_required", False))
    if document_required:
        document_status = getattr(request.app.state, "document_status", None)
        checks["document_qdrant_index"] = bool(
            document_status
            and document_status.get("ready")
            and document_status.get("metadata_matches")
            and document_status.get("payload_fields_match")
        )
    status_code = 200 if all(checks.values()) else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if status_code == 200 else "not_ready",
            "checks": checks,
            "runtime": getattr(request.app.state, "runtime_metadata", {}),
            "vector_error": getattr(request.app.state, "vector_error", None),
            "document_error": getattr(request.app.state, "document_error", None),
        },
    )
