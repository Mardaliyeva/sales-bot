from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
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
    database: DatabaseDep,
    catalog: CatalogDep,
) -> JSONResponse:
    database_ready = await database.is_ready()
    checks = {"database": database_ready, "catalog": catalog.ready}
    status_code = 200 if all(checks.values()) else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if status_code == 200 else "not_ready", "checks": checks},
    )
