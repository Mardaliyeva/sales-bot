from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_database, get_repository
from app.api.errors import ApiError
from app.api.schemas import SessionCreateRequest, SessionCreateResponse
from app.db.repositories import ConversationRepository
from app.db.session import Database

router = APIRouter(prefix="/v1", tags=["sessions"])
DatabaseDep = Annotated[Database, Depends(get_database)]
RepositoryDep = Annotated[ConversationRepository, Depends(get_repository)]


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session(
    database: DatabaseDep,
    repository: RepositoryDep,
    _: SessionCreateRequest | None = None,
) -> SessionCreateResponse:
    if not await database.is_ready():
        raise ApiError(503, "database_unavailable", "Məlumat bazası hazır deyil.")
    try:
        session = await repository.create_session()
    except Exception as exc:
        raise ApiError(503, "database_unavailable", "Məlumat bazası hazır deyil.") from exc
    return SessionCreateResponse(
        session_id=session.id,
        status="active",
        expires_at=session.expires_at,
    )
