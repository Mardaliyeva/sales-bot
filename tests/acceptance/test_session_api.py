from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.deps import get_database, get_repository
from app.api.errors import ApiError, api_error_handler
from app.api.routes_sessions import router


class ReadyDatabase:
    async def is_ready(self) -> bool:
        return True


class SessionRepository:
    async def create_session(self) -> object:
        return SimpleNamespace(
            id=uuid.uuid4(),
            status="active",
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )


@pytest.mark.asyncio
async def test_session_can_be_created_without_request_body() -> None:
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    app.include_router(router)
    app.dependency_overrides[get_database] = ReadyDatabase
    app.dependency_overrides[get_repository] = SessionRepository

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/sessions")

    assert response.status_code == 201
    assert response.json()["status"] == "active"
