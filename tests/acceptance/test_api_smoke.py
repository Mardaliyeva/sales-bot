from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.main import create_app


@pytest.mark.asyncio
async def test_liveness_and_validation_do_not_call_provider(settings: object) -> None:
    app = create_app(settings)  # type: ignore[arg-type]
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        live = await client.get("/health/live")
        invalid = await client.post("/v1/chat", json={"session_id": "not-a-uuid", "message": ""})
    assert live.status_code == 200
    assert live.json() == {"status": "live"}
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
