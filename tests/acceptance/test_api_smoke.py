from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from app import main as main_module
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


@pytest.mark.asyncio
@pytest.mark.parametrize("status_ready", [True, False])
async def test_vector_startup_is_optional_and_never_blocks_api(
    settings: object,
    monkeypatch: pytest.MonkeyPatch,
    status_ready: bool,
) -> None:
    runtime_settings = settings.model_copy(  # type: ignore[union-attr]
        update={
            "customer_azure_openai_endpoint": "https://azure.example.com",
            "customer_azure_openai_api_key": SecretStr("azure-test-key"),
            "qdrant_url": "https://qdrant.example.com",
            "qdrant_api_key": SecretStr("qdrant-test-key"),
        }
    )

    class FakeEmbeddings:
        dimensions = 3072

        def close(self) -> None:
            pass

    class FakeStatus:
        ready = status_ready
        indexed_count = 300

    class FakeStore:
        def status(self, *args: object, **kwargs: object) -> FakeStatus:
            del args, kwargs
            return FakeStatus()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        main_module.AzureEmbeddingClient,
        "from_settings",
        lambda *args, **kwargs: FakeEmbeddings(),
    )
    monkeypatch.setattr(
        main_module.QdrantProductStore,
        "from_settings",
        lambda *args, **kwargs: FakeStore(),
    )

    app = create_app(runtime_settings)
    async with LifespanManager(app):
        assert app.state.product_search.semantic_enabled is status_ready
