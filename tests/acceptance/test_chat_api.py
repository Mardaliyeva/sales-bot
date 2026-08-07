from __future__ import annotations

import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.agent.locks import SessionLockManager
from app.agent.types import AgentResult
from app.api.deps import get_agent_runtime, get_database, get_lock_manager, get_repository
from app.api.errors import ApiError, api_error_handler
from app.api.routes_chat import router


class ReadyDatabase:
    async def is_ready(self) -> bool:
        return True


class ChatRepository:
    async def start_run(self, session_id: uuid.UUID, request_id: uuid.UUID, message: str) -> tuple:
        del request_id, message
        return SimpleNamespace(id=uuid.uuid4()), None, SimpleNamespace(id=session_id)


class FakeRuntime:
    def __init__(self, presentation: dict | None) -> None:
        self.presentation = presentation

    async def run(self, **_: object) -> AgentResult:
        return AgentResult(
            message_id=uuid.UUID("33333333-3333-4333-8333-333333333333"),
            answer="Birinci seçim qiymət və zəmanət baxımından daha uyğundur.",
            used_tools=["product_search"] if self.presentation else [],
            presentation=self.presentation,
        )


def product_presentation() -> dict:
    return {
        "type": "product_cards",
        "title": "1200 AZN büdcəyə uyğun 2 məhsul tapdım",
        "total": 2,
        "shown_count": 2,
        "recommended_product_id": "prd_televisions_008",
        "items": [
            {
                "product_id": "prd_televisions_008",
                "name": "Samsung Q70D QLED 4K",
                "sku": "SYN-TV-SMS-008",
                "price": 569.99,
                "currency": "AZN",
                "stock_status": "in_stock",
                "rating": 5.0,
                "warranty_months": 36,
                "highlights": ["55\" ekran QLED", "8K UHD", "120 Hz", "HDR yoxdur", "Tizen"],
                "budget_remaining": 630.01,
            }
        ],
    }


async def post_chat(presentation: dict | None) -> httpx.Response:
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    app.include_router(router)
    app.dependency_overrides[get_database] = ReadyDatabase
    app.dependency_overrides[get_repository] = ChatRepository
    app.dependency_overrides[get_agent_runtime] = lambda: FakeRuntime(presentation)
    app.dependency_overrides[get_lock_manager] = SessionLockManager

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/v1/chat",
            json={
                "session_id": "11111111-1111-4111-8111-111111111111",
                "message": "1200 AZN büdcəyə 55 düym TV göstər",
            },
        )


@pytest.mark.asyncio
async def test_chat_response_exposes_product_cards() -> None:
    response = await post_chat(product_presentation())

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Birinci seçim")
    assert payload["presentation"]["type"] == "product_cards"
    assert payload["presentation"]["items"][0]["sku"] == "SYN-TV-SMS-008"
    assert payload["presentation"]["items"][0]["budget_remaining"] == 630.01
    assert payload["presentation"]["result_kind"] == "matches"
    assert payload["presentation"]["items"][0]["differences"] == []


@pytest.mark.asyncio
async def test_chat_response_exposes_alternative_metadata() -> None:
    presentation = product_presentation()
    presentation.update(
        {
            "result_kind": "alternatives",
            "requested_label": "Samsung Future TV",
            "title": "Samsung Future TV tapılmadı — yaxın alternativlər",
            "relaxed_fields": ["color_code"],
        }
    )
    presentation["items"][0]["differences"] = ["Rəng fərqlidir: Qara"]

    response = await post_chat(presentation)

    assert response.status_code == 200
    payload = response.json()["presentation"]
    assert payload["result_kind"] == "alternatives"
    assert payload["requested_label"] == "Samsung Future TV"
    assert payload["relaxed_fields"] == ["color_code"]
    assert payload["items"][0]["differences"] == ["Rəng fərqlidir: Qara"]


@pytest.mark.asyncio
async def test_chat_response_omits_optional_presentation_for_plain_text() -> None:
    response = await post_chat(None)

    assert response.status_code == 200
    assert "presentation" not in response.json()
