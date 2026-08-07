from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.api.errors import ApiError
from app.api.routes_debug import _legacy_trace, get_debug_trace
from app.db.models import AgentRun, ChatMessage
from app.db.repositories import DebugRunSnapshot


def make_snapshot(*, debug_trace: dict[str, object] | None) -> DebugRunSnapshot:
    session_id = uuid.uuid4()
    run_id = uuid.uuid4()
    final_message = ChatMessage(
        id=uuid.uuid4(),
        session_id=session_id,
        run_id=run_id,
        sequence_no=2,
        role="assistant",
        content="Cavab",
    )
    run = AgentRun(
        id=run_id,
        request_id=uuid.uuid4(),
        session_id=session_id,
        status="completed",
        model="gpt-test",
        tool_count=0,
        model_rounds=1,
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        latency_ms=100,
        debug_trace=debug_trace,
    )
    return DebugRunSnapshot(run=run, messages=[final_message], final_message_id=final_message.id)


class FakeRepository:
    def __init__(self, snapshot: DebugRunSnapshot | None) -> None:
        self.snapshot = snapshot

    async def get_debug_run(self, **_: object) -> DebugRunSnapshot | None:
        return self.snapshot


def make_request(*, enabled: bool, app_env: str = "development") -> SimpleNamespace:
    tools = SimpleNamespace(
        debug_source_state=lambda: {
            "product_catalog": {"configured": True, "product_count": 300},
            "semantic_qdrant": {"configured": True},
            "documents": {"configured": False},
        }
    )
    settings = SimpleNamespace(
        app_env=app_env,
        debug_panel_enabled=enabled,
        qdrant_collection_name="sales_bot_products_semantic_v2",
    )
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings, tools=tools)))


@pytest.mark.asyncio
async def test_full_debug_trace_is_returned_for_matching_session() -> None:
    stored = {
        "trace_version": 1,
        "detail_level": "full",
        "status": "completed",
        "diagnosis": {
            "code": "catalog_not_checked",
            "title": "Birbaşa cavab",
            "detail": "Kataloq yoxlanılmadı.",
            "catalog_checked": False,
            "data_status": "Yoxlanılmayıb",
            "result_count": None,
        },
        "data_sources": {"documents": {"configured": False}},
        "timeline": [],
        "warnings": [],
        "metrics": {"tool_count": 0},
    }
    snapshot = make_snapshot(debug_trace=stored)

    response = await get_debug_trace(
        make_request(enabled=True),
        FakeRepository(snapshot),  # type: ignore[arg-type]
        snapshot.run.session_id,
        request_id=snapshot.run.request_id,
    )

    assert response.request_id == snapshot.run.request_id
    assert response.message_id == snapshot.final_message_id
    assert response.diagnosis is not None
    assert response.diagnosis["code"] == "catalog_not_checked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "app_env"),
    [(False, "development"), (True, "production")],
)
async def test_debug_endpoint_is_hidden_unless_development_and_enabled(
    enabled: bool,
    app_env: str,
) -> None:
    snapshot = make_snapshot(debug_trace=None)
    with pytest.raises(ApiError) as captured:
        await get_debug_trace(
            make_request(enabled=enabled, app_env=app_env),
            FakeRepository(snapshot),  # type: ignore[arg-type]
            snapshot.run.session_id,
            request_id=snapshot.run.request_id,
        )
    assert captured.value.status_code == 404


@pytest.mark.asyncio
async def test_unknown_or_cross_session_trace_returns_not_found() -> None:
    with pytest.raises(ApiError) as captured:
        await get_debug_trace(
            make_request(enabled=True),
            FakeRepository(None),  # type: ignore[arg-type]
            uuid.uuid4(),
            request_id=uuid.uuid4(),
        )
    assert captured.value.code == "debug_trace_not_found"


def test_legacy_trace_reports_partial_data_without_inventing_ranks() -> None:
    snapshot = make_snapshot(debug_trace=None)
    trace = _legacy_trace(
        snapshot,
        {"product_catalog": {"configured": True, "product_count": 300}},
    )

    assert trace["detail_level"] == "legacy_partial"
    assert trace["diagnosis"]["code"] == "legacy_partial"
    assert trace["diagnosis"]["data_status"] == "Yoxlanılmayıb"
    assert all(event.get("retrieval") is None for event in trace["timeline"])
