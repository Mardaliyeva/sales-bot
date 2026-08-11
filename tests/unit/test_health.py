from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.api.routes_health import ready


class ReadyDatabase:
    async def is_ready(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_enabled_document_search_participates_in_readiness() -> None:
    vector_status = {
        "ready": True,
        "metadata_matches": True,
        "payload_fields_match": True,
        "exists": True,
        "indexed_count": 300,
        "expected_count": 300,
        "missing_product_ids": [],
        "extra_product_ids": [],
    }
    state = SimpleNamespace(
        vector_status=vector_status,
        vector_runtime_required=True,
        vector_error=None,
        document_runtime_required=True,
        document_status={
            "ready": False,
            "metadata_matches": False,
            "payload_fields_match": True,
        },
        document_error="DocumentBaselineError",
        runtime_metadata={},
    )
    response = await ready(
        SimpleNamespace(app=SimpleNamespace(state=state)),  # type: ignore[arg-type]
        ReadyDatabase(),  # type: ignore[arg-type]
        SimpleNamespace(ready=True),  # type: ignore[arg-type]
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["checks"]["document_qdrant_index"] is False
    assert payload["document_error"] == "DocumentBaselineError"

