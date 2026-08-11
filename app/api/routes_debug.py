from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_repository
from app.api.errors import ApiError
from app.api.schemas import DebugTraceResponse
from app.db.repositories import ConversationRepository, DebugRunSnapshot

router = APIRouter(prefix="/v1/debug", tags=["debug"])
RepositoryDep = Annotated[ConversationRepository, Depends(get_repository)]


@router.get("/traces", response_model=DebugTraceResponse)
async def get_debug_trace(
    request: Request,
    repository: RepositoryDep,
    session_id: UUID,
    request_id: UUID | None = None,
    message_id: UUID | None = None,
) -> DebugTraceResponse:
    settings = request.app.state.settings
    if settings.app_env.casefold() != "development" or not settings.debug_panel_enabled:
        raise ApiError(404, "debug_not_found", "Debug endpoint mövcud deyil.")
    if (request_id is None) == (message_id is None):
        raise ApiError(
            422,
            "invalid_debug_lookup",
            "Dəqiq bir request_id və ya message_id verilməlidir.",
        )

    snapshot = await repository.get_debug_run(
        session_id=session_id,
        request_id=request_id,
        message_id=message_id,
    )
    if snapshot is None:
        raise ApiError(404, "debug_trace_not_found", "Bu cavab üçün debug trace tapılmadı.")

    if snapshot.run.debug_trace is not None:
        payload = dict(snapshot.run.debug_trace)
        payload.update(
            {
                "request_id": snapshot.run.request_id,
                "session_id": snapshot.run.session_id,
                "message_id": snapshot.final_message_id,
                "status": snapshot.run.status,
            }
        )
    else:
        payload = _legacy_trace(snapshot, _current_source_state(request))
    return DebugTraceResponse.model_validate(payload)


def _current_source_state(request: Request) -> dict[str, Any]:
    tools = request.app.state.tools
    source_state = tools.debug_source_state()
    semantic = source_state.get("semantic_qdrant")
    if isinstance(semantic, dict) and semantic.get("configured"):
        semantic["collection"] = request.app.state.settings.qdrant_collection_name
    documents = source_state.get("documents")
    if isinstance(documents, dict) and documents.get("configured"):
        documents["collection"] = request.app.state.settings.qdrant_document_collection_name
    return source_state


def _legacy_trace(
    snapshot: DebugRunSnapshot,
    data_sources: dict[str, Any],
) -> dict[str, Any]:
    tool_messages = [message for message in snapshot.messages if message.role == "tool"]
    product_messages = [message for message in tool_messages if message.tool_name == "product_search"]
    document_messages = [message for message in tool_messages if message.tool_name == "document_search"]
    last_result = product_messages[-1].tool_result if product_messages else None
    result_items = (last_result or {}).get("items", []) if last_result else []
    if document_messages:
        document_result = document_messages[-1].tool_result or {}
        document_chunks = document_result.get("chunks", [])
        if document_result.get("status") == "success":
            data_status = (
                f"Tapılıb: {len(document_chunks)}" if document_chunks else "Uyğun nəticə yoxdur"
            )
            observed_outcome = "document_chunks_found" if document_chunks else "no_document_match"
        else:
            data_status = "Müəyyən edilə bilmədi"
            observed_outcome = "document_search_failed"
    elif not product_messages:
        data_status = "Yoxlanılmayıb"
        observed_outcome = "catalog_not_checked"
    elif last_result and last_result.get("status") == "success":
        data_status = f"Tapılıb: {len(result_items)}" if result_items else "Uyğun nəticə yoxdur"
        observed_outcome = "products_found" if result_items else "no_retrieval_match"
    else:
        data_status = "Müəyyən edilə bilmədi"
        observed_outcome = "tool_error"

    timeline: list[dict[str, Any]] = [
        {
            "stage": "legacy_run_summary",
            "status": snapshot.run.status,
            "detail": "Bu run tam debug trace əlavə edilməzdən əvvəl yaradılıb.",
        }
    ]
    for message in tool_messages:
        result = message.tool_result or {}
        items = result.get("items", []) if result.get("status") == "success" else []
        chunks = result.get("chunks", []) if result.get("status") == "success" else []
        timeline.append(
            {
                "stage": message.tool_name or "tool",
                "status": "completed" if result.get("status") == "success" else "failed",
                "arguments": message.tool_arguments or {},
                "result": {
                    "status": result.get("status"),
                    "code": result.get("code"),
                    "total": result.get("total"),
                    "returned_products": [
                        {"product_id": item.get("product_id"), "name": item.get("name")}
                        for item in items
                    ],
                    "returned_chunks": [
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "document_id": chunk.get("document_id"),
                            "title": chunk.get("title"),
                            "heading": chunk.get("heading"),
                            "score": chunk.get("score"),
                        }
                        for chunk in chunks
                        if isinstance(chunk, dict)
                    ],
                },
                "retrieval": None,
            }
        )

    return {
        "trace_version": 1,
        "detail_level": "legacy_partial",
        "request_id": snapshot.run.request_id,
        "session_id": snapshot.run.session_id,
        "message_id": snapshot.final_message_id,
        "status": snapshot.run.status,
        "model": {"deployment": snapshot.run.model},
        "diagnosis": {
            "code": "legacy_partial",
            "title": "Köhnə run üçün qismən məlumat",
            "detail": "Namizəd rank və score-ları əvvəldən saxlanılmadığı üçün göstərilə bilmir.",
            "catalog_checked": bool(product_messages),
            "documents_checked": bool(document_messages),
            "data_status": data_status,
            "result_count": len(result_items) if last_result else None,
            "observed_outcome": observed_outcome,
        },
        "data_sources": data_sources,
        "timeline": timeline,
        "warnings": [],
        "metrics": {
            "model_rounds": snapshot.run.model_rounds,
            "tool_count": snapshot.run.tool_count,
            "input_tokens": snapshot.run.input_tokens,
            "output_tokens": snapshot.run.output_tokens,
            "reasoning_tokens": snapshot.run.reasoning_tokens,
            "latency_ms": snapshot.run.latency_ms,
        },
    }
