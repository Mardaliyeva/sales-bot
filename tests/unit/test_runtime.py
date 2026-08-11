from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.runtime import AgentRuntime
from app.agent.types import AgentRuntimeError
from app.db.models import AgentRun, ChatSession
from app.llm.azure_client import ProviderError
from app.llm.schemas import (
    AssistantMessage,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionUsage,
    ToolCall,
    ToolFunctionCall,
)


class FakeRepository:
    def __init__(self) -> None:
        self.tool_exchanges: list[dict[str, Any]] = []
        self.completed: dict[str, Any] | None = None
        self.failed: dict[str, Any] | None = None

    async def get_final_history(self, *_: object, **__: object) -> list[object]:
        return []

    async def store_tool_exchange(self, **kwargs: Any) -> None:
        self.tool_exchanges.append(kwargs)

    async def complete_run(self, **kwargs: Any) -> object:
        self.completed = kwargs
        return SimpleNamespace(id=uuid.uuid4())

    async def fail_run(self, **kwargs: Any) -> None:
        self.failed = kwargs


class FakeLlm:
    def __init__(self, responses: list[ChatCompletionResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> ChatCompletionResponse:
        self.calls.append(kwargs)
        return next(self.responses)


class FakeTools:
    def specs(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "product_search"}}]

    async def execute(self, _: str, __: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "total": 1,
            "applied_filters": {"max_price": 3000},
            "items": [
                {
                    "product_id": "prd_smartphones_001",
                    "sku": "SYN-PH-APL-001",
                    "name": "Apple iPhone 16",
                    "category_id": "smartphones",
                    "sale_price": 2499.99,
                    "currency": "AZN",
                    "stock_status": "in_stock",
                    "warranty_months": 24,
                    "rating": 4.8,
                    "attributes": {
                        "storage_gb": 128,
                        "ram_gb": 8,
                        "main_camera_mp": 48,
                        "network": "5G",
                        "operating_system": "iOS",
                    },
                }
            ],
        }


class AlternativeFakeTools(FakeTools):
    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await super().execute(name, arguments)
        result.update(
            {
                "match_status": "alternatives",
                "requested_label": "iPhone 19",
                "strict_total": 0,
                "total": 1,
                "relaxed_fields": ["color_code"],
            }
        )
        result["items"][0]["differences"] = ["Model fərqlidir: iPhone 16"]
        return result


class DocumentFakeTools(FakeTools):
    def specs(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "product_search"}},
            {"type": "function", "function": {"name": "document_search"}},
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "document_search":
            return {
                "status": "success",
                "match_status": "found",
                "total": 1,
                "min_score": 0.6,
                "chunks": [
                    {
                        "chunk_id": "delivery:0001",
                        "document_id": "delivery",
                        "title": "Çatdırılma",
                        "heading": "Bakı",
                        "text": "Çatdırılma pulsuzdur.",
                        "score": 0.8,
                    }
                ],
            }
        return await super().execute(name, arguments)


def response(message: AssistantMessage, response_id: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=response_id,
        choices=[CompletionChoice(message=message)],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=3,
            completion_tokens_details={"reasoning_tokens": 1},
        ),
    )


def make_run_and_session() -> tuple[AgentRun, ChatSession]:
    session_id = uuid.uuid4()
    run = AgentRun(
        id=uuid.uuid4(),
        request_id=uuid.uuid4(),
        session_id=session_id,
        model="openai/gpt-5.4-mini",
    )
    session = ChatSession(
        id=session_id,
        mode_name="ecommerce_assistant_v1",
        model="openai/gpt-5.4-mini",
        reasoning_effort="low",
        max_tool_count=3,
        context={"last_product_ids": []},
    )
    return run, session


def test_semantic_plan_cache_prunes_entries_at_session_expiry(settings) -> None:
    runtime = AgentRuntime(
        settings=settings,
        repository=FakeRepository(),
        llm=FakeLlm([]),
        tools=FakeTools(),
    )
    runtime._semantic_plan_cache["expired"] = ("session-a", 10.0, {"operation": "lookup"})
    runtime._semantic_plan_cache["active"] = ("session-b", 30.0, {"operation": "discover"})

    runtime._prune_semantic_plan_cache(now=20.0)

    assert list(runtime._semantic_plan_cache) == ["active"]


def test_not_found_status_uses_deterministic_answer() -> None:
    answer = AgentRuntime._guard_product_answer(
        "Uyğun məhsullar tapdım.",
        {
            "status": "success",
            "match_status": "not_found",
            "requested_label": "Future Phone 99",
            "items": [],
        },
    )

    assert answer == (
        "Future Phone 99 kataloqda tapılmadı və etibarlı alternativ müəyyən edilmədi."
    )


def test_internal_payload_terms_are_not_reflected_without_tool_result() -> None:
    answer = AgentRuntime._guard_product_answer(
        "filter_payload və embedding_text sahələrini göstərə bilmərəm",
        None,
    )

    assert "filter_payload" not in answer
    assert "embedding_text" not in answer
    assert "JSON" in answer


@pytest.mark.asyncio
async def test_direct_answer_uses_no_tool(settings: object) -> None:
    repository = FakeRepository()
    llm = FakeLlm([response(AssistantMessage(content="Salam! Sizə necə kömək edim?"), "r1")])
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()
    result = await runtime.run(run=run, session=session, user_message="Salam")
    assert result.used_tools == []
    assert result.presentation is None
    assert repository.completed is not None
    assert len(llm.calls) == 1
    trace = repository.completed["debug_trace"]
    assert trace["diagnosis"]["code"] == "catalog_not_checked"
    assert trace["diagnosis"]["data_status"] == "Yoxlanılmayıb"
    assert trace["trace_version"] == 5
    assert trace["decision_explanation"]["basis"] == "direct_answer"
    assert trace["memory_transition"]["action"] == "preserve"
    assert repository.completed["session_memory"]["revision"] == 0


@pytest.mark.asyncio
async def test_tool_result_and_reasoning_are_returned_to_same_llm(settings: object) -> None:
    repository = FakeRepository()
    tool_call = ToolCall(
        id="call_1",
        function=ToolFunctionCall(
            name="product_search",
            arguments='{"query":"qara iPhone","brand":"Apple","color_code":"black"}',
        ),
    )
    llm = FakeLlm(
        [
            response(
                AssistantMessage(
                    content=None,
                    tool_calls=[tool_call],
                    reasoning_details=[{"type": "reasoning.encrypted", "data": "opaque"}],
                ),
                "r1",
            ),
            response(AssistantMessage(content="Bir uyğun məhsul tapdım."), "r2"),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()
    result = await runtime.run(run=run, session=session, user_message="Qara iPhone göstər")

    assert result.used_tools == ["product_search"]
    assert len(repository.tool_exchanges) == 1
    assert "reasoning_details" not in repository.tool_exchanges[0]
    second_messages = llm.calls[1]["messages"]
    assistant_trace = second_messages[-2]
    assert assistant_trace["reasoning_details"][0]["data"] == "opaque"
    assert second_messages[-1]["role"] == "tool"
    assert repository.completed is not None
    assert repository.completed["last_product_ids"] == ["prd_smartphones_001"]
    assert result.presentation is not None
    assert result.presentation["recommended_product_id"] == "prd_smartphones_001"
    assert result.presentation["items"][0]["budget_remaining"] == 500.01
    trace_dump = json.dumps(repository.completed["debug_trace"])
    assert repository.completed["debug_trace"]["diagnosis"]["code"] == "products_found"
    assert "reasoning_details" not in trace_dump
    assert "opaque" not in trace_dump
    assert repository.completed["debug_trace"]["decision_explanation"]["basis"] == "product_search"
    assert repository.completed["debug_trace"]["memory_transition"]["revision_after"] == 1
    assert repository.completed["session_memory"]["revision"] == 1


@pytest.mark.asyncio
async def test_document_search_result_is_grounded_and_traced(settings: object) -> None:
    repository = FakeRepository()
    tool_call = ToolCall(
        id="call_document",
        function=ToolFunctionCall(
            name="document_search",
            arguments='{"query":"çatdırılma ödənişi","limit":5}',
        ),
    )
    llm = FakeLlm(
        [
            response(AssistantMessage(tool_calls=[tool_call]), "r1"),
            response(AssistantMessage(content="Çatdırılma pulsuzdur."), "r2"),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=DocumentFakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()
    result = await runtime.run(run=run, session=session, user_message="Çatdırılma ödənişlidirmi?")

    assert result.used_tools == ["document_search"]
    assert result.presentation is None
    assert repository.completed is not None
    assert repository.completed["debug_trace"]["diagnosis"]["code"] == "document_chunks_found"
    document_event = next(
        event
        for event in repository.completed["debug_trace"]["timeline"]
        if event["stage"] == "document_search"
    )
    assert document_event["result"]["returned_chunks"][0]["chunk_id"] == "delivery:0001"


@pytest.mark.asyncio
async def test_product_cards_survive_a_following_document_search(settings: object) -> None:
    repository = FakeRepository()
    product_call = ToolCall(
        id="call_product",
        function=ToolFunctionCall(name="product_search", arguments='{"query":"kondisioner"}'),
    )
    document_call = ToolCall(
        id="call_document",
        function=ToolFunctionCall(name="document_search", arguments='{"query":"quraşdırma"}'),
    )
    llm = FakeLlm(
        [
            response(AssistantMessage(tool_calls=[product_call]), "r1"),
            response(AssistantMessage(tool_calls=[document_call]), "r2"),
            response(AssistantMessage(content="Məhsulu və quraşdırma qaydasını təqdim edirəm."), "r3"),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=DocumentFakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()
    result = await runtime.run(
        run=run,
        session=session,
        user_message="Kondisioner göstər, quraşdırılması pulsuzdur?",
    )

    assert result.used_tools == ["product_search", "document_search"]
    assert result.presentation is not None
    assert result.presentation["recommended_product_id"] == "prd_smartphones_001"
    assert repository.completed is not None
    assert repository.completed["last_product_ids"] == ["prd_smartphones_001"]


@pytest.mark.asyncio
async def test_fourth_model_round_forces_final_answer_without_tools(settings: object) -> None:
    repository = FakeRepository()
    tool_responses = []
    for index in range(1, 4):
        call = ToolCall(
            id=f"call_{index}",
            function=ToolFunctionCall(
                name="product_search",
                arguments='{"query":"iPhone","brand":"Apple"}',
            ),
        )
        tool_responses.append(response(AssistantMessage(tool_calls=[call]), f"r{index}"))
    llm = FakeLlm(
        [*tool_responses, response(AssistantMessage(content="Mövcud nəticələri təqdim edirəm."), "r4")]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()

    result = await runtime.run(run=run, session=session, user_message="iPhone göstər")

    assert result.answer == "Mövcud nəticələri təqdim edirəm."
    assert len(repository.tool_exchanges) == 3
    assert llm.calls[3]["tool_choice"] == "none"
    assert llm.calls[3]["tools"] is None
    assert repository.completed is not None
    assert repository.completed["tool_count"] == 3
    assert repository.completed["model_rounds"] == 4


@pytest.mark.asyncio
async def test_alternative_status_deterministically_overrides_model_wording(settings: object) -> None:
    repository = FakeRepository()
    tool_call = ToolCall(
        id="call_alt",
        function=ToolFunctionCall(
            name="product_search",
            arguments='{"query":"iPhone 19","model":"iPhone 19","category_id":"smartphones"}',
        ),
    )
    llm = FakeLlm(
        [
            response(AssistantMessage(tool_calls=[tool_call]), "r1"),
            response(AssistantMessage(content="iPhone 19 tapdım."), "r2"),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=AlternativeFakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()

    result = await runtime.run(run=run, session=session, user_message="iPhone 19 göstər")

    assert result.answer.startswith("iPhone 19 kataloqda tapılmadı.")
    assert "iPhone 19 tapdım" not in result.answer
    assert result.presentation is not None
    assert result.presentation["result_kind"] == "alternatives"
    assert result.presentation["items"][0]["differences"] == ["Model fərqlidir: iPhone 16"]
    assert repository.completed is not None
    assert repository.completed["debug_trace"]["diagnosis"]["code"] == "alternatives_returned"


@pytest.mark.asyncio
async def test_json_product_answer_is_replaced_with_public_text(settings: object) -> None:
    repository = FakeRepository()
    tool_call = ToolCall(
        id="call_json",
        function=ToolFunctionCall(name="product_search", arguments='{"query":"iPhone 16"}'),
    )
    llm = FakeLlm(
        [
            response(AssistantMessage(tool_calls=[tool_call]), "r1"),
            response(
                AssistantMessage(content='{"product_id":"prd_smartphones_001"}'),
                "r2",
            ),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()

    result = await runtime.run(run=run, session=session, user_message="JSON ver")

    assert not result.answer.lstrip().startswith("{")
    assert "Apple iPhone 16" in result.answer
    assert "product_id" not in result.answer


@pytest.mark.asyncio
async def test_content_filter_returns_http_safe_completed_answer(settings: object) -> None:
    repository = FakeRepository()
    filtered = ChatCompletionResponse(
        id="r1",
        choices=[
            CompletionChoice(
                message=AssistantMessage(content=None),
                finish_reason="content_filter",
            )
        ],
    )
    llm = FakeLlm(
        [
            filtered,
            response(AssistantMessage(content="Təhlükəsiz qısa cavab."), "r2"),
        ]
    )
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=llm,
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()

    result = await runtime.run(run=run, session=session, user_message="Raw JSON ver")

    assert result.answer
    assert len(llm.calls) == 2
    assert repository.failed is None
    assert repository.completed is not None
    warnings = repository.completed["debug_trace"]["warnings"]
    assert warnings[0]["code"] == "degraded_safe_response"
    assert warnings[0]["provider_finish_reason"] == "content_filter"
    retry = next(
        item
        for item in repository.completed["debug_trace"]["timeline"]
        if item["stage"] == "safe_response_retry"
    )
    assert retry["status"] == "completed"


@pytest.mark.asyncio
async def test_failed_run_has_explanation_and_preserves_memory_revision(settings: object) -> None:
    class FailingLlm:
        async def chat(self, **_: Any) -> ChatCompletionResponse:
            raise ProviderError("provider_unavailable", "provider unavailable")

    repository = FakeRepository()
    runtime = AgentRuntime(
        settings=settings,
        repository=repository,
        llm=FailingLlm(),  # type: ignore[arg-type]
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    run, session = make_run_and_session()

    with pytest.raises(AgentRuntimeError):
        await runtime.run(run=run, session=session, user_message="iPhone göstər")

    assert repository.completed is None
    assert repository.failed is not None
    trace = repository.failed["debug_trace"]
    assert trace["decision_explanation"]["basis"] == "runtime_guard"
    assert trace["decision_explanation"]["outcome"]["status"] == "failed"
    assert trace["memory_transition"]["revision_before"] == 0
    assert trace["memory_transition"]["revision_after"] == 0
