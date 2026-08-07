from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.runtime import AgentRuntime
from app.db.models import AgentRun, ChatSession
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
