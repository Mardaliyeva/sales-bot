from __future__ import annotations

import json

import httpx
import pytest

from app.llm.azure_client import AzureChatClient, ProviderError


@pytest.mark.asyncio
async def test_chat_sends_azure_v1_tool_configuration(settings: object) -> None:
    captured: dict[str, object] = {}
    captured_url = ""
    captured_api_key = ""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url, captured_api_key
        captured.update(json.loads(request.content))
        captured_url = str(request.url)
        captured_api_key = request.headers["api-key"]
        return httpx.Response(
            200,
            json={
                "id": "response_1",
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Salam"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    client = AzureChatClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    try:
        response = await client.chat(
            messages=[{"role": "user", "content": "Salam"}],
            tools=[{"type": "function", "function": {"name": "product_search", "parameters": {}}}],
            tool_choice="auto",
            request_id="req_1",
            model_round=1,
        )
    finally:
        await client.close()

    assert response.first_choice.message.content == "Salam"
    assert captured_url == "https://test-resource.openai.azure.com/openai/v1/chat/completions"
    assert captured_api_key == "test-azure-key"
    assert captured["parallel_tool_calls"] is False
    assert captured["max_completion_tokens"] == 800
    assert captured["model"] == "gpt-5.4-mini"
    assert "reasoning_effort" not in captured
    assert "reasoning" not in captured
    assert "max_tokens" not in captured


@pytest.mark.asyncio
async def test_no_tool_round_uses_reasoning_without_tool_configuration(settings: object) -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "response_final",
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hazırdır"}}],
            },
        )

    client = AzureChatClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    try:
        await client.chat(
            messages=[{"role": "user", "content": "Salam"}],
            tools=None,
            tool_choice="none",
            request_id="req_final",
            model_round=4,
        )
    finally:
        await client.close()

    assert captured["reasoning_effort"] == "low"
    assert "tools" not in captured
    assert "tool_choice" not in captured
    assert "parallel_tool_calls" not in captured


@pytest.mark.asyncio
async def test_auth_error_is_not_retried(settings: object) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "secret provider detail"}})

    client = AzureChatClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    try:
        with pytest.raises(ProviderError) as captured:
            await client.chat(messages=[], tools=None, tool_choice="none", request_id="req_1", model_round=1)
    finally:
        await client.close()
    assert calls == 1
    assert captured.value.error_type == "provider_auth_error"
    assert "secret provider detail" not in str(captured.value)


@pytest.mark.asyncio
async def test_retryable_error_is_retried(settings: object) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "provider-secret-detail"})
        return httpx.Response(
            200,
            json={
                "id": "response_2",
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hazırdır"}}],
            },
        )

    client = AzureChatClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    try:
        response = await client.chat(
            messages=[],
            tools=None,
            tool_choice="none",
            request_id="req_2",
            model_round=1,
        )
    finally:
        await client.close()

    assert calls == 2
    assert response.first_choice.message.content == "Hazırdır"
