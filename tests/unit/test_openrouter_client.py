from __future__ import annotations

import json

import httpx
import pytest

from app.llm.openrouter_client import OpenRouterClient, ProviderError


@pytest.mark.asyncio
async def test_chat_sends_expected_tool_configuration(settings: object) -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "response_1",
                "model": "openai/gpt-5.4-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Salam"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
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
    assert captured["parallel_tool_calls"] is False
    assert captured["reasoning"] == {"effort": "low", "exclude": False}
    assert captured["model"] == "openai/gpt-5.4-mini"


@pytest.mark.asyncio
async def test_auth_error_is_not_retried(settings: object) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "secret provider detail"}})

    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    try:
        with pytest.raises(ProviderError) as captured:
            await client.chat(messages=[], tools=None, tool_choice="none", request_id="req_1", model_round=1)
    finally:
        await client.close()
    assert calls == 1
    assert captured.value.error_type == "provider_auth_or_credit_error"
    assert "secret provider detail" not in str(captured.value)
