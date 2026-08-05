from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.llm.schemas import ChatCompletionResponse

logger = logging.getLogger(__name__)
RETRYABLE_STATUSES = {429, 502, 503, 504}


class ProviderError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    pass


class OpenRouterClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            transport=transport,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Title": "sales-bot",
            },
        )

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        request_id: str,
        model_round: int,
    ) -> ChatCompletionResponse:
        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "reasoning": {"effort": self.settings.reasoning_effort, "exclude": False},
            "max_tokens": self.settings.max_output_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        for attempt in range(2):
            logger.info(
                "llm.request_started",
                extra={
                    "request_id": request_id,
                    "model": self.settings.openrouter_model,
                    "model_round": model_round,
                    "attempt": attempt + 1,
                },
            )
            try:
                response = await self.client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    continue
                raise ProviderTimeoutError("provider_timeout", "OpenRouter sorğusu vaxtı keçdi") from exc
            except httpx.RequestError as exc:
                if attempt == 0:
                    continue
                raise ProviderError("provider_network_error", "OpenRouter şəbəkə xətası") from exc

            if response.status_code in RETRYABLE_STATUSES and attempt == 0:
                await self._wait_before_retry(response)
                continue
            if response.is_error:
                raise self._map_http_error(response.status_code)

            try:
                parsed = ChatCompletionResponse.model_validate(response.json())
                _ = parsed.first_choice
            except (ValueError, ValidationError) as exc:
                raise ProviderError("provider_protocol_error", "OpenRouter cavabı etibarsızdır") from exc
            logger.info(
                "llm.request_completed",
                extra={
                    "request_id": request_id,
                    "model": parsed.model or self.settings.openrouter_model,
                    "model_round": model_round,
                    "status": "success",
                },
            )
            return parsed

        raise ProviderError("provider_unavailable", "OpenRouter müvəqqəti əlçatan deyil")

    async def close(self) -> None:
        await self.client.aclose()

    async def _wait_before_retry(self, response: httpx.Response) -> None:
        raw = response.headers.get("Retry-After")
        try:
            delay = float(raw) if raw is not None else 0.25
        except ValueError:
            delay = 0.25
        await asyncio.sleep(min(max(delay, 0.0), self.settings.llm_timeout_seconds))

    @staticmethod
    def _map_http_error(status_code: int) -> ProviderError:
        if status_code in {401, 402}:
            error_type = "provider_auth_or_credit_error"
        elif status_code == 400:
            error_type = "provider_bad_request"
        elif status_code == 429:
            error_type = "provider_rate_limit"
        elif status_code >= 500:
            error_type = "provider_unavailable"
        else:
            error_type = "provider_http_error"
        return ProviderError(
            error_type,
            f"OpenRouter request statusu: {status_code}",
            status_code=status_code,
        )
