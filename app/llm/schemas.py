from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolFunctionCall(ProviderModel):
    name: str
    arguments: str


class ToolCall(ProviderModel):
    id: str
    type: str = "function"
    function: ToolFunctionCall


class AssistantMessage(ProviderModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_details: list[dict[str, Any]] | None = None


class CompletionChoice(ProviderModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = None


class CompletionUsage(ProviderModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: dict[str, Any] | None = None

    @property
    def reasoning_tokens(self) -> int:
        if not self.completion_tokens_details:
            return 0
        value = self.completion_tokens_details.get("reasoning_tokens", 0)
        return int(value) if isinstance(value, int | float) else 0


class ChatCompletionResponse(ProviderModel):
    id: str | None = None
    model: str | None = None
    choices: list[CompletionChoice]
    usage: CompletionUsage = CompletionUsage()

    @property
    def first_choice(self) -> CompletionChoice:
        if len(self.choices) != 1:
            raise ValueError("Provider dəqiq bir choice qaytarmalıdır")
        return self.choices[0]
