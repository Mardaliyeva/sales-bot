from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionCreateRequest(StrictModel):
    pass


class SessionCreateResponse(StrictModel):
    session_id: UUID
    status: Literal["active"]
    expires_at: datetime


class ChatRequest(StrictModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=4000)


class ProductCardItem(StrictModel):
    product_id: str
    name: str
    sku: str
    price: float
    currency: str
    stock_status: Literal["in_stock", "out_of_stock"]
    rating: float
    warranty_months: int
    highlights: list[str]
    differences: list[str] = Field(default_factory=list)
    budget_remaining: float | None = None


class ProductCardsPresentation(StrictModel):
    type: Literal["product_cards"]
    result_kind: Literal["matches", "alternatives", "exact_conflict", "comparison"] = "matches"
    requested_label: str | None = None
    title: str
    total: int
    shown_count: int
    recommended_product_id: str | None = None
    requested_item: ProductCardItem | None = None
    constraint_conflicts: list[str] = Field(default_factory=list)
    relaxed_fields: list[str] = Field(default_factory=list)
    items: list[ProductCardItem]


class ChatResponse(StrictModel):
    request_id: UUID
    session_id: UUID
    message_id: UUID
    answer: str
    used_tools: list[str]
    finish_reason: Literal["completed"]
    presentation: ProductCardsPresentation | None = None


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    request_id: UUID | None = None
    error: ErrorDetail


class DebugTraceResponse(StrictModel):
    trace_version: int
    detail_level: Literal["full", "legacy_partial"]
    request_id: UUID
    session_id: UUID
    message_id: UUID | None
    status: Literal["running", "completed", "failed"]
    model: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    prompt: dict[str, Any] = Field(default_factory=dict)
    diagnosis: dict[str, Any] | None
    data_sources: dict[str, Any]
    timeline: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    metrics: dict[str, Any]
    decision_explanation: dict[str, Any] | None = None
    memory_transition: dict[str, Any] | None = None
    continuation_context_before: str | None = None
    continuation_context_after: str | None = None
