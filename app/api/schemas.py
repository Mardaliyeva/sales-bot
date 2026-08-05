from __future__ import annotations

from datetime import datetime
from typing import Literal
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


class ChatResponse(StrictModel):
    request_id: UUID
    session_id: UUID
    message_id: UUID
    answer: str
    used_tools: list[str]
    finish_reason: Literal["completed"]


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    request_id: UUID | None = None
    error: ErrorDetail
