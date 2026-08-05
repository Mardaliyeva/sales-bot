from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(slots=True)
class AgentResult:
    message_id: uuid.UUID
    answer: str
    used_tools: list[str]


class AgentRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, http_status: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.user_message = message
        self.http_status = http_status
