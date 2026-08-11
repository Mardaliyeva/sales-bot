from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from app.tools.schemas import DocumentSearchArguments, DocumentSearchResult


class DocumentSearchBackend(Protocol):
    def search_with_trace(self, arguments: DocumentSearchArguments) -> Any: ...


class DocumentSearchBackendError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        debug_trace: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.debug_trace = debug_trace


@dataclass(frozen=True)
class DocumentSearchExecution:
    result: DocumentSearchResult
    debug_trace: dict[str, Any] | None


class DocumentSearchTool:
    name = "document_search"

    def __init__(self, backend: DocumentSearchBackend) -> None:
        self.backend = backend

    async def execute_with_trace(
        self,
        arguments: DocumentSearchArguments,
    ) -> DocumentSearchExecution:
        execution = await asyncio.to_thread(self.backend.search_with_trace, arguments)
        return DocumentSearchExecution(
            result=execution.result,
            debug_trace=execution.debug_trace,
        )

    def debug_source_state(self) -> dict[str, Any]:
        source_state = getattr(self.backend, "debug_source_state", None)
        return source_state() if callable(source_state) else {}

