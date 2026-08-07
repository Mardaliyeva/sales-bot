from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from app.tools.schemas import ProductSearchArguments, ProductSearchResult


class ProductSearchBackend(Protocol):
    def search(self, arguments: ProductSearchArguments) -> ProductSearchResult: ...


class ProductSearchBackendError(RuntimeError):
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
class ProductSearchExecution:
    result: ProductSearchResult
    debug_trace: dict[str, Any] | None


class ProductSearchTool:
    name = "product_search"

    def __init__(self, backend: ProductSearchBackend) -> None:
        self.backend = backend

    async def execute(self, arguments: ProductSearchArguments) -> ProductSearchResult:
        execution = await self.execute_with_trace(arguments)
        return execution.result

    async def execute_with_trace(self, arguments: ProductSearchArguments) -> ProductSearchExecution:
        return await asyncio.to_thread(self._execute_sync, arguments)

    def debug_source_state(self) -> dict[str, Any]:
        source_state = getattr(self.backend, "debug_source_state", None)
        return source_state() if callable(source_state) else {}

    def _execute_sync(self, arguments: ProductSearchArguments) -> ProductSearchExecution:
        search_with_trace = getattr(self.backend, "search_with_trace", None)
        if callable(search_with_trace):
            execution = search_with_trace(arguments)
            return ProductSearchExecution(
                result=execution.result,
                debug_trace=execution.debug_trace,
            )
        return ProductSearchExecution(result=self.backend.search(arguments), debug_trace=None)
