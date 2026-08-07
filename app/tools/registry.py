from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.tools.product_search import ProductSearchBackendError, ProductSearchTool
from app.tools.schemas import ProductSearchArguments


class UnknownToolError(ValueError):
    pass


class ToolArgumentsError(ValueError):
    pass


@dataclass(frozen=True)
class ToolExecution:
    result: dict[str, Any]
    debug_trace: dict[str, Any] | None = None


class ToolRegistry:
    def __init__(self, product_search: ProductSearchTool, timeout_seconds: float) -> None:
        self.product_search = product_search
        self.timeout_seconds = timeout_seconds

    def specs(self) -> list[dict[str, Any]]:
        schema = ProductSearchArguments.model_json_schema()
        return [
            {
                "type": "function",
                "function": {
                    "name": "product_search",
                    "description": (
                        "Cari sintetik məhsul kataloqunda Qdrant exact/semantic axtarışı edir. "
                        "SKU, product_id və dəqiq model üçün uyğun identifier field-ini doldur. "
                        "İstifadəçinin 'mütləq' və ya 'yalnız' dediyi filter field-lərini "
                        "required_filter_fields daxilində də göndər. Nəticədəki match_status exact, "
                        "normal uyğunluq, alternativ və tapılmama hallarını bir-birindən ayırır. "
                        "Qiymət, stok və ümumi filterləri ayrıca field-lərdə, kateqoriyaya məxsus "
                        "texniki xüsusiyyətləri isə attribute_filters daxilində göndər."
                    ),
                    "parameters": schema,
                },
            }
        ]

    async def execute(self, name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        execution = await self.execute_with_trace(name, raw_arguments)
        return execution.result

    async def execute_with_trace(
        self,
        name: str,
        raw_arguments: dict[str, Any],
    ) -> ToolExecution:
        if name != self.product_search.name:
            raise UnknownToolError(f"Naməlum tool: {name}")
        try:
            arguments = ProductSearchArguments.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolArgumentsError("product_search arqumentləri etibarsızdır") from exc
        try:
            execution = await asyncio.wait_for(
                self.product_search.execute_with_trace(arguments),
                timeout=self.timeout_seconds,
            )
        except ProductSearchBackendError as exc:
            return ToolExecution(
                result={"status": "error", "code": exc.code, "message": exc.message},
                debug_trace=exc.debug_trace,
            )
        return ToolExecution(
            result=execution.result.model_dump(mode="json"),
            debug_trace=execution.debug_trace,
        )

    def debug_source_state(self) -> dict[str, Any]:
        return self.product_search.debug_source_state()
