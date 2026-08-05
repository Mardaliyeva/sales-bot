from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from app.tools.product_search import ProductSearchTool
from app.tools.schemas import ProductSearchArguments


class UnknownToolError(ValueError):
    pass


class ToolArgumentsError(ValueError):
    pass


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
                        "Cari sintetik məhsul kataloqunda istifadəçi tələblərinə uyğun məhsulları axtarır. "
                        "Qiymət, stok və xüsusiyyət faktları üçün bu alətdən istifadə et."
                    ),
                    "parameters": schema,
                },
            }
        ]

    async def execute(self, name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        if name != self.product_search.name:
            raise UnknownToolError(f"Naməlum tool: {name}")
        try:
            arguments = ProductSearchArguments.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolArgumentsError("product_search arqumentləri etibarsızdır") from exc
        result = await asyncio.wait_for(
            self.product_search.execute(arguments),
            timeout=self.timeout_seconds,
        )
        return result.model_dump(mode="json")
