from __future__ import annotations

import asyncio
from typing import Protocol

from app.tools.schemas import ProductSearchArguments, ProductSearchResult


class ProductSearchBackend(Protocol):
    def search(self, arguments: ProductSearchArguments) -> ProductSearchResult: ...


class ProductSearchTool:
    name = "product_search"

    def __init__(self, backend: ProductSearchBackend) -> None:
        self.backend = backend

    async def execute(self, arguments: ProductSearchArguments) -> ProductSearchResult:
        return await asyncio.to_thread(self.backend.search, arguments)
