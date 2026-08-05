from __future__ import annotations

from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments, ProductSearchResult


class ProductSearchTool:
    name = "product_search"

    def __init__(self, catalog: ProductCatalog) -> None:
        self.catalog = catalog

    async def execute(self, arguments: ProductSearchArguments) -> ProductSearchResult:
        return self.catalog.search(arguments)
