from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CategoryId = Literal[
    "smartphones",
    "tablets",
    "laptops",
    "air_conditioners",
    "televisions",
    "headphones",
]
ColorCode = Literal["black", "white", "blue", "green", "gold", "gray", "silver", "pink"]
SortOption = Literal["relevance", "price_asc", "price_desc", "rating_desc"]


class ProductSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    category_id: CategoryId | None = None
    brand: str | None = Field(default=None, max_length=80)
    model_family: str | None = Field(default=None, max_length=80)
    color_code: ColorCode | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    in_stock: bool | None = None
    storage_gb: int | None = Field(default=None, gt=0)
    ram_gb: int | None = Field(default=None, gt=0)
    btu: int | None = Field(default=None, gt=0)
    screen_size_in: float | None = Field(default=None, gt=0)
    connectivity: str | None = Field(default=None, max_length=80)
    active_noise_cancellation: bool | None = None
    sort: SortOption = "relevance"
    limit: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_price_range(self) -> ProductSearchArguments:
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError("min_price max_price-dan böyük ola bilməz")
        return self


class ProductSearchItem(BaseModel):
    product_id: str
    name: str
    category_id: str
    category_name: str
    brand: str
    model_family: str
    color_code: str
    color_name: str
    sale_price: float
    currency: str
    stock_status: str
    warranty_months: int
    rating: float
    attributes: dict[str, Any]
    short_description: str


class ProductSearchResult(BaseModel):
    status: Literal["success"] = "success"
    total: int
    applied_filters: dict[str, Any]
    items: list[ProductSearchItem]
