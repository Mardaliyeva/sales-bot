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
AttributeOperator = Literal["eq", "gte", "lte", "in", "contains_any"]
AttributeField = Literal[
    "active_noise_cancellation",
    "battery_hours",
    "battery_mah",
    "btu",
    "cellular",
    "charging_type",
    "connectivity",
    "coverage_max_m2",
    "coverage_min_m2",
    "cpu_brand",
    "cpu_model",
    "display_size_in",
    "energy_class",
    "form_factor",
    "gpu",
    "hdmi_count",
    "hdr",
    "indoor_unit_dimensions_mm",
    "inverter",
    "main_camera_mp",
    "microphone",
    "modes",
    "network",
    "noise_level_db",
    "operating_system",
    "outdoor_unit_dimensions_mm",
    "panel_type",
    "pen_support",
    "ram_gb",
    "refresh_rate_hz",
    "refrigerant",
    "resolution",
    "screen_size_in",
    "sim_type",
    "smart_tv_os",
    "storage_gb",
    "storage_type",
    "system_type",
    "water_resistance",
    "weight_kg",
    "wifi",
]
TopLevelPreferenceField = Literal[
    "brand",
    "model_family",
    "color_code",
    "min_price",
    "storage_gb",
    "ram_gb",
    "btu",
    "screen_size_in",
    "connectivity",
    "active_noise_cancellation",
]
RequiredFilterField = TopLevelPreferenceField | AttributeField
MatchStatus = Literal["exact_match", "matching_products", "alternatives", "not_found"]

NUMERIC_ATTRIBUTE_FIELDS = frozenset(
    {
        "battery_hours",
        "battery_mah",
        "btu",
        "coverage_max_m2",
        "coverage_min_m2",
        "display_size_in",
        "hdmi_count",
        "main_camera_mp",
        "noise_level_db",
        "ram_gb",
        "refresh_rate_hz",
        "screen_size_in",
        "storage_gb",
        "weight_kg",
    }
)
BOOLEAN_ATTRIBUTE_FIELDS = frozenset(
    {
        "active_noise_cancellation",
        "cellular",
        "hdr",
        "inverter",
        "microphone",
        "pen_support",
        "wifi",
    }
)
TEXT_ATTRIBUTE_FIELDS = frozenset(
    {
        "charging_type",
        "connectivity",
        "cpu_brand",
        "cpu_model",
        "energy_class",
        "form_factor",
        "gpu",
        "network",
        "operating_system",
        "panel_type",
        "refrigerant",
        "resolution",
        "sim_type",
        "smart_tv_os",
        "storage_type",
        "system_type",
        "water_resistance",
    }
)
LIST_ATTRIBUTE_FIELDS = frozenset({"modes"})
DIMENSION_ATTRIBUTE_FIELDS = frozenset(
    {"indoor_unit_dimensions_mm", "outdoor_unit_dimensions_mm"}
)


class AttributeFilter(BaseModel):
    """Validated filter for a category-specific Qdrant payload field."""

    model_config = ConfigDict(extra="forbid")

    field: AttributeField = Field(description="Kataloq attributes daxilindəki dəqiq payload field-i")
    operator: AttributeOperator = Field(
        description="Rəqəm üçün eq/gte/lte, mətn üçün eq/in, boolean üçün eq, siyahı üçün contains_any"
    )
    value: str | int | float | bool | list[str] | list[int] = Field(
        description="Field tipinə və operatora uyğun filter dəyəri"
    )

    @model_validator(mode="after")
    def validate_field_operator_and_value(self) -> AttributeFilter:
        field = self.field
        operator = self.operator
        value = self.value
        if field in NUMERIC_ATTRIBUTE_FIELDS:
            if operator not in {"eq", "gte", "lte"}:
                raise ValueError(f"{field} üçün operator eq, gte və ya lte olmalıdır")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} üçün rəqəm dəyəri tələb olunur")
        elif field in BOOLEAN_ATTRIBUTE_FIELDS:
            if operator != "eq" or not isinstance(value, bool):
                raise ValueError(f"{field} üçün eq və boolean dəyəri tələb olunur")
        elif field in TEXT_ATTRIBUTE_FIELDS:
            if operator == "eq" and isinstance(value, str):
                return self
            if operator == "in" and isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                return self
            raise ValueError(f"{field} üçün eq mətn və ya in mətn siyahısı tələb olunur")
        elif field in LIST_ATTRIBUTE_FIELDS:
            if operator != "contains_any" or not isinstance(value, list) or not value:
                raise ValueError(f"{field} üçün contains_any və boş olmayan siyahı tələb olunur")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field} siyahısında yalnız mətn ola bilər")
        elif field in DIMENSION_ATTRIBUTE_FIELDS:
            if operator != "eq" or not isinstance(value, str):
                raise ValueError(f"{field} üçün eq və 'en x hündürlük x dərinlik' mətni tələb olunur")
        return self


class ProductSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    product_id: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçi tam product_id veribsə həmin dəyər",
    )
    sku: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçi tam SKU veribsə həmin dəyər",
    )
    model: str | None = Field(
        default=None,
        max_length=120,
        description="İstifadəçinin dəqiq dediyi model; brand adını modelə əlavə etmə",
    )
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
    attribute_filters: list[AttributeFilter] = Field(
        default_factory=list,
        max_length=20,
        description="Kateqoriyaya məxsus texniki parametrlər üçün type-safe Qdrant filterləri",
    )
    required_filter_fields: list[RequiredFilterField] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "İstifadəçinin 'mütləq' və ya 'yalnız' dediyi filter field-ləri. "
            "Kateqoriya, maksimum büdcə və stok onsuz da həmişə sərtdir."
        ),
    )
    sort: SortOption = "relevance"
    limit: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_filters(self) -> ProductSearchArguments:
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError("min_price max_price-dan böyük ola bilməz")
        seen = {(item.field, item.operator) for item in self.attribute_filters}
        if len(seen) != len(self.attribute_filters):
            raise ValueError("Eyni attribute field və operator təkrar verilə bilməz")
        if len(self.required_filter_fields) != len(set(self.required_filter_fields)):
            raise ValueError("required_filter_fields təkrarlanmamalıdır")
        provided_fields = {
            field
            for field in (
                "brand",
                "model_family",
                "color_code",
                "min_price",
                "storage_gb",
                "ram_gb",
                "btu",
                "screen_size_in",
                "connectivity",
                "active_noise_cancellation",
            )
            if getattr(self, field) is not None
        }
        provided_fields.update(item.field for item in self.attribute_filters)
        missing_required = sorted(set(self.required_filter_fields) - provided_fields)
        if missing_required:
            raise ValueError(
                "required_filter_fields dəyəri uyğun filter ilə birlikdə verilməlidir: "
                + ", ".join(missing_required)
            )
        return self

    @property
    def has_exact_identifier(self) -> bool:
        return any((self.product_id, self.sku, self.model))


class ProductSearchItem(BaseModel):
    product_id: str
    sku: str
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
    differences: list[str] = Field(default_factory=list)


class ProductSearchResult(BaseModel):
    status: Literal["success"] = "success"
    match_status: MatchStatus
    requested_label: str | None = None
    strict_total: int = Field(ge=0)
    total: int
    applied_filters: dict[str, Any]
    relaxed_fields: list[str] = Field(default_factory=list)
    items: list[ProductSearchItem]
