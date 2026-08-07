from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools.schemas import AttributeFilter, ProductSearchArguments


@pytest.mark.parametrize(
    ("field", "operator", "value"),
    [
        ("battery_mah", "gte", 5000),
        ("cpu_brand", "eq", "Intel"),
        ("operating_system", "in", ["Windows 11", "Linux"]),
        ("wifi", "eq", True),
        ("modes", "contains_any", ["soyutma"]),
        ("indoor_unit_dimensions_mm", "eq", "900x300x220"),
    ],
)
def test_attribute_filter_accepts_operator_for_field_type(
    field: str,
    operator: str,
    value: object,
) -> None:
    parsed = AttributeFilter.model_validate(
        {"field": field, "operator": operator, "value": value}
    )

    assert parsed.field == field


@pytest.mark.parametrize(
    "payload",
    [
        {"field": "battery_mah", "operator": "contains_any", "value": [5000]},
        {"field": "wifi", "operator": "eq", "value": "yes"},
        {"field": "cpu_brand", "operator": "gte", "value": "Intel"},
        {"field": "modes", "operator": "contains_any", "value": []},
        {"field": "unknown", "operator": "eq", "value": "x"},
    ],
)
def test_attribute_filter_rejects_unknown_or_type_incompatible_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AttributeFilter.model_validate(payload)


def test_product_search_rejects_duplicate_attribute_operator() -> None:
    with pytest.raises(ValidationError):
        ProductSearchArguments(
            query="laptop",
            attribute_filters=[
                {"field": "ram_gb", "operator": "gte", "value": 16},
                {"field": "ram_gb", "operator": "gte", "value": 32},
            ],
        )


def test_required_filter_field_must_have_matching_filter_value() -> None:
    with pytest.raises(ValidationError):
        ProductSearchArguments(
            query="yalnız Apple telefon",
            required_filter_fields=["brand"],
        )


def test_required_attribute_field_is_accepted_with_attribute_filter() -> None:
    arguments = ProductSearchArguments(
        query="wifi mütləq olsun",
        attribute_filters=[{"field": "wifi", "operator": "eq", "value": True}],
        required_filter_fields=["wifi"],
    )

    assert arguments.required_filter_fields == ["wifi"]
