from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

MAX_PRODUCT_CARDS = 3


def build_product_cards(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result or result.get("status") != "success":
        return None

    match_status = result.get("match_status")
    if match_status == "not_found":
        return None

    raw_items = result.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None

    display_items = [item for item in raw_items[:MAX_PRODUCT_CARDS] if isinstance(item, dict)]
    if not display_items:
        return None

    filters = result.get("applied_filters")
    applied_filters = filters if isinstance(filters, dict) else {}
    max_price = _number(applied_filters.get("max_price"))
    currency = str(display_items[0].get("currency") or "AZN")
    total = _positive_int(result.get("total"), fallback=len(raw_items))

    items: list[dict[str, Any]] = []
    for item in display_items:
        price = _number(item.get("sale_price"))
        card = {
            "product_id": str(item.get("product_id") or ""),
            "name": str(item.get("name") or ""),
            "sku": str(item.get("sku") or ""),
            "price": price or 0.0,
            "currency": str(item.get("currency") or currency),
            "stock_status": str(item.get("stock_status") or "out_of_stock"),
            "rating": _number(item.get("rating")) or 0.0,
            "warranty_months": _positive_int(item.get("warranty_months"), fallback=0),
            "highlights": product_highlights(
                str(item.get("category_id") or ""),
                item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
            ),
            "differences": [
                str(difference)
                for difference in item.get("differences", [])
                if isinstance(difference, str) and difference.strip()
            ],
        }
        if max_price is not None and price is not None and max_price >= price:
            card["budget_remaining"] = round(max_price - price, 2)
        items.append(card)

    if not items[0]["product_id"]:
        return None

    result_kind = "alternatives" if match_status == "alternatives" else "matches"
    requested_label = str(result.get("requested_label") or "").strip() or None
    if result_kind == "alternatives":
        title = (
            f"{requested_label} tapılmadı — yaxın alternativlər"
            if requested_label
            else "Dəqiq uyğunluq tapılmadı — yaxın alternativlər"
        )
    elif max_price is not None:
        title = f"{_display_number(max_price)} {currency} büdcəyə uyğun {total} məhsul tapdım"
    else:
        title = f"{total} uyğun məhsul tapdım"

    return {
        "type": "product_cards",
        "result_kind": result_kind,
        "requested_label": requested_label,
        "title": title,
        "total": total,
        "shown_count": len(items),
        "recommended_product_id": items[0]["product_id"],
        "relaxed_fields": [
            str(field)
            for field in result.get("relaxed_fields", [])
            if isinstance(field, str) and field.strip()
        ],
        "items": items,
    }


def product_highlights(category_id: str, attributes: dict[str, Any]) -> list[str]:
    builders = {
        "smartphones": _smartphone_highlights,
        "tablets": _tablet_highlights,
        "laptops": _laptop_highlights,
        "air_conditioners": _air_conditioner_highlights,
        "televisions": _television_highlights,
        "headphones": _headphone_highlights,
    }
    builder = builders.get(category_id)
    return builder(attributes)[:5] if builder else []


def _smartphone_highlights(attributes: dict[str, Any]) -> list[str]:
    return _present(
        _unit(attributes, "storage_gb", "GB yaddaş"),
        _unit(attributes, "ram_gb", "GB RAM"),
        _unit(attributes, "main_camera_mp", "MP kamera"),
        _text(attributes, "network"),
        _text(attributes, "operating_system"),
    )


def _tablet_highlights(attributes: dict[str, Any]) -> list[str]:
    return _present(
        _unit(attributes, "storage_gb", "GB yaddaş"),
        _unit(attributes, "ram_gb", "GB RAM"),
        _screen(attributes),
        _boolean_label(attributes, "cellular", "Mobil şəbəkə", "Wi-Fi model"),
        _boolean_label(attributes, "pen_support", "Qələm dəstəyi", "Qələm dəstəyi yoxdur"),
    )


def _laptop_highlights(attributes: dict[str, Any]) -> list[str]:
    cpu = " ".join(
        value
        for value in (_text(attributes, "cpu_brand"), _text(attributes, "cpu_model"))
        if value
    )
    storage = " ".join(
        value
        for value in (_raw_unit(attributes, "storage_gb", "GB"), _text(attributes, "storage_type"))
        if value
    )
    return _present(
        cpu,
        _unit(attributes, "ram_gb", "GB RAM"),
        storage,
        _text(attributes, "gpu"),
        _text(attributes, "operating_system"),
    )


def _air_conditioner_highlights(attributes: dict[str, Any]) -> list[str]:
    coverage_min = _number(attributes.get("coverage_min_m2"))
    coverage_max = _number(attributes.get("coverage_max_m2"))
    coverage = None
    if coverage_min is not None and coverage_max is not None:
        coverage = f"{_display_number(coverage_min)}–{_display_number(coverage_max)} m²"
    return _present(
        _unit(attributes, "btu", "BTU"),
        coverage,
        _boolean_label(attributes, "inverter", "İnverter", "İnvertersiz"),
        _suffix(attributes, "energy_class", " enerji sinfi"),
        _boolean_label(attributes, "wifi", "Wi-Fi", "Wi-Fi yoxdur"),
    )


def _television_highlights(attributes: dict[str, Any]) -> list[str]:
    screen_panel = " ".join(
        value for value in (_screen(attributes), _text(attributes, "panel_type")) if value
    )
    return _present(
        screen_panel,
        _text(attributes, "resolution"),
        _unit(attributes, "refresh_rate_hz", "Hz"),
        _boolean_label(attributes, "hdr", "HDR", "HDR yoxdur"),
        _text(attributes, "smart_tv_os"),
    )


def _headphone_highlights(attributes: dict[str, Any]) -> list[str]:
    return _present(
        _text(attributes, "form_factor"),
        _text(attributes, "connectivity"),
        _boolean_label(
            attributes,
            "active_noise_cancellation",
            "ANC",
            "ANC yoxdur",
        ),
        _unit(attributes, "battery_hours", "saat batareya"),
        _text(attributes, "water_resistance"),
    )


def _present(*values: str | None) -> list[str]:
    return [value for value in values if value]


def _text(attributes: dict[str, Any], key: str) -> str | None:
    value = attributes.get(key)
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _unit(attributes: dict[str, Any], key: str, unit: str) -> str | None:
    value = _number(attributes.get(key))
    return f"{_display_number(value)} {unit}" if value is not None else None


def _raw_unit(attributes: dict[str, Any], key: str, unit: str) -> str | None:
    return _unit(attributes, key, unit)


def _screen(attributes: dict[str, Any]) -> str | None:
    value = _number(attributes.get("display_size_in"))
    if value is None:
        value = _number(attributes.get("screen_size_in"))
    return f'{_display_number(value)}" ekran' if value is not None else None


def _suffix(attributes: dict[str, Any], key: str, suffix: str) -> str | None:
    value = _text(attributes, key)
    return f"{value}{suffix}" if value else None


def _boolean_label(
    attributes: dict[str, Any],
    key: str,
    true_label: str,
    false_label: str,
) -> str | None:
    value = attributes.get(key)
    if not isinstance(value, bool):
        return None
    return true_label if value else false_label


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return fallback
    return converted if converted >= 0 else fallback


def _display_number(value: float) -> str:
    try:
        decimal = Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return str(value)
    if decimal == decimal.to_integral():
        return str(int(decimal))
    return format(decimal.normalize(), "f").replace(".", ",")
