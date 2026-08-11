from __future__ import annotations

import pytest

from app.agent.presentation import build_product_cards, product_highlights


@pytest.mark.parametrize(
    ("category_id", "attributes", "expected"),
    [
        (
            "smartphones",
            {
                "storage_gb": 128,
                "ram_gb": 8,
                "main_camera_mp": 48,
                "network": "5G",
                "operating_system": "Android",
            },
            ["128 GB yaddaş", "8 GB RAM", "48 MP kamera", "5G", "Android"],
        ),
        (
            "tablets",
            {
                "storage_gb": 256,
                "ram_gb": 8,
                "display_size_in": 11,
                "cellular": False,
                "pen_support": True,
            },
            ["256 GB yaddaş", "8 GB RAM", '11" ekran', "Wi-Fi model", "Qələm dəstəyi"],
        ),
        (
            "laptops",
            {
                "cpu_brand": "Intel",
                "cpu_model": "Core i7",
                "ram_gb": 16,
                "storage_gb": 512,
                "storage_type": "SSD",
                "gpu": "RTX 4060",
                "operating_system": "Windows 11",
            },
            ["Intel Core i7", "16 GB RAM", "512 GB SSD", "RTX 4060", "Windows 11"],
        ),
        (
            "air_conditioners",
            {
                "btu": 12000,
                "coverage_min_m2": 20,
                "coverage_max_m2": 35,
                "inverter": True,
                "energy_class": "A++",
                "wifi": False,
            },
            ["12000 BTU", "20–35 m²", "İnverter", "A++ enerji sinfi", "Wi-Fi yoxdur"],
        ),
        (
            "televisions",
            {
                "screen_size_in": 55,
                "panel_type": "QLED",
                "resolution": "4K UHD",
                "refresh_rate_hz": 120,
                "hdr": True,
                "smart_tv_os": "Tizen",
            },
            ['55" ekran QLED', "4K UHD", "120 Hz", "HDR", "Tizen"],
        ),
        (
            "headphones",
            {
                "form_factor": "Over-ear",
                "connectivity": "Bluetooth 5.3",
                "active_noise_cancellation": True,
                "battery_hours": 30,
                "water_resistance": "IPX4",
            },
            ["Over-ear", "Bluetooth 5.3", "ANC", "30 saat batareya", "IPX4"],
        ),
    ],
)
def test_category_highlights_are_ordered_and_limited(
    category_id: str,
    attributes: dict[str, object],
    expected: list[str],
) -> None:
    assert product_highlights(category_id, attributes) == expected
    assert len(expected) <= 5


def _item(index: int) -> dict[str, object]:
    return {
        "product_id": f"prd_televisions_{index:03d}",
        "sku": f"SYN-TV-SMS-{index:03d}",
        "name": f"Samsung TV {index}",
        "category_id": "televisions",
        "sale_price": 569.99 if index == 1 else 700 + index,
        "currency": "AZN",
        "stock_status": "in_stock",
        "warranty_months": 36,
        "rating": 5.0,
        "attributes": {
            "screen_size_in": 55,
            "panel_type": "QLED",
            "resolution": "4K UHD",
            "refresh_rate_hz": 120,
            "hdr": True,
            "smart_tv_os": "Tizen",
        },
    }


def test_product_cards_limit_items_and_calculate_budget() -> None:
    result = {
        "status": "success",
        "total": 8,
        "applied_filters": {"max_price": 1200},
        "items": [_item(index) for index in range(1, 6)],
    }

    presentation = build_product_cards(result)

    assert presentation is not None
    assert presentation["title"] == "1200 AZN büdcəyə uyğun 8 məhsul tapdım"
    assert presentation["total"] == 8
    assert presentation["shown_count"] == 3
    assert presentation["recommended_product_id"] == "prd_televisions_001"
    assert presentation["result_kind"] == "matches"
    assert presentation["relaxed_fields"] == []
    assert [item["product_id"] for item in presentation["items"]] == [
        "prd_televisions_001",
        "prd_televisions_002",
        "prd_televisions_003",
    ]
    assert presentation["items"][0]["budget_remaining"] == 630.01


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"status": "error", "items": [_item(1)]},
        {"status": "success", "total": 0, "items": []},
    ],
)
def test_product_cards_are_omitted_without_successful_items(result: dict | None) -> None:
    assert build_product_cards(result) is None


def test_alternative_cards_explain_missing_request_and_differences() -> None:
    item = _item(1)
    item["differences"] = ["Rəng fərqlidir: Qara"]
    result = {
        "status": "success",
        "match_status": "alternatives",
        "requested_label": "Samsung Future TV",
        "strict_total": 0,
        "total": 1,
        "applied_filters": {"category_id": "televisions", "color_code": "gold"},
        "relaxed_fields": ["color_code"],
        "items": [item],
    }

    presentation = build_product_cards(result)

    assert presentation is not None
    assert presentation["result_kind"] == "alternatives"
    assert presentation["title"] == "Samsung Future TV tapılmadı — yaxın alternativlər"
    assert presentation["requested_label"] == "Samsung Future TV"
    assert presentation["items"][0]["differences"] == ["Rəng fərqlidir: Qara"]


def test_exact_conflict_presents_requested_product_separately() -> None:
    requested = _item(1)
    alternative = _item(2)
    result = {
        "status": "success",
        "match_status": "exact_conflict",
        "requested_label": "Samsung QN900D",
        "strict_total": 0,
        "total": 1,
        "applied_filters": {"max_price": 500},
        "requested_item": requested,
        "constraint_conflicts": ["Qiymət büdcəni keçir: 1186.79 AZN"],
        "recommended_product_id": alternative["product_id"],
        "display_product_ids": [alternative["product_id"]],
        "items": [alternative],
    }

    presentation = build_product_cards(result)

    assert presentation is not None
    assert presentation["result_kind"] == "exact_conflict"
    assert presentation["requested_item"]["product_id"] == requested["product_id"]
    assert presentation["items"][0]["product_id"] == alternative["product_id"]
    assert presentation["constraint_conflicts"] == ["Qiymət büdcəni keçir: 1186.79 AZN"]
