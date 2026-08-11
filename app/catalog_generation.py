from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.config import PROJECT_ROOT

CATALOG_DIR = PROJECT_ROOT / "data" / "catalog"
PRODUCTS_PATH = CATALOG_DIR / "products.jsonl"
SCHEMA_PATH = CATALOG_DIR / "product.schema.json"
MANIFEST_PATH = CATALOG_DIR / "manifest.json"
RULES_PATH = CATALOG_DIR / "generation_rules.json"
GOLDEN_QUERIES_PATH = CATALOG_DIR / "golden_queries.json"
CHALLENGE_QUERIES_PATH = PROJECT_ROOT / "data" / "evals" / "product_retrieval_challenge.json"
DATASET_VERSION = "1.1.0"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_products(path: Path = PRODUCTS_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _encoded_value(product: dict[str, Any], rule: dict[str, Any]) -> Any | None:
    if product["category"]["id"] != rule["category_id"]:
        return None
    match = re.search(rule["pattern"], str(product.get(rule["source_field"], "")), re.I)
    if match is None:
        return None
    captured = match.group(1).upper()
    if rule["transform"] == "multiply":
        return int(captured) * int(rule["factor"])
    if rule["transform"] == "value_map":
        return rule["values"][captured]
    raise ValueError(f"Naməlum generator transform-u: {rule['transform']}")


def _fact_text(field: str, value: Any) -> str:
    label = field.replace("_", " ")
    if isinstance(value, bool):
        return f"{label}: {'bəli' if value else 'xeyr'}"
    if isinstance(value, list):
        return f"{label}: {', '.join(map(str, value))}"
    return f"{label}: {value}"


def regenerate_record(
    source: dict[str, Any],
    rules: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    product = deepcopy(source)
    product["dataset_version"] = DATASET_VERSION
    for rule in rules:
        encoded = _encoded_value(product, rule)
        if encoded is not None:
            product["attributes"][rule["target_attribute"]] = encoded

    facts = [_fact_text(field, value) for field, value in sorted(product["attributes"].items())]
    identity = (
        f"{product['name']}. {product['category']['name']}. {product['brand']}. "
        f"{product['model_family']}. {product['color']['name']} rəng."
    )
    product["short_description"] = f"{identity} {'; '.join(facts[:6])}."
    product["description"] = (
        f"{identity} Bu sintetik kataloq record-u eyni strukturlaşdırılmış generator "
        f"mənbəyindən hazırlanıb. Xüsusiyyətlər: {'; '.join(facts)}. "
        f"Zəmanət: {product['warranty_months']} ay."
    )
    product["embedding_text"] = " ".join(
        [
            product["name"],
            product["category"]["name"],
            product["brand"],
            product["model"],
            product["model_family"],
            product["color"]["name"],
            *facts,
            *map(str, product.get("tags", [])),
        ]
    )
    filter_payload: dict[str, Any] = {
        "brand": product["brand"],
        "category_id": product["category"]["id"],
        "color_code": product["color"]["code"],
        "currency": product["price"]["currency"],
        "in_stock": product["stock"]["status"] == "in_stock",
        "model_family": product["model_family"],
        "sale_price": product["price"]["sale"],
    }
    indexed_attributes = {
        "storage_gb",
        "ram_gb",
        "btu",
        "screen_size_in",
        "connectivity",
        "active_noise_cancellation",
    }
    filter_payload.update(
        {
            field: value
            for field, value in product["attributes"].items()
            if field in indexed_attributes
        }
    )
    product["filter_payload"] = filter_payload
    return product


def validate_records(
    products: Sequence[dict[str, Any]],
    rules: Sequence[dict[str, Any]],
) -> list[str]:
    schema = _read_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, product in enumerate(products, start=1):
        product_id = str(product.get("product_id", f"line:{index}"))
        if product_id in seen_ids:
            errors.append(f"{product_id}: təkrar product_id")
        seen_ids.add(product_id)
        schema_error = next(validator.iter_errors(product), None)
        if schema_error:
            errors.append(f"{product_id}: {schema_error.message}")
        for rule in rules:
            encoded = _encoded_value(product, rule)
            if encoded is None:
                continue
            actual = product.get("attributes", {}).get(rule["target_attribute"])
            if actual != encoded:
                errors.append(
                    f"{product_id}: {rule['target_attribute']}={actual!r}, model/name={encoded!r}"
                )
    return errors


def regenerate_catalog() -> list[dict[str, Any]]:
    rules_document = _read_json(RULES_PATH)
    rules = rules_document["encoded_attribute_rules"]
    products = [regenerate_record(product, rules) for product in _read_products()]
    errors = validate_records(products, rules)
    if errors:
        raise ValueError("\n".join(errors))

    serialized = "".join(
        json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for product in products
    )
    PRODUCTS_PATH.write_text(serialized, encoding="utf-8", newline="\n")
    _refresh_rule_affected_eval_expectations(products, rules)

    manifest = _read_json(MANIFEST_PATH)
    manifest["dataset_version"] = DATASET_VERSION
    manifest["generator"] = {
        "schema_version": rules_document["schema_version"],
        "rules_file": RULES_PATH.name,
        "record_source": "single_structured_record",
    }
    manifest["product_count"] = len(products)
    manifest["category_counts"] = dict(
        sorted(Counter(product["category"]["id"] for product in products).items())
    )
    manifest["in_stock_count"] = sum(
        product["stock"]["status"] == "in_stock" for product in products
    )
    manifest["discounted_count"] = sum(
        product["price"].get("discount_percent", 0) > 0 for product in products
    )
    manifest["validation"] = {"status": "passed", "error_count": 0}
    manifest["checksums"] = {
        "products_sha256": hashlib.sha256(PRODUCTS_PATH.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        "generation_rules_sha256": hashlib.sha256(RULES_PATH.read_bytes()).hexdigest(),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return products


def _matches_filters(product: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        if field in {"required_filter_fields", "search_intent", "requested_fields"}:
            continue
        if field == "category_id":
            actual = product["category"]["id"]
        elif field == "color_code":
            actual = product["color"]["code"]
        elif field == "in_stock":
            actual = product["stock"]["status"] == "in_stock"
        elif field == "min_price":
            if float(product["price"]["sale"]) < float(expected):
                return False
            continue
        elif field == "max_price":
            if float(product["price"]["sale"]) > float(expected):
                return False
            continue
        elif field in product:
            actual = product[field]
        else:
            actual = product.get("attributes", {}).get(field)
        if actual != expected:
            return False
    return True


def _refresh_rule_affected_eval_expectations(
    products: Sequence[dict[str, Any]],
    rules: Sequence[dict[str, Any]],
) -> None:
    affected = {
        (str(rule["category_id"]), str(rule["target_attribute"])) for rule in rules
    }
    for path in (GOLDEN_QUERIES_PATH, CHALLENGE_QUERIES_PATH):
        cases = json.loads(path.read_text(encoding="utf-8-sig"))
        changed = False
        for case in cases:
            filters = case.get("filters", {})
            category_id = filters.get("category_id")
            affected_fields = {
                field for category, field in affected if category == category_id
            }
            if not affected_fields.intersection(filters) or not case.get("expected_product_ids"):
                continue
            case["expected_product_ids"] = [
                product["product_id"]
                for product in products
                if _matches_filters(product, filters)
            ]
            changed = True
        if changed:
            path.write_text(
                json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )


def check_catalog() -> list[str]:
    rules = _read_json(RULES_PATH)["encoded_attribute_rules"]
    return validate_records(_read_products(), rules)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Sintetik kataloqu deterministik yoxla/regenerate et")
    parser.add_argument("command", choices=("check", "regenerate"))
    args = parser.parse_args(argv)
    try:
        if args.command == "regenerate":
            products = regenerate_catalog()
            print(f"{len(products)} məhsul deterministik regenerate edildi.")
            return 0
        errors = check_catalog()
        if errors:
            print("\n".join(errors))
            return 1
        print("Kataloq schema və ad–atribut uyğunluğu keçdi.")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Kataloq generasiya xətası: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
