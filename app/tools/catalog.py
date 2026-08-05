from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.tools.schemas import ProductSearchArguments, ProductSearchItem, ProductSearchResult

AZ_TRANSLATION = str.maketrans({"ə": "e", "ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"})
TOKEN_RE = re.compile(r"[a-z0-9]+")


class CatalogLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogCandidate:
    product: dict[str, Any]
    score: float


def normalize_text(value: str) -> str:
    folded = value.casefold().translate(AZ_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(value)))


class ProductCatalog:
    def __init__(self, products_path: Path) -> None:
        self.products_path = products_path
        self.catalog_dir = products_path.parent
        self.products: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self.ready = False

    def load(self) -> None:
        manifest_path = self.catalog_dir / "manifest.json"
        schema_path = self.catalog_dir / "product.schema.json"
        for path in (self.products_path, manifest_path, schema_path):
            if not path.is_file():
                raise CatalogLoadError(f"Katalog faylı tapılmadı: {path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
        expected = manifest.get("checksums", {})
        self._verify_checksum(self.products_path, expected.get("products_sha256"))
        self._verify_checksum(schema_path, expected.get("schema_sha256"))

        validator = Draft202012Validator(schema)
        loaded: list[dict[str, Any]] = []
        with self.products_path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                error = next(validator.iter_errors(product), None)
                if error is not None:
                    raise CatalogLoadError(f"Məhsul sxemi xətası, sətir {line_number}: {error.message}")
                loaded.append(product)

        if len(loaded) != manifest.get("product_count"):
            raise CatalogLoadError("Manifestdəki məhsul sayı products.jsonl ilə uyğun deyil")
        if any(item.get("dataset_version") != manifest.get("dataset_version") for item in loaded):
            raise CatalogLoadError("Dataset versiyaları uyğun deyil")

        self.products = loaded
        self.manifest = manifest
        self.ready = True

    @staticmethod
    def _verify_checksum(path: Path, expected: str | None) -> None:
        if not expected:
            raise CatalogLoadError(f"Checksum manifestdə yoxdur: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise CatalogLoadError(f"Checksum uyğun deyil: {path.name}")

    def search(self, args: ProductSearchArguments) -> ProductSearchResult:
        if not self.ready:
            raise CatalogLoadError("Katalog hazır deyil")

        filtered = [product for product in self.products if self._matches(product, args)]
        query_tokens = tokens(args.query)
        scored = [(self._score(product, args.query, query_tokens), product) for product in filtered]

        if not self._has_structured_filter(args):
            scored = [(score, product) for score, product in scored if score > 0]

        if args.sort == "price_asc":
            scored.sort(key=lambda pair: (pair[1]["price"]["sale"], pair[1]["product_id"]))
        elif args.sort == "price_desc":
            scored.sort(key=lambda pair: (-pair[1]["price"]["sale"], pair[1]["product_id"]))
        elif args.sort == "rating_desc":
            scored.sort(
                key=lambda pair: (
                    -pair[1]["rating"],
                    pair[1]["price"]["sale"],
                    pair[1]["product_id"],
                )
            )
        else:
            scored.sort(key=lambda pair: (-pair[0], pair[1]["product_id"]))

        items = [self._to_result(product) for _, product in scored[: args.limit]]
        return ProductSearchResult(
            total=len(scored),
            applied_filters=self._applied_filters(args),
            items=items,
        )

    def rank_candidates(
        self,
        args: ProductSearchArguments,
        *,
        limit: int = 20,
    ) -> list[CatalogCandidate]:
        if not self.ready:
            raise CatalogLoadError("Kataloq hazır deyil")
        if limit <= 0:
            raise ValueError("Namizəd limiti müsbət olmalıdır")

        filtered = [product for product in self.products if self._matches(product, args)]
        query_tokens = tokens(args.query)
        scored = [(self._score(product, args.query, query_tokens), product) for product in filtered]
        if not self._has_structured_filter(args):
            scored = [(score, product) for score, product in scored if score > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["product_id"]))
        return [
            CatalogCandidate(product=product, score=score)
            for score, product in scored[:limit]
        ]

    def count_filtered(self, args: ProductSearchArguments) -> int:
        if not self.ready:
            raise CatalogLoadError("Kataloq hazır deyil")
        return sum(1 for product in self.products if self._matches(product, args))

    @staticmethod
    def _has_structured_filter(args: ProductSearchArguments) -> bool:
        values = args.model_dump(exclude={"query", "sort", "limit"})
        return any(value is not None for value in values.values())

    @staticmethod
    def _matches(product: dict[str, Any], args: ProductSearchArguments) -> bool:
        payload = product["filter_payload"]
        exact_fields = {
            "category_id": args.category_id,
            "color_code": args.color_code,
            "in_stock": args.in_stock,
            "storage_gb": args.storage_gb,
            "ram_gb": args.ram_gb,
            "btu": args.btu,
            "screen_size_in": args.screen_size_in,
            "active_noise_cancellation": args.active_noise_cancellation,
        }
        for field, expected in exact_fields.items():
            if expected is not None and payload.get(field) != expected:
                return False

        text_fields = {
            "brand": args.brand,
            "model_family": args.model_family,
            "connectivity": args.connectivity,
        }
        for field, expected in text_fields.items():
            actual_normalized = normalize_text(str(payload.get(field, "")))
            if expected is not None and actual_normalized != normalize_text(expected):
                return False

        price = float(payload["sale_price"])
        if args.min_price is not None and price < args.min_price:
            return False
        return not (args.max_price is not None and price > args.max_price)

    @staticmethod
    def _score(product: dict[str, Any], query: str, query_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        haystack = " ".join(
            [
                product["embedding_text"],
                product["name"],
                product["sku"],
                product["product_id"],
            ]
        )
        haystack_normalized = normalize_text(haystack)
        overlap = len(query_tokens & tokens(haystack)) / len(query_tokens)
        exact_bonus = 1.0 if normalize_text(query) in haystack_normalized else 0.0
        model_bonus = 0.5 if normalize_text(product["model"]) in normalize_text(query) else 0.0
        return overlap + exact_bonus + model_bonus

    @staticmethod
    def _applied_filters(args: ProductSearchArguments) -> dict[str, Any]:
        return args.model_dump(exclude={"query", "sort", "limit"}, exclude_none=True)

    @staticmethod
    def _to_result(product: dict[str, Any]) -> ProductSearchItem:
        return ProductSearchItem(
            product_id=product["product_id"],
            name=product["name"],
            category_id=product["category"]["id"],
            category_name=product["category"]["name"],
            brand=product["brand"],
            model_family=product["model_family"],
            color_code=product["color"]["code"],
            color_name=product["color"]["name"],
            sale_price=product["price"]["sale"],
            currency=product["price"]["currency"],
            stock_status=product["stock"]["status"],
            warranty_months=product["warranty_months"],
            rating=product["rating"],
            attributes=product["attributes"],
            short_description=product["short_description"],
        )
