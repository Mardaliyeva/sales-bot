from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.tools.schemas import ProductSearchArguments, ProductSearchItem

AZ_TRANSLATION = str.maketrans(
    {"ə": "e", "ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"}
)


class CatalogLoadError(RuntimeError):
    pass


def normalize_text(value: str) -> str:
    folded = value.casefold().translate(AZ_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).strip()


class ProductCatalog:
    """Validated JSON source of truth used only to hydrate Qdrant result IDs."""

    def __init__(self, products_path: Path) -> None:
        self.products_path = products_path
        self.catalog_dir = products_path.parent
        self.products: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self.ready = False
        self._products_by_id: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        manifest_path = self.catalog_dir / "manifest.json"
        schema_path = self.catalog_dir / "product.schema.json"
        for path in (self.products_path, manifest_path, schema_path):
            if not path.is_file():
                raise CatalogLoadError(f"Kataloq faylı tapılmadı: {path}")

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
                    raise CatalogLoadError(
                        f"Məhsul sxemi xətası, sətir {line_number}: {error.message}"
                    )
                loaded.append(product)

        if len(loaded) != manifest.get("product_count"):
            raise CatalogLoadError("Manifestdəki məhsul sayı products.jsonl ilə uyğun deyil")
        if any(item.get("dataset_version") != manifest.get("dataset_version") for item in loaded):
            raise CatalogLoadError("Dataset versiyaları uyğun deyil")
        products_by_id = {str(product["product_id"]): product for product in loaded}
        if len(products_by_id) != len(loaded):
            raise CatalogLoadError("Kataloqda təkrar product_id var")

        self.products = loaded
        self._products_by_id = products_by_id
        self.manifest = manifest
        self.ready = True

    def hydrate(self, product_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not self.ready:
            raise CatalogLoadError("Kataloq hazır deyil")
        products: list[dict[str, Any]] = []
        for product_id in product_ids:
            product = self._products_by_id.get(product_id)
            if product is None:
                raise CatalogLoadError(f"Qdrant ID-si JSON kataloqda yoxdur: {product_id}")
            products.append(product)
        return products

    @staticmethod
    def applied_filters(args: ProductSearchArguments) -> dict[str, Any]:
        return args.model_dump(
            exclude={"query", "sort", "limit", "required_filter_fields"},
            exclude_none=True,
            mode="json",
        )

    @staticmethod
    def to_result(product: dict[str, Any]) -> ProductSearchItem:
        return ProductSearchItem(
            product_id=product["product_id"],
            sku=product["sku"],
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

    @staticmethod
    def _verify_checksum(path: Path, expected: str | None) -> None:
        if not expected:
            raise CatalogLoadError(f"Checksum manifestdə yoxdur: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise CatalogLoadError(f"Checksum uyğun deyil: {path.name}")
