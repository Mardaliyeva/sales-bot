from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from app.tools.schemas import (
    TEXT_ATTRIBUTE_FIELDS,
    ProductEntity,
    ProductSearchArguments,
    ProductSearchItem,
)

AZ_TRANSLATION = str.maketrans(
    {"ə": "e", "ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"}
)


class CatalogLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalSearchArguments:
    arguments: ProductSearchArguments
    corrections: list[dict[str, Any]]
    facet_mapping: list[dict[str, Any]]


@dataclass(frozen=True)
class EntityCandidate:
    product_id: str
    score: float
    resolution: str


@dataclass(frozen=True)
class EntityResolution:
    entity_id: str
    raw_text: str
    status: str
    product_id: str | None
    candidates: tuple[EntityCandidate, ...]
    reason: str | None = None
    constraint_field: str | None = None
    constraint_value: str | None = None


@dataclass(frozen=True)
class FacetResolution:
    field: str
    raw_value: str
    status: str
    canonical_value: str | None
    candidates: tuple[str, ...] = ()


class CatalogBackend(Protocol):
    """Data-source-neutral contract for semantic entity resolution."""

    manifest: dict[str, Any]

    def hydrate(self, product_ids: Sequence[str]) -> list[dict[str, Any]]: ...

    def resolve_entity(self, entity: ProductEntity) -> EntityResolution: ...

    def supports_field(self, field: str) -> bool: ...

    def product_by_id(self, product_id: str) -> dict[str, Any] | None: ...

    def resolve_predicate_value(self, field: str, value: str) -> FacetResolution: ...

    def facet_values(self, field: str) -> tuple[str, ...]: ...

    def resolve_facet_namespaces(self, value: str) -> tuple[FacetResolution, ...]: ...


def normalize_text(value: str) -> str:
    folded = value.casefold().translate(AZ_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).strip()


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        mismatches = [
            index
            for index, pair in enumerate(zip(left, right, strict=True))
            if pair[0] != pair[1]
        ]
        if len(mismatches) == 2:
            first, second = mismatches
            if second == first + 1 and left[first] == right[second] and left[second] == right[first]:
                return True
    if len(left) > len(right):
        left, right = right, left
    short_index = long_index = differences = 0
    while short_index < len(left) and long_index < len(right):
        if left[short_index] == right[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        if len(left) == len(right):
            short_index += 1
        long_index += 1
    return True


class ProductCatalog:
    """Validated JSON source of truth used only to hydrate Qdrant result IDs."""

    def __init__(self, products_path: Path) -> None:
        self.products_path = products_path
        self.catalog_dir = products_path.parent
        self.products: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self.ready = False
        self._products_by_id: dict[str, dict[str, Any]] = {}
        self._facets: dict[str, dict[str, set[str]]] = {}
        self._global_facets: dict[str, set[str]] = {}
        self._identifier_indexes: dict[str, dict[str, set[str]]] = {}
        self._entity_token_index: dict[str, set[str]] = {}
        self._token_deletion_index: dict[str, set[str]] = {}
        self._token_signature_index: dict[str, set[str]] = {}
        self._model_tokens_by_id: dict[str, tuple[str, ...]] = {}
        self._catalog_fields: set[str] = set()

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
        self._facets = self._build_facet_registry(loaded)
        self._global_facets = self._build_global_facets(self._facets)
        self._build_resolution_indexes(loaded)
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

    def canonicalize_search_arguments(
        self,
        args: ProductSearchArguments,
    ) -> CanonicalSearchArguments:
        """Resolve model/filter text only from loaded catalog facets.

        No product, brand or model aliases live in code. A value is changed only
        when the catalog provides one unique normalized exact/prefix match.
        """

        products = [
            product
            for product in self.products
            if args.category_id is None or product["category"]["id"] == args.category_id
        ]
        if args.brand:
            canonical_brand = self._resolve_unique(args.brand, {p["brand"] for p in products})
            if canonical_brand:
                products = [p for p in products if p["brand"] == canonical_brand]

        corrections: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        updates: dict[str, Any] = {}
        required_fields = set(args.required_filter_fields)

        def map_value(field: str, value: str | None, candidates: set[str]) -> str | None:
            if not value:
                return value
            canonical = self._resolve_unique(value, candidates)
            mappings.append(
                {
                    "field": field,
                    "original": value,
                    "canonical": canonical,
                    "matched": canonical is not None,
                }
            )
            if canonical and canonical != value:
                corrections.append(
                    {
                        "field": field,
                        "action": "canonicalized",
                        "original": value,
                        "canonical": canonical,
                    }
                )
            return canonical or value

        def map_preference(field: str, value: str | None, candidates: set[str]) -> str | None:
            mapped = map_value(field, value, candidates)
            if value and self._resolve_unique(value, candidates) is None and field not in required_fields:
                corrections.append(
                    {
                        "field": field,
                        "action": "removed_unmapped_preference_filter",
                        "original": value,
                        "reason": "kept_in_semantic_query",
                    }
                )
                return None
            return mapped

        updates["brand"] = map_preference("brand", args.brand, {p["brand"] for p in products})
        updates["model_family"] = map_preference(
            "model_family",
            args.model_family,
            {p["model_family"] for p in products},
        )
        updates["model"] = map_value("model", args.model, {p["model"] for p in products})

        attribute_updates = []
        for attribute_filter in args.attribute_filters:
            if attribute_filter.field not in TEXT_ATTRIBUTE_FIELDS:
                attribute_updates.append(attribute_filter)
                continue
            values = {
                str(product.get("attributes", {}).get(attribute_filter.field))
                for product in products
                if product.get("attributes", {}).get(attribute_filter.field) is not None
            }
            raw_values = (
                attribute_filter.value
                if isinstance(attribute_filter.value, list)
                else [attribute_filter.value]
            )
            unmapped_optional = any(
                self._resolve_unique(str(value), values) is None
                and attribute_filter.field not in required_fields
                for value in raw_values
            )
            if unmapped_optional:
                corrections.append(
                    {
                        "field": f"attributes.{attribute_filter.field}",
                        "action": "removed_unmapped_preference_filter",
                        "original": attribute_filter.value,
                        "reason": "kept_in_semantic_query",
                    }
                )
                continue
            mapped_values = [
                map_value(f"attributes.{attribute_filter.field}", str(value), values)
                for value in raw_values
            ]
            new_value: Any = mapped_values if isinstance(attribute_filter.value, list) else mapped_values[0]
            attribute_updates.append(attribute_filter.model_copy(update={"value": new_value}))
        updates["attribute_filters"] = attribute_updates

        if args.connectivity:
            connectivity_values = {
                str(product.get("attributes", {}).get("connectivity"))
                for product in products
                if product.get("attributes", {}).get("connectivity") is not None
            }
            updates["connectivity"] = map_preference(
                "connectivity", args.connectivity, connectivity_values
            )

        canonical = args.model_copy(update=updates)
        return CanonicalSearchArguments(canonical, corrections, mappings)

    def supports_field(self, field: str) -> bool:
        return field in self._catalog_fields

    @property
    def semantic_fields(self) -> tuple[str, ...]:
        return tuple(sorted(self._catalog_fields))

    def facet_values(self, field: str) -> tuple[str, ...]:
        """Return bounded schema metadata without exposing product records."""
        return tuple(sorted(self._global_facets.get(field, ())))

    def resolve_facet_namespaces(self, value: str) -> tuple[FacetResolution, ...]:
        """Find exact catalog namespaces for a raw facet value using startup indexes."""
        wanted = normalize_text(value)
        matches = []
        for field, candidates in self._global_facets.items():
            exact = sorted(
                candidate for candidate in candidates if normalize_text(candidate) == wanted
            )
            if len(exact) == 1:
                matches.append(
                    FacetResolution(field, value, "resolved", exact[0], tuple(exact))
                )
        return tuple(matches)

    def product_by_id(self, product_id: str) -> dict[str, Any] | None:
        return self._products_by_id.get(product_id)

    def resolve_predicate_value(self, field: str, value: str) -> FacetResolution:
        candidates = self._global_facets.get(field)
        if not candidates:
            return FacetResolution(field, value, "not_applicable", value)
        wanted = normalize_text(value)
        exact = sorted(
            candidate for candidate in candidates if normalize_text(candidate) == wanted
        )
        if len(exact) == 1:
            return FacetResolution(field, value, "resolved", exact[0], tuple(exact))
        prefix = sorted(
            candidate
            for candidate in candidates
            if normalize_text(candidate).startswith(wanted + " ")
            or wanted.startswith(normalize_text(candidate) + " ")
        )
        if len(prefix) == 1:
            return FacetResolution(field, value, "resolved", prefix[0], tuple(prefix))
        choices = tuple(exact or prefix)
        return FacetResolution(
            field,
            value,
            "ambiguous" if choices else "unmapped",
            None,
            choices[:10],
        )

    def resolve_entity(self, entity: ProductEntity) -> EntityResolution:
        """Resolve an extracted entity without applying natural-language rules."""

        wanted = normalize_text(entity.raw_text)
        if not wanted:
            return EntityResolution(entity.entity_id, entity.raw_text, "unresolved", None, ())
        index_order = {
            "product_id": ("product_id",),
            "sku": ("sku",),
            "model": ("model",),
            "model_family": (),
            "auto": ("product_id", "sku", "model", "name"),
        }[entity.identifier_type]
        for index_name in index_order:
            product_ids = self._identifier_indexes[index_name].get(wanted, set())
            if product_ids:
                return self._resolution_from_ids(
                    entity, product_ids, f"normalized_exact:{index_name}", 1.0
                )

        facet_fields = (
            ("model_family", "brand", "category_id")
            if entity.identifier_type in {"auto", "model_family"}
            else ()
        )
        facet_matches = [
            (field, candidate)
            for field in facet_fields
            for candidate in self._global_facets.get(field, ())
            if normalize_text(candidate) == wanted
        ]
        if len(facet_matches) == 1:
            field, value = facet_matches[0]
            return EntityResolution(
                entity.entity_id,
                entity.raw_text,
                "constraint_entity",
                None,
                (),
                "normalized_exact:facet",
                field,
                value,
            )
        if len(facet_matches) > 1:
            return EntityResolution(
                entity.entity_id,
                entity.raw_text,
                "ambiguous",
                None,
                (),
                "multiple_facet_namespaces",
            )

        tokens = tuple(token for token in re.split(r"\s+", wanted) if token)
        token_ids: set[str] | None = None
        for token in tokens:
            product_ids = self._entity_token_index.get(token, set())
            token_ids = set(product_ids) if token_ids is None else token_ids & product_ids
        if token_ids:
            return self._resolution_from_ids(entity, token_ids, "unique_token", 0.9)

        typo_ids: set[str] | None = None
        for token in tokens:
            vocabulary = (
                ({token} if token in self._entity_token_index else set())
                if any(character.isdigit() for character in token)
                else self._token_candidates(token)
            )
            product_ids = (
                set().union(*(self._entity_token_index[item] for item in vocabulary))
                if vocabulary
                else set()
            )
            typo_ids = product_ids if typo_ids is None else typo_ids & product_ids
        if typo_ids:
            same_length = {
                product_id
                for product_id in typo_ids
                if len(self._model_tokens_by_id.get(product_id, ())) == len(tokens)
            }
            if same_length:
                typo_ids = same_length
            return self._resolution_from_ids(entity, typo_ids, "single_edit_token", 0.78)
        return EntityResolution(
            entity.entity_id,
            entity.raw_text,
            "unresolved",
            None,
            (),
            "no_catalog_candidate",
        )

    def _resolution_from_ids(
        self,
        entity: ProductEntity,
        product_ids: set[str],
        resolution: str,
        score: float,
    ) -> EntityResolution:
        candidates = tuple(
            EntityCandidate(product_id, score, resolution) for product_id in sorted(product_ids)[:10]
        )
        if len(product_ids) == 1:
            return EntityResolution(
                entity.entity_id,
                entity.raw_text,
                "resolved",
                candidates[0].product_id,
                candidates,
            )
        return EntityResolution(
            entity.entity_id,
            entity.raw_text,
            "ambiguous",
            None,
            candidates,
            "multiple_catalog_candidates",
        )

    def _build_resolution_indexes(self, products: Sequence[dict[str, Any]]) -> None:
        indexes: dict[str, dict[str, set[str]]] = {
            name: {} for name in ("product_id", "sku", "model", "name")
        }
        token_index: dict[str, set[str]] = {}
        catalog_fields = {
            "product_id", "sku", "model", "name", "category_id", "brand",
            "model_family", "color_code", "price", "sale_price", "stock_status",
            "in_stock", "warranty_months", "rating",
        }
        vocabulary: set[str] = set()
        model_tokens_by_id: dict[str, tuple[str, ...]] = {}
        for product in products:
            product_id = str(product["product_id"])
            values = {
                "product_id": product_id,
                "sku": str(product["sku"]),
                "model": str(product["model"]),
                "name": str(product["name"]),
            }
            for name, raw_value in values.items():
                indexes[name].setdefault(normalize_text(raw_value), set()).add(product_id)
            model_tokens_by_id[product_id] = tuple(
                token for token in re.split(r"\s+", normalize_text(values["model"])) if token
            )
            entity_text = normalize_text(" ".join((values["model"], values["name"])))
            for token in set(re.split(r"\s+", entity_text)):
                if token:
                    vocabulary.add(token)
                    token_index.setdefault(token, set()).add(product_id)
            catalog_fields.update(str(field) for field in product.get("attributes", {}))
        deletion_index: dict[str, set[str]] = {}
        for token in vocabulary:
            keys = {token, *(token[:index] + token[index + 1 :] for index in range(len(token)))}
            for key in keys:
                deletion_index.setdefault(key, set()).add(token)
        self._identifier_indexes = indexes
        self._entity_token_index = token_index
        self._token_deletion_index = deletion_index
        signature_index: dict[str, set[str]] = {}
        for token in vocabulary:
            signature_index.setdefault("".join(sorted(token)), set()).add(token)
        self._token_signature_index = signature_index
        self._model_tokens_by_id = model_tokens_by_id
        self._catalog_fields = catalog_fields

    @staticmethod
    def _build_global_facets(
        facets_by_category: dict[str, dict[str, set[str]]],
    ) -> dict[str, set[str]]:
        global_facets: dict[str, set[str]] = {
            "category_id": set(facets_by_category),
        }
        field_names = {
            "brand": "brand",
            "model": "model",
            "model_family": "model_family",
            "color": "color_code",
        }
        for facets in facets_by_category.values():
            for source_field, values in facets.items():
                target_field = field_names.get(source_field)
                if target_field is None and source_field.startswith("attributes."):
                    target_field = source_field.removeprefix("attributes.")
                if target_field:
                    global_facets.setdefault(target_field, set()).update(values)
        return global_facets

    def _token_candidates(self, token: str) -> set[str]:
        keys = {token, *(token[:index] + token[index + 1 :] for index in range(len(token)))}
        candidates = set().union(*(self._token_deletion_index.get(key, set()) for key in keys))
        candidates.update(self._token_signature_index.get("".join(sorted(token)), set()))
        return {candidate for candidate in candidates if _edit_distance_at_most_one(token, candidate)}

    @staticmethod
    def _resolve_unique(value: str, candidates: set[str]) -> str | None:
        wanted = normalize_text(value)
        if not wanted:
            return None
        exact = [candidate for candidate in candidates if normalize_text(candidate) == wanted]
        if len(exact) == 1:
            return exact[0]
        prefix = [
            candidate
            for candidate in candidates
            if normalize_text(candidate).startswith(wanted + " ")
            or wanted.startswith(normalize_text(candidate) + " ")
        ]
        return prefix[0] if len(prefix) == 1 else None

    @staticmethod
    def _longest_query_facet(query: str, candidates: set[str]) -> str | None:
        normalized_query = normalize_text(query)
        matches = []
        for candidate in candidates:
            normalized = normalize_text(candidate)
            if normalized and normalized in normalized_query:
                matches.append((len(normalized), candidate))
        if not matches:
            return None
        longest = max(length for length, _ in matches)
        winners = [candidate for length, candidate in matches if length == longest]
        return winners[0] if len(winners) == 1 else None

    @staticmethod
    def _build_facet_registry(products: Sequence[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
        registry: dict[str, dict[str, set[str]]] = {}
        for product in products:
            category_id = str(product["category"]["id"])
            facets = registry.setdefault(
                category_id,
                {"brand": set(), "model": set(), "model_family": set(), "color": set()},
            )
            facets["brand"].add(str(product["brand"]))
            facets["model"].add(str(product["model"]))
            facets["model_family"].add(str(product["model_family"]))
            facets["color"].add(str(product["color"]["code"]))
            for field, value in product.get("attributes", {}).items():
                if isinstance(value, str):
                    facets.setdefault(f"attributes.{field}", set()).add(value)
        return registry

    @staticmethod
    def applied_filters(args: ProductSearchArguments) -> dict[str, Any]:
        return args.model_dump(
            exclude={
                "query",
                "search_intent",
                "requested_fields",
                "sort",
                "limit",
                "required_filter_fields",
            },
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


# Explicit adapter name for dependency injection when the future database-backed
# catalog is introduced. ProductCatalog remains as a backward-compatible name.
JsonlCatalogBackend = ProductCatalog
