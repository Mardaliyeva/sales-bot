from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.tools.catalog import CatalogLoadError, ProductCatalog, normalize_text
from app.tools.product_search import ProductSearchBackendError
from app.tools.schemas import AttributeFilter, MatchStatus, ProductSearchArguments, ProductSearchResult
from app.vectorstores.qdrant import (
    DEFAULT_QUERY_CANDIDATES,
    DEFAULT_SORT_CANDIDATES,
    VectorSearchHit,
)

logger = logging.getLogger(__name__)
ALTERNATIVE_LIMIT = 3
DEFAULT_ALTERNATIVE_MIN_SCORE = 0.39
VISUAL_PREFERENCE_FIELDS = ("color_code",)
TECHNICAL_PREFERENCE_FIELDS = (
    "min_price",
    "storage_gb",
    "ram_gb",
    "btu",
    "screen_size_in",
    "connectivity",
    "active_noise_cancellation",
)
AFFINITY_PREFERENCE_FIELDS = ("brand", "model_family")


class QueryEmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]: ...


class QdrantCandidateStore(Protocol):
    collection_name: str

    def count_candidates(self, args: ProductSearchArguments) -> int: ...

    def exact_candidates(
        self,
        args: ProductSearchArguments,
        *,
        include_structured_filters: bool,
    ) -> list[VectorSearchHit]: ...

    def search_candidates(
        self,
        vector: list[float],
        args: ProductSearchArguments,
        *,
        candidate_limit: int = DEFAULT_QUERY_CANDIDATES,
    ) -> list[VectorSearchHit]: ...


@dataclass(frozen=True)
class QdrantSearchExecution:
    result: ProductSearchResult
    debug_trace: dict[str, Any]


@dataclass(frozen=True)
class AlternativeSearchExecution:
    selected: list[tuple[VectorSearchHit, tuple[str, ...]]]
    semantic_hits: list[VectorSearchHit]
    stages: list[dict[str, Any]]
    semantic_state: str


class QdrantProductSearch:
    """Exact payload lookup plus semantic Qdrant search; no lexical fallback."""

    def __init__(
        self,
        catalog: ProductCatalog,
        embeddings: QueryEmbeddingBackend | None = None,
        store: QdrantCandidateStore | None = None,
        *,
        relevance_candidate_limit: int = DEFAULT_QUERY_CANDIDATES,
        sort_candidate_limit: int = DEFAULT_SORT_CANDIDATES,
        alternative_min_score: float = DEFAULT_ALTERNATIVE_MIN_SCORE,
    ) -> None:
        if not catalog.ready:
            raise ValueError("Qdrant axtarışı üçün kataloq əvvəlcədən yüklənməlidir")
        if (embeddings is None) != (store is None):
            raise ValueError("Embedding və Qdrant birlikdə konfiqurasiya edilməlidir")
        if relevance_candidate_limit <= 0 or sort_candidate_limit <= 0:
            raise ValueError("Qdrant namizəd limitləri müsbət olmalıdır")
        if not 0 <= alternative_min_score <= 1:
            raise ValueError("Alternativ semantic score həddi 0 və 1 arasında olmalıdır")
        self.catalog = catalog
        self.embeddings = embeddings
        self.store = store
        self.relevance_candidate_limit = relevance_candidate_limit
        self.sort_candidate_limit = sort_candidate_limit
        self.alternative_min_score = alternative_min_score

    @property
    def semantic_enabled(self) -> bool:
        return self.embeddings is not None and self.store is not None

    def search(self, args: ProductSearchArguments) -> ProductSearchResult:
        return self.search_with_trace(args).result

    def search_with_trace(self, args: ProductSearchArguments) -> QdrantSearchExecution:
        if self.embeddings is None or self.store is None:
            trace = self._empty_trace(args, semantic_state="not_configured")
            raise ProductSearchBackendError(
                "product_search_unavailable",
                "Məhsul axtarışı hazırda əlçatan deyil. Bir qədər sonra yenidən cəhd edin.",
                debug_trace=trace,
            )

        try:
            filtered_count = self.store.count_candidates(args)
            exact_unfiltered = self.store.exact_candidates(
                args,
                include_structured_filters=False,
            )
            exact_filtered = self.store.exact_candidates(
                args,
                include_structured_filters=True,
            )
        except Exception as exc:
            raise self._unavailable(args, "failed", exc) from exc

        if exact_filtered:
            ordered_hits = self._sort_hits(exact_filtered, args.sort)
            selected_hits = ordered_hits[: args.limit]
            products = self._hydrate(args, selected_hits, semantic_state="not_run_exact_match")
            result = self._result(
                args,
                products,
                match_status="exact_match",
                strict_total=len(exact_filtered),
                total=len(exact_filtered),
            )
            return QdrantSearchExecution(
                result=result,
                debug_trace=self._trace(
                    args,
                    filtered_count=filtered_count,
                    exact_unfiltered=exact_unfiltered,
                    exact_filtered=exact_filtered,
                    semantic_hits=[],
                    ordered_hits=ordered_hits,
                    hydrated_ids=[product["product_id"] for product in products],
                    semantic_state="not_run_exact_match",
                    exact_filter_conflict=False,
                    total=len(exact_filtered),
                    match_status="exact_match",
                    strict_total=len(exact_filtered),
                ),
            )

        if not args.has_exact_identifier and filtered_count > 0:
            candidate_limit = (
                self.sort_candidate_limit
                if args.sort != "relevance"
                else self.relevance_candidate_limit
            )
            try:
                vector = self._embed_query(args)
                semantic_hits = self.store.search_candidates(
                    vector,
                    args,
                    candidate_limit=candidate_limit,
                )
            except Exception as exc:
                raise self._unavailable(args, "failed", exc, filtered_count=filtered_count) from exc

            ordered_hits = self._sort_hits(semantic_hits, args.sort)
            selected_hits = ordered_hits[: args.limit]
            products = self._hydrate(args, selected_hits, semantic_state="active")
            match_status: MatchStatus = "matching_products" if products else "not_found"
            result = self._result(
                args,
                products,
                match_status=match_status,
                strict_total=filtered_count,
                total=filtered_count if products else 0,
            )
            logger.info(
                "product_search.qdrant_completed",
                extra={
                    "exact_count": 0,
                    "semantic_count": len(semantic_hits),
                    "returned_count": len(products),
                    "match_status": match_status,
                    "sort": args.sort,
                },
            )
            return QdrantSearchExecution(
                result=result,
                debug_trace=self._trace(
                    args,
                    filtered_count=filtered_count,
                    exact_unfiltered=[],
                    exact_filtered=[],
                    semantic_hits=semantic_hits,
                    ordered_hits=ordered_hits,
                    hydrated_ids=[product["product_id"] for product in products],
                    semantic_state="active" if products else "no_semantic_match",
                    exact_filter_conflict=False,
                    total=filtered_count if products else 0,
                    match_status=match_status,
                    strict_total=filtered_count,
                ),
            )

        try:
            alternatives = self._search_alternatives(args)
        except Exception as exc:
            raise self._unavailable(args, "failed", exc, filtered_count=filtered_count) from exc

        selected_hits = [hit for hit, _ in alternatives.selected]
        products = self._hydrate(
            args,
            selected_hits,
            semantic_state=alternatives.semantic_state,
        )
        differences: dict[str, list[str]] = {}
        visible_relaxed: list[str] = []
        for product, (_, relaxed_fields) in zip(products, alternatives.selected, strict=True):
            product_differences = self._product_differences(args, product, relaxed_fields)
            differences[product["product_id"]] = product_differences
            for field in relaxed_fields:
                if field not in visible_relaxed and self._field_was_requested(args, field):
                    visible_relaxed.append(field)
        match_status = "alternatives" if products else "not_found"
        result = self._result(
            args,
            products,
            match_status=match_status,
            strict_total=0,
            total=len(products),
            relaxed_fields=visible_relaxed,
            differences=differences,
        )
        logger.info(
            "product_search.qdrant_completed",
            extra={
                "exact_count": 0,
                "semantic_count": len(alternatives.semantic_hits),
                "returned_count": len(products),
                "match_status": match_status,
                "sort": args.sort,
            },
        )
        return QdrantSearchExecution(
            result=result,
            debug_trace=self._trace(
                args,
                filtered_count=filtered_count,
                exact_unfiltered=exact_unfiltered,
                exact_filtered=[],
                semantic_hits=alternatives.semantic_hits,
                ordered_hits=selected_hits,
                hydrated_ids=[product["product_id"] for product in products],
                semantic_state=alternatives.semantic_state,
                exact_filter_conflict=bool(exact_unfiltered and not exact_filtered),
                total=len(products),
                match_status=match_status,
                strict_total=0,
                relaxed_fields=visible_relaxed,
                alternative_stages=alternatives.stages,
            ),
        )

    def _embed_query(self, args: ProductSearchArguments) -> list[float]:
        if self.embeddings is None:
            raise RuntimeError("Embedding backend konfiqurasiya edilməyib")
        vectors = self.embeddings.embed([args.query], text_version=DEFAULT_TEXT_VERSION)
        if len(vectors) != 1:
            raise ValueError("Query üçün dəqiq bir embedding yaranmalıdır")
        return vectors[0]

    def _search_alternatives(self, args: ProductSearchArguments) -> AlternativeSearchExecution:
        if self.store is None:
            raise RuntimeError("Qdrant store konfiqurasiya edilməyib")
        stages = self._alternative_stages(args)
        counted_stages: list[tuple[ProductSearchArguments, tuple[str, ...], int]] = []
        for stage_args, relaxed_fields in stages:
            count = self.store.count_candidates(stage_args)
            counted_stages.append((stage_args, relaxed_fields, count))

        stage_trace = [
            {
                "relaxed_fields": list(relaxed_fields),
                "filters": ProductCatalog.applied_filters(stage_args),
                "filtered_count": count,
                "candidate_count": 0,
                "selected_count": 0,
            }
            for stage_args, relaxed_fields, count in counted_stages
        ]
        if not any(count for _, _, count in counted_stages):
            return AlternativeSearchExecution(
                selected=[],
                semantic_hits=[],
                stages=stage_trace,
                semantic_state="not_run_no_alternative_candidates",
            )

        vector = self._embed_query(args)
        candidate_limit = (
            self.sort_candidate_limit if args.sort != "relevance" else self.relevance_candidate_limit
        )
        target = min(args.limit, ALTERNATIVE_LIMIT)
        selected: list[tuple[VectorSearchHit, tuple[str, ...]]] = []
        selected_ids: set[str] = set()
        semantic_by_id: dict[str, VectorSearchHit] = {}

        for index, (stage_args, relaxed_fields, count) in enumerate(counted_stages):
            if count == 0 or len(selected) >= target:
                continue
            hits = self.store.search_candidates(
                vector,
                stage_args,
                candidate_limit=candidate_limit,
            )
            eligible = [hit for hit in hits if hit.score >= self.alternative_min_score]
            stage_trace[index]["candidate_count"] = len(eligible)
            for hit in eligible:
                previous = semantic_by_id.get(hit.product_id)
                if previous is None or hit.score > previous.score:
                    semantic_by_id[hit.product_id] = hit
            for hit in self._sort_hits(eligible, args.sort):
                if hit.product_id in selected_ids:
                    continue
                selected.append((hit, relaxed_fields))
                selected_ids.add(hit.product_id)
                stage_trace[index]["selected_count"] += 1
                if len(selected) >= target:
                    break

        semantic_hits = sorted(semantic_by_id.values(), key=lambda hit: (-hit.score, hit.product_id))
        return AlternativeSearchExecution(
            selected=selected,
            semantic_hits=semantic_hits,
            stages=stage_trace,
            semantic_state="active_alternatives" if selected else "no_credible_alternatives",
        )

    def _alternative_stages(
        self,
        args: ProductSearchArguments,
    ) -> list[tuple[ProductSearchArguments, tuple[str, ...]]]:
        base = self._alternative_base_args(args)
        stages: list[tuple[ProductSearchArguments, tuple[str, ...]]] = [(base, ())]
        current = base
        cumulative: list[str] = []
        required = set(args.required_filter_fields)
        groups = (
            (VISUAL_PREFERENCE_FIELDS, False),
            (TECHNICAL_PREFERENCE_FIELDS, True),
            (AFFINITY_PREFERENCE_FIELDS, False),
        )
        for fields, relax_attributes in groups:
            updates: dict[str, Any] = {}
            removed: list[str] = []
            for field in fields:
                if field not in required and getattr(current, field) is not None:
                    updates[field] = None
                    removed.append(field)
            if relax_attributes:
                kept_attributes = []
                for attribute_filter in current.attribute_filters:
                    if attribute_filter.field in required:
                        kept_attributes.append(attribute_filter)
                    else:
                        removed.append(attribute_filter.field)
                if len(kept_attributes) != len(current.attribute_filters):
                    updates["attribute_filters"] = kept_attributes
            if not removed:
                continue
            current = current.model_copy(update=updates)
            for field in removed:
                if field not in cumulative:
                    cumulative.append(field)
            stages.append((current, tuple(cumulative)))
        return stages

    def _alternative_base_args(self, args: ProductSearchArguments) -> ProductSearchArguments:
        updates: dict[str, Any] = {
            "product_id": None,
            "sku": None,
            "model": None,
            "limit": min(args.limit, ALTERNATIVE_LIMIT),
        }
        query_text = normalize_text(" ".join(value for value in (args.query, args.model) if value))
        products = [
            product
            for product in self.catalog.products
            if args.category_id is None or product["category"]["id"] == args.category_id
        ]
        if args.brand is None:
            inferred_brand = self._catalog_phrase(query_text, {product["brand"] for product in products})
            if inferred_brand is not None:
                updates["brand"] = inferred_brand
        if args.model_family is None:
            inferred_family = self._catalog_phrase(
                query_text,
                {product["model_family"] for product in products},
            )
            if inferred_family is not None:
                updates["model_family"] = inferred_family
        return args.model_copy(update=updates)

    @staticmethod
    def _catalog_phrase(query_text: str, values: set[str]) -> str | None:
        matches = []
        for value in values:
            normalized = normalize_text(value)
            if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", query_text):
                matches.append((len(normalized), value))
        return max(matches, default=(0, None))[1]

    @staticmethod
    def _requested_label(args: ProductSearchArguments) -> str | None:
        return args.model or args.sku or args.product_id

    @staticmethod
    def _field_was_requested(args: ProductSearchArguments, field: str) -> bool:
        if any(attribute_filter.field == field for attribute_filter in args.attribute_filters):
            return True
        return hasattr(args, field) and getattr(args, field) is not None

    def _product_differences(
        self,
        args: ProductSearchArguments,
        product: dict[str, Any],
        relaxed_fields: tuple[str, ...],
    ) -> list[str]:
        differences: list[str] = []
        if args.model and normalize_text(str(product.get("model", ""))) != normalize_text(args.model):
            differences.append(f"Model fərqlidir: {product.get('model') or product.get('name')}")
        elif args.sku or args.product_id:
            differences.append("Dəqiq məhsul əvəzinə yaxın alternativdir")
        for field in relaxed_fields:
            if not self._field_was_requested(args, field):
                continue
            if self._product_matches_field(args, product, field):
                continue
            difference = self._format_difference(product, field)
            if difference and difference not in differences:
                differences.append(difference)
        return differences

    def _product_matches_field(
        self,
        args: ProductSearchArguments,
        product: dict[str, Any],
        field: str,
    ) -> bool:
        attribute_filters = [item for item in args.attribute_filters if item.field == field]
        if attribute_filters:
            actual = product.get("attributes", {}).get(field)
            return all(self._attribute_matches(item, actual) for item in attribute_filters)
        actual = self._product_field_value(product, field)
        requested = getattr(args, field, None)
        if field == "min_price":
            return actual is not None and requested is not None and float(actual) >= float(requested)
        if isinstance(requested, str) and isinstance(actual, str):
            return normalize_text(actual) == normalize_text(requested)
        return actual == requested

    @staticmethod
    def _attribute_matches(attribute_filter: AttributeFilter, actual: Any) -> bool:
        value = attribute_filter.value
        if attribute_filter.operator == "gte":
            return actual is not None and float(actual) >= float(value)  # type: ignore[arg-type]
        if attribute_filter.operator == "lte":
            return actual is not None and float(actual) <= float(value)  # type: ignore[arg-type]
        if attribute_filter.operator == "in":
            return isinstance(value, list) and any(
                normalize_text(str(actual)) == normalize_text(str(item)) for item in value
            )
        if attribute_filter.operator == "contains_any":
            actual_values = actual if isinstance(actual, list) else [actual]
            expected_values = value if isinstance(value, list) else [value]
            return any(
                normalize_text(str(actual_item)) == normalize_text(str(expected_item))
                for actual_item in actual_values
                for expected_item in expected_values
            )
        if isinstance(value, str) and isinstance(actual, str):
            return normalize_text(actual) == normalize_text(value)
        return actual == value

    @staticmethod
    def _product_field_value(product: dict[str, Any], field: str) -> Any:
        if field == "brand":
            return product.get("brand")
        if field == "model_family":
            return product.get("model_family")
        if field == "color_code":
            return product.get("color", {}).get("code")
        if field == "min_price":
            return product.get("price", {}).get("sale")
        return product.get("attributes", {}).get(field)

    @staticmethod
    def _format_difference(product: dict[str, Any], field: str) -> str | None:
        attributes = product.get("attributes", {})
        values: dict[str, tuple[str, Any, str]] = {
            "brand": ("Brend fərqlidir", product.get("brand"), ""),
            "model_family": ("Model ailəsi fərqlidir", product.get("model_family"), ""),
            "color_code": ("Rəng fərqlidir", product.get("color", {}).get("name"), ""),
            "min_price": (
                "Qiymət istənilən minimumdan aşağıdır",
                product.get("price", {}).get("sale"),
                " AZN",
            ),
            "storage_gb": ("Yaddaş fərqlidir", attributes.get("storage_gb"), " GB"),
            "ram_gb": ("RAM fərqlidir", attributes.get("ram_gb"), " GB"),
            "btu": ("BTU fərqlidir", attributes.get("btu"), " BTU"),
            "screen_size_in": ("Ekran ölçüsü fərqlidir", attributes.get("screen_size_in"), '"'),
            "connectivity": ("Bağlantı fərqlidir", attributes.get("connectivity"), ""),
            "active_noise_cancellation": (
                "Aktiv səsboğma fərqlidir",
                "var" if attributes.get("active_noise_cancellation") else "yoxdur",
                "",
            ),
        }
        label, actual, suffix = values.get(
            field,
            (field.replace("_", " ").capitalize() + " fərqlidir", attributes.get(field), ""),
        )
        return f"{label}: {actual}{suffix}" if actual is not None else f"{label}: məlumat yoxdur"

    def debug_source_state(self) -> dict[str, Any]:
        category_counts = self.catalog.manifest.get("category_counts", {})
        return {
            "product_catalog_json": {
                "configured": True,
                "ready": self.catalog.ready,
                "product_count": len(self.catalog.products),
                "dataset_version": self.catalog.manifest.get("dataset_version"),
                "categories": sorted(category_counts),
                "role": "full_product_hydration",
            },
            "semantic_qdrant": {
                "configured": self.semantic_enabled,
                "ready": self.semantic_enabled,
                "collection": getattr(self.store, "collection_name", None),
                "alternative_min_score": self.alternative_min_score,
                "role": "exact_filters_and_semantic_candidates",
            },
            "documents": {
                "configured": False,
                "detail": "Document və config məlumat mənbəyi bu versiyada yoxdur.",
            },
        }

    def _hydrate(
        self,
        args: ProductSearchArguments,
        hits: Sequence[VectorSearchHit],
        *,
        semantic_state: str,
    ) -> list[dict[str, Any]]:
        try:
            return self.catalog.hydrate([hit.product_id for hit in hits])
        except CatalogLoadError as exc:
            raise self._unavailable(args, semantic_state, exc) from exc

    @staticmethod
    def _sort_hits(hits: Sequence[VectorSearchHit], sort: str) -> list[VectorSearchHit]:
        semantic_rank = {hit.product_id: rank for rank, hit in enumerate(hits, start=1)}
        if sort == "price_asc":
            return sorted(
                hits,
                key=lambda hit: (
                    float(hit.payload.get("sale_price", float("inf"))),
                    semantic_rank[hit.product_id],
                    hit.product_id,
                ),
            )
        if sort == "price_desc":
            return sorted(
                hits,
                key=lambda hit: (
                    -float(hit.payload.get("sale_price", float("-inf"))),
                    semantic_rank[hit.product_id],
                    hit.product_id,
                ),
            )
        if sort == "rating_desc":
            return sorted(
                hits,
                key=lambda hit: (
                    -float(hit.payload.get("rating", 0.0)),
                    semantic_rank[hit.product_id],
                    float(hit.payload.get("sale_price", float("inf"))),
                    hit.product_id,
                ),
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.product_id))

    @staticmethod
    def _result(
        args: ProductSearchArguments,
        products: Sequence[dict[str, Any]],
        *,
        match_status: MatchStatus,
        strict_total: int,
        total: int,
        relaxed_fields: Sequence[str] = (),
        differences: dict[str, list[str]] | None = None,
    ) -> ProductSearchResult:
        item_differences = differences or {}
        items = []
        for product in products:
            item = ProductCatalog.to_result(product)
            product_differences = item_differences.get(product["product_id"], [])
            if product_differences:
                item = item.model_copy(update={"differences": product_differences})
            items.append(item)
        return ProductSearchResult(
            match_status=match_status,
            requested_label=QdrantProductSearch._requested_label(args),
            strict_total=strict_total,
            total=total,
            applied_filters=ProductCatalog.applied_filters(args),
            relaxed_fields=list(relaxed_fields),
            items=items,
        )

    @staticmethod
    def _candidate(hit: VectorSearchHit, rank: int, *, selected: bool = False) -> dict[str, Any]:
        return {
            "product_id": hit.product_id,
            "name": hit.payload.get("name"),
            "rank": rank,
            "score": hit.score,
            "sale_price": hit.payload.get("sale_price"),
            "rating": hit.payload.get("rating"),
            "selected": selected,
        }

    def _trace(
        self,
        args: ProductSearchArguments,
        *,
        filtered_count: int,
        exact_unfiltered: Sequence[VectorSearchHit],
        exact_filtered: Sequence[VectorSearchHit],
        semantic_hits: Sequence[VectorSearchHit],
        ordered_hits: Sequence[VectorSearchHit],
        hydrated_ids: Sequence[str],
        semantic_state: str,
        exact_filter_conflict: bool,
        total: int,
        match_status: MatchStatus,
        strict_total: int,
        relaxed_fields: Sequence[str] = (),
        alternative_stages: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        selected_ids = set(hydrated_ids)
        return {
            "mode": "qdrant_only_v2",
            "query": args.query,
            "filters": ProductCatalog.applied_filters(args),
            "sort": args.sort,
            "qdrant_checked": True,
            "filtered_count": filtered_count,
            "exact_product_ids": [hit.product_id for hit in exact_unfiltered],
            "matching_exact_product_ids": [hit.product_id for hit in exact_filtered],
            "exact_filter_conflict": exact_filter_conflict,
            "exact_candidates": [
                self._candidate(hit, rank, selected=hit.product_id in selected_ids)
                for rank, hit in enumerate(exact_filtered or exact_unfiltered, start=1)
            ],
            "semantic_candidates": [
                self._candidate(hit, rank, selected=hit.product_id in selected_ids)
                for rank, hit in enumerate(semantic_hits, start=1)
            ],
            "semantic_state": semantic_state,
            "match_status": match_status,
            "requested_label": self._requested_label(args),
            "strict_total": strict_total,
            "relaxed_fields": list(relaxed_fields),
            "alternative_stages": list(alternative_stages),
            "sorted_candidates": [
                self._candidate(hit, rank, selected=hit.product_id in selected_ids)
                for rank, hit in enumerate(ordered_hits, start=1)
            ],
            "hydrated_product_ids": list(hydrated_ids),
            "returned_product_ids": list(hydrated_ids),
            "total": total,
        }

    def _empty_trace(
        self,
        args: ProductSearchArguments,
        *,
        semantic_state: str,
        filtered_count: int = 0,
    ) -> dict[str, Any]:
        return self._trace(
            args,
            filtered_count=filtered_count,
            exact_unfiltered=[],
            exact_filtered=[],
            semantic_hits=[],
            ordered_hits=[],
            hydrated_ids=[],
            semantic_state=semantic_state,
            exact_filter_conflict=False,
            total=0,
            match_status="not_found",
            strict_total=0,
        )

    def _unavailable(
        self,
        args: ProductSearchArguments,
        semantic_state: str,
        exc: Exception,
        *,
        filtered_count: int = 0,
    ) -> ProductSearchBackendError:
        logger.warning(
            "product_search.qdrant_failed",
            extra={"error_type": type(exc).__name__, "semantic_state": semantic_state},
        )
        return ProductSearchBackendError(
            "product_search_unavailable",
            "Məhsul axtarışı hazırda əlçatan deyil. Bir qədər sonra yenidən cəhd edin.",
            debug_trace=self._empty_trace(
                args,
                semantic_state=semantic_state,
                filtered_count=filtered_count,
            ),
        )
