from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.retrieval.semantic_plan import (
    SemanticPlanValidationError,
    compile_semantic_plan,
    expression_matches,
    iter_predicates,
    preference_score,
)
from app.tools.catalog import CatalogLoadError, ProductCatalog, normalize_text
from app.tools.product_search import ProductSearchBackendError
from app.tools.schemas import (
    AttributeFilter,
    MatchStatus,
    ProductQueryPlan,
    ProductSearchArguments,
    ProductSearchResult,
    RankingObjective,
)
from app.vectorstores.qdrant import (
    DEFAULT_QUERY_CANDIDATES,
    DEFAULT_SORT_CANDIDATES,
    QdrantProductStore,
    VectorSearchHit,
)

logger = logging.getLogger(__name__)
ALTERNATIVE_LIMIT = 3
DEFAULT_ALTERNATIVE_MIN_SCORE = 0.39
DEFAULT_ENTITY_RESOLUTION_MIN_SCORE = 0.62
DEFAULT_ENTITY_RESOLUTION_MARGIN = 0.06
DEFAULT_DIRECTIONAL_SEMANTIC_CANDIDATES = 50
DEFAULT_DIRECTIONAL_FIELD_CANDIDATES = 20
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

    def ordered_candidates(
        self,
        args: ProductSearchArguments,
        *,
        field: str,
        direction: str,
        candidate_limit: int = DEFAULT_DIRECTIONAL_FIELD_CANDIDATES,
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
    candidate_generation_lanes: list[dict[str, Any]]
    ranking_components: dict[str, dict[str, Any]]


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
        entity_resolution_min_score: float = DEFAULT_ENTITY_RESOLUTION_MIN_SCORE,
        entity_resolution_margin: float = DEFAULT_ENTITY_RESOLUTION_MARGIN,
        directional_ranking_enabled: bool = True,
    ) -> None:
        if not catalog.ready:
            raise ValueError("Qdrant axtarışı üçün kataloq əvvəlcədən yüklənməlidir")
        if (embeddings is None) != (store is None):
            raise ValueError("Embedding və Qdrant birlikdə konfiqurasiya edilməlidir")
        if relevance_candidate_limit <= 0 or sort_candidate_limit <= 0:
            raise ValueError("Qdrant namizəd limitləri müsbət olmalıdır")
        if not 0 <= alternative_min_score <= 1:
            raise ValueError("Alternativ semantic score həddi 0 və 1 arasında olmalıdır")
        if not 0 <= entity_resolution_min_score <= 1:
            raise ValueError("Entity resolution score həddi 0 və 1 arasında olmalıdır")
        if not 0 <= entity_resolution_margin <= 1:
            raise ValueError("Entity resolution margin 0 və 1 arasında olmalıdır")
        self.catalog = catalog
        self.embeddings = embeddings
        self.store = store
        self.relevance_candidate_limit = relevance_candidate_limit
        self.sort_candidate_limit = sort_candidate_limit
        self.alternative_min_score = alternative_min_score
        self.entity_resolution_min_score = entity_resolution_min_score
        self.entity_resolution_margin = entity_resolution_margin
        self.directional_ranking_enabled = directional_ranking_enabled

    @property
    def semantic_enabled(self) -> bool:
        return self.embeddings is not None and self.store is not None

    def search(self, args: ProductSearchArguments | ProductQueryPlan) -> ProductSearchResult:
        return self.search_with_trace(args).result

    def search_with_trace(
        self,
        args: ProductSearchArguments | ProductQueryPlan,
    ) -> QdrantSearchExecution:
        if isinstance(args, ProductQueryPlan):
            return self._search_semantic_plan(args)
        return self._search_flat_with_trace(args)

    def _search_semantic_plan(self, plan: ProductQueryPlan) -> QdrantSearchExecution:
        try:
            compilation = compile_semantic_plan(plan, self.catalog)
        except SemanticPlanValidationError as exc:
            return self._semantic_clarification(
                plan,
                canonical_hash=None,
                clarification={
                    "reason": "semantic_plan_invalid",
                    "question": "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?",
                    "detail": str(exc),
                },
                trace={"validation_error": str(exc)},
            )
        semantic_resolution_trace: list[dict[str, Any]] = []
        if compilation.clarification is None and self.semantic_enabled:
            try:
                overrides, semantic_resolution_trace, ambiguous_entity_ids = (
                    self._semantic_entity_resolution(compilation)
                )
            except Exception as exc:
                fallback_arguments = (
                    compilation.arguments[0]
                    if compilation.arguments
                    else ProductSearchArguments(query=plan.query)
                )
                raise self._unavailable(
                    fallback_arguments,
                    "entity_resolution_failed",
                    exc,
                ) from exc
            if ambiguous_entity_ids:
                clarification = {
                    "reason": "ambiguous_entity",
                    "question": "Hansı dəqiq məhsulu nəzərdə tutursunuz?",
                    "entity_ids": ambiguous_entity_ids,
                }
                return self._semantic_clarification(
                    plan,
                    canonical_hash=compilation.canonical_hash,
                    clarification=clarification,
                    trace={
                        "mode": "semantic_plan_v1",
                        "raw_semantic_plan": plan.model_dump(mode="json"),
                        "canonical_semantic_plan": compilation.canonical_plan,
                        "grounded_semantic_plan": compilation.canonical_plan,
                        "canonical_query_hash": compilation.canonical_hash,
                        "evidence_validation": compilation.evidence_validation,
                        "numeric_provenance": list(compilation.numeric_provenance),
                        "plan_corrections": list(compilation.plan_corrections),
                        "field_capability_resolution": list(
                            compilation.field_capability_resolution
                        ),
                        "facet_mapping": list(compilation.facet_mapping),
                        "semantic_entity_candidates": semantic_resolution_trace,
                    },
                )
            if overrides:
                compilation = compile_semantic_plan(
                    plan,
                    self.catalog,
                    resolution_overrides=overrides,
                )
        semantic_trace: dict[str, Any] = {
            "mode": "semantic_plan_v1",
            "raw_semantic_plan": plan.model_dump(mode="json"),
            "canonical_semantic_plan": compilation.canonical_plan,
            "grounded_semantic_plan": compilation.canonical_plan,
            "canonical_query_hash": compilation.canonical_hash,
            "evidence_validation": compilation.evidence_validation,
            "numeric_provenance": list(compilation.numeric_provenance),
            "plan_corrections": list(compilation.plan_corrections),
            "field_capability_resolution": list(
                compilation.field_capability_resolution
            ),
            "ranking_objectives": [
                objective.model_dump(mode="json")
                for objective in compilation.plan.ranking_objectives
            ],
            "resolved_entities": [self._resolution_trace(item) for item in compilation.resolutions],
            "entity_candidates": [self._resolution_trace(item) for item in compilation.resolutions],
            "semantic_entity_candidates": semantic_resolution_trace,
            "facet_mapping": list(compilation.facet_mapping),
            "unavailable_requested_values": list(
                compilation.unavailable_requested_values
            ),
            "retrieval_executed": False,
            "ambiguity_reason": (
                compilation.clarification.get("reason") if compilation.clarification else None
            ),
        }
        if compilation.clarification:
            return self._semantic_clarification(
                plan,
                canonical_hash=compilation.canonical_hash,
                clarification=compilation.clarification,
                trace=semantic_trace,
            )
        if compilation.deterministic_empty:
            return self._semantic_unavailable_match(plan, compilation, semantic_trace)
        if plan.operation == "compare":
            return self._execute_comparison(compilation, semantic_trace)

        selected_execution: QdrantSearchExecution | None = None
        branch_traces: list[dict[str, Any]] = []
        for branch_index, arguments in enumerate(compilation.arguments, start=1):
            execution = self._search_flat_with_trace(arguments)
            branch_traces.append(
                {
                    "branch": branch_index,
                    "compiled_expression": (
                        arguments.semantic_filter_expression.model_dump(mode="json")
                        if arguments.semantic_filter_expression
                        else None
                    ),
                    "compiled_filter": self._compiled_filter_trace(arguments),
                    "match_status": execution.result.match_status,
                    "strict_total": execution.result.strict_total,
                    "retrieval": execution.debug_trace,
                }
            )
            selected_execution = execution
            if execution.result.strict_total > 0 or execution.result.match_status in {
                "exact_match",
                "matching_products",
            }:
                break
        if selected_execution is None:
            return self._semantic_clarification(
                plan,
                canonical_hash=compilation.canonical_hash,
                clarification={
                    "reason": "semantic_plan_invalid",
                    "question": "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?",
                },
                trace=semantic_trace,
            )
        result = selected_execution.result
        requested_id = result.requested_item.product_id if result.requested_item else None
        if requested_id:
            filtered_items = [item for item in result.items if item.product_id != requested_id]
            if len(filtered_items) != len(result.items):
                display_ids = [
                    product_id
                    for product_id in result.display_product_ids
                    if product_id != requested_id
                ]
                result = result.model_copy(
                    update={
                        "items": filtered_items,
                        "total": len(filtered_items),
                        "display_product_ids": display_ids,
                        "recommended_product_id": display_ids[0] if display_ids else requested_id,
                    }
                )
        result = result.model_copy(
            update={
                "requested_label": next(
                    (
                        entity.raw_text
                        for entity in reversed(plan.entities)
                        if entity.state == "selected"
                    ),
                    result.requested_label,
                ),
                "operation": plan.operation,
                "resolved_entities": [
                    self._resolution_trace(item) for item in compilation.resolutions
                ],
                "entity_results": [
                    {
                        "entity_id": item.entity_id,
                        "status": item.status,
                        "product_id": item.product_id,
                    }
                    for item in compilation.resolutions
                ],
                "canonical_query_hash": compilation.canonical_hash,
                "clarification": None,
                "unavailable_requested_values": list(
                    compilation.unavailable_requested_values
                ),
                "ranking_objectives": compilation.plan.ranking_objectives,
                "plan_corrections": list(compilation.plan_corrections),
            }
        )
        semantic_trace["compiled_filter"] = branch_traces[-1]["compiled_filter"]
        semantic_trace["fallback_branches"] = branch_traces
        semantic_trace["selected_branch"] = len(branch_traces)
        semantic_trace["match_status"] = result.match_status
        semantic_trace["returned_product_ids"] = result.display_product_ids
        flat_trace = selected_execution.debug_trace
        semantic_trace.update(
            {
                "query": plan.query,
                "sort": plan.sort,
                "qdrant_checked": flat_trace.get("qdrant_checked", True),
                "retrieval_executed": True,
                "semantic_state": flat_trace.get("semantic_state"),
                "filtered_count": flat_trace.get("filtered_count", 0),
                "filters": branch_traces[-1]["compiled_filter"],
                "exact_candidates": flat_trace.get("exact_candidates", []),
                "exact_product_ids": flat_trace.get("exact_product_ids", []),
                "matching_exact_product_ids": flat_trace.get(
                    "matching_exact_product_ids", []
                ),
                "semantic_candidates": flat_trace.get("semantic_candidates", []),
                "sorted_candidates": flat_trace.get("sorted_candidates", []),
                "hydrated_product_ids": flat_trace.get("hydrated_product_ids", []),
                "strict_total": result.strict_total,
                "relaxed_fields": result.relaxed_fields,
                "alternative_stages": flat_trace.get("alternative_stages", []),
                "exact_filter_conflict": flat_trace.get("exact_filter_conflict", False),
                "candidate_generation_lanes": flat_trace.get(
                    "candidate_generation_lanes",
                    [],
                ),
                "ranking_components": flat_trace.get("ranking_components", {}),
                "ranking_mode": flat_trace.get("ranking_mode", "none"),
                "total": result.total,
            }
        )
        return QdrantSearchExecution(result=result, debug_trace=semantic_trace)

    @staticmethod
    def _compiled_filter_trace(arguments: ProductSearchArguments) -> dict[str, Any] | None:
        compiled = QdrantProductStore.build_filter(arguments)
        return compiled.model_dump(mode="json", exclude_none=True) if compiled else None

    def _semantic_entity_resolution(
        self,
        compilation: Any,
    ) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
        if self.embeddings is None or self.store is None:
            return {}, [], []
        unresolved = [
            resolution
            for resolution in compilation.resolutions
            if resolution.status == "unresolved"
        ]
        entity_by_id = {
            entity.entity_id: entity
            for entity in compilation.plan.entities
            if entity.state == "selected"
        }
        overrides: dict[str, str] = {}
        trace: list[dict[str, Any]] = []
        ambiguous: list[str] = []
        for resolution in unresolved:
            entity = entity_by_id.get(resolution.entity_id)
            if entity is None or entity.identifier_type != "auto":
                trace.append(
                    {
                        "entity_id": resolution.entity_id,
                        "raw_text": resolution.raw_text,
                        "candidates": [],
                        "decision": "exact_identifier_unresolved",
                    }
                )
                continue
            vectors = self.embeddings.embed(
                [resolution.raw_text],
                text_version=DEFAULT_TEXT_VERSION,
            )
            if len(vectors) != 1:
                raise ValueError("Entity resolution üçün bir embedding tələb olunur")
            arguments = ProductSearchArguments(
                query=resolution.raw_text,
                limit=3,
            )
            hits = self.store.search_candidates(
                vectors[0],
                arguments,
                candidate_limit=5,
            )
            strong = [
                hit for hit in hits if hit.score >= self.entity_resolution_min_score
            ]
            entry = {
                "entity_id": resolution.entity_id,
                "raw_text": resolution.raw_text,
                "candidates": [
                    {
                        "product_id": hit.product_id,
                        "score": hit.score,
                        "name": hit.payload.get("name"),
                    }
                    for hit in hits
                ],
                "decision": "unresolved",
            }
            if strong:
                runner_up = strong[1] if len(strong) > 1 else None
                if (
                    runner_up is not None
                    and strong[0].score - runner_up.score < self.entity_resolution_margin
                ):
                    ambiguous.append(resolution.entity_id)
                    entry["decision"] = "ambiguous"
                else:
                    overrides[resolution.entity_id] = strong[0].product_id
                    entry["decision"] = "resolved"
                    entry["product_id"] = strong[0].product_id
            trace.append(entry)
        return overrides, trace, ambiguous

    def _execute_comparison(
        self,
        compilation: Any,
        semantic_trace: dict[str, Any],
    ) -> QdrantSearchExecution:
        items = []
        entity_results: list[dict[str, Any]] = []
        retrievals: list[dict[str, Any]] = []
        for resolution, arguments in zip(
            compilation.resolutions,
            compilation.arguments,
            strict=True,
        ):
            if resolution.status != "resolved":
                entity_results.append(
                    {
                        "entity_id": resolution.entity_id,
                        "status": "not_found",
                        "product_id": None,
                    }
                )
                continue
            execution = self._search_flat_with_trace(arguments)
            retrievals.append(execution.debug_trace)
            exact_item = execution.result.requested_item
            if exact_item is None and execution.result.items:
                exact_item = execution.result.items[0]
            entity_results.append(
                {
                    "entity_id": resolution.entity_id,
                    "status": execution.result.match_status,
                    "product_id": exact_item.product_id if exact_item else None,
                    "constraint_conflicts": execution.result.constraint_conflicts,
                }
            )
            if exact_item and all(item.product_id != exact_item.product_id for item in items):
                items.append(exact_item)
        items = items[:3]
        display_ids = [item.product_id for item in items]
        all_resolved = bool(entity_results) and all(
            item.get("product_id") for item in entity_results
        )
        result = ProductSearchResult(
            match_status="exact_match" if all_resolved else "matching_products",
            requested_label=" / ".join(
                entity.raw_text
                for entity in compilation.plan.entities
                if entity.state == "selected"
            )
            or None,
            strict_total=len(items),
            total=len(items),
            applied_filters={},
            items=items,
            recommended_product_id=(
                display_ids[0]
                if display_ids and compilation.plan.recommendation_requested
                else None
            ),
            display_product_ids=display_ids,
            operation="compare",
            resolved_entities=[
                self._resolution_trace(item) for item in compilation.resolutions
            ],
            entity_results=entity_results,
            canonical_query_hash=compilation.canonical_hash,
            unavailable_requested_values=list(
                compilation.unavailable_requested_values
            ),
            ranking_objectives=compilation.plan.ranking_objectives,
            plan_corrections=list(compilation.plan_corrections),
        )
        semantic_trace["comparison_retrievals"] = retrievals
        semantic_trace["entity_results"] = entity_results
        semantic_trace["match_status"] = result.match_status
        semantic_trace["returned_product_ids"] = display_ids
        semantic_trace.update(
            {
                "query": compilation.plan.query,
                "sort": compilation.plan.sort,
                "qdrant_checked": True,
                "retrieval_executed": True,
                "semantic_state": "comparison",
                "filtered_count": len(items),
                "filters": {},
                "exact_candidates": [],
                "exact_product_ids": display_ids,
                "matching_exact_product_ids": display_ids,
                "semantic_candidates": [],
                "sorted_candidates": [],
                "hydrated_product_ids": display_ids,
                "strict_total": len(items),
                "relaxed_fields": [],
                "alternative_stages": [],
                "exact_filter_conflict": False,
                "total": len(items),
            }
        )
        return QdrantSearchExecution(result=result, debug_trace=semantic_trace)

    def _semantic_unavailable_match(
        self,
        plan: ProductQueryPlan,
        compilation: Any,
        trace: dict[str, Any],
    ) -> QdrantSearchExecution:
        result = ProductSearchResult(
            match_status="not_found",
            requested_label=next(
                (
                    entity.raw_text
                    for entity in reversed(plan.entities)
                    if entity.state == "selected"
                ),
                None,
            ),
            strict_total=0,
            total=0,
            applied_filters={},
            items=[],
            operation=plan.operation,
            resolved_entities=[
                self._resolution_trace(item) for item in compilation.resolutions
            ],
            entity_results=[
                {
                    "entity_id": item.entity_id,
                    "status": item.status,
                    "product_id": item.product_id,
                }
                for item in compilation.resolutions
            ],
            canonical_query_hash=compilation.canonical_hash,
            unavailable_requested_values=list(
                compilation.unavailable_requested_values
            ),
        )
        return QdrantSearchExecution(
            result=result,
            debug_trace={
                **trace,
                "query": plan.query,
                "sort": plan.sort,
                "qdrant_checked": False,
                "retrieval_executed": False,
                "retrieval_skipped_reason": "unavailable_required_value",
                "semantic_state": "deterministic_not_found",
                "match_status": "not_found",
                "strict_total": 0,
                "total": 0,
                "returned_product_ids": [],
            },
        )

    def _semantic_clarification(
        self,
        plan: ProductQueryPlan,
        *,
        canonical_hash: str | None,
        clarification: dict[str, Any],
        trace: dict[str, Any],
    ) -> QdrantSearchExecution:
        result = ProductSearchResult(
            match_status="clarification_required",
            strict_total=0,
            total=0,
            applied_filters={},
            items=[],
            operation=plan.operation,
            canonical_query_hash=canonical_hash,
            clarification=clarification,
            unavailable_requested_values=list(
                trace.get("unavailable_requested_values", [])
            ),
            ranking_objectives=plan.ranking_objectives,
            plan_corrections=list(trace.get("plan_corrections", [])),
        )
        return QdrantSearchExecution(
            result=result,
            debug_trace={
                **trace,
                "query": plan.query,
                "sort": plan.sort,
                "qdrant_checked": bool(trace.get("semantic_entity_candidates")),
                "retrieval_executed": False,
                "semantic_state": "clarification_required",
                "filtered_count": 0,
                "filters": {},
                "exact_candidates": [],
                "exact_product_ids": [],
                "matching_exact_product_ids": [],
                "semantic_candidates": [],
                "sorted_candidates": [],
                "hydrated_product_ids": [],
                "strict_total": 0,
                "relaxed_fields": [],
                "alternative_stages": [],
                "exact_filter_conflict": False,
                "total": 0,
                "semantic_plan_invalid": clarification.get("reason")
                == "semantic_plan_invalid",
                "clarification": clarification,
                "match_status": "clarification_required",
                "returned_product_ids": [],
            },
        )

    @staticmethod
    def _resolution_trace(resolution: Any) -> dict[str, Any]:
        return {
            "entity_id": resolution.entity_id,
            "raw_text": resolution.raw_text,
            "status": resolution.status,
            "product_id": resolution.product_id,
            "reason": resolution.reason,
            "constraint_field": resolution.constraint_field,
            "constraint_value": resolution.constraint_value,
            "candidates": [
                {
                    "product_id": candidate.product_id,
                    "score": candidate.score,
                    "resolution": candidate.resolution,
                }
                for candidate in resolution.candidates
            ],
        }

    def _search_flat_with_trace(self, args: ProductSearchArguments) -> QdrantSearchExecution:
        original_args = args
        if args.semantic_plan_compiled:
            argument_corrections: list[dict[str, Any]] = []
            facet_mapping: list[dict[str, Any]] = []
        else:
            canonical = self.catalog.canonicalize_search_arguments(args)
            args = canonical.arguments
            argument_corrections = list(canonical.corrections)
            facet_mapping = canonical.facet_mapping
        if self.embeddings is None or self.store is None:
            trace = self._empty_trace(
                args,
                semantic_state="not_configured",
                original_args=original_args,
                argument_corrections=argument_corrections,
                facet_mapping=facet_mapping,
            )
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
            ordered_hits = self._sort_hits_for_args(exact_filtered, args)
            selected_hits = ordered_hits[: args.limit]
            products = self._hydrate(args, selected_hits, semantic_state="not_run_exact_match")
            result = self._result(
                args,
                products,
                match_status="exact_match",
                strict_total=len(exact_filtered),
                total=len(exact_filtered),
                argument_corrections=argument_corrections,
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
                    original_args=original_args,
                    argument_corrections=argument_corrections,
                    facet_mapping=facet_mapping,
                ),
            )

        if exact_unfiltered:
            exact_products = self._hydrate(
                args,
                exact_unfiltered[:1],
                semantic_state="not_run_exact_conflict",
            )
            requested_product = exact_products[0]
            conflicts = self._constraint_conflicts(args, requested_product)
            try:
                alternatives = self._search_alternatives(args)
            except Exception as exc:
                raise self._unavailable(args, "failed", exc, filtered_count=filtered_count) from exc
            selected_hits = [hit for hit, _ in alternatives.selected]
            alternative_products = self._hydrate(
                args,
                selected_hits,
                semantic_state=alternatives.semantic_state,
            )
            differences = {
                product["product_id"]: self._product_differences(args, product, relaxed_fields)
                for product, (_, relaxed_fields) in zip(
                    alternative_products,
                    alternatives.selected,
                    strict=True,
                )
            }
            visible_relaxed = list(
                dict.fromkeys(
                    field
                    for _, relaxed_fields in alternatives.selected
                    for field in relaxed_fields
                    if self._field_was_requested(args, field)
                )
            )
            result = self._result(
                args,
                alternative_products,
                match_status="exact_conflict",
                strict_total=0,
                total=len(alternative_products),
                relaxed_fields=visible_relaxed,
                differences=differences,
                requested_item=requested_product,
                constraint_conflicts=conflicts,
                argument_corrections=argument_corrections,
                ranking_components=alternatives.ranking_components,
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
                    hydrated_ids=[product["product_id"] for product in alternative_products],
                    semantic_state=alternatives.semantic_state,
                    exact_filter_conflict=True,
                    total=len(alternative_products),
                    match_status="exact_conflict",
                    strict_total=0,
                    relaxed_fields=visible_relaxed,
                    alternative_stages=alternatives.stages,
                    original_args=original_args,
                    argument_corrections=argument_corrections,
                    facet_mapping=facet_mapping,
                    constraint_conflicts=conflicts,
                    candidate_generation_lanes=(
                        alternatives.candidate_generation_lanes
                    ),
                    ranking_components=alternatives.ranking_components,
                ),
            )

        if not args.has_exact_identifier and filtered_count > 0:
            candidate_limit = (
                DEFAULT_DIRECTIONAL_SEMANTIC_CANDIDATES
                if args.semantic_ranking_objectives
                else self.sort_candidate_limit
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

            candidate_pool, candidate_generation_lanes = self._directional_candidate_pool(
                semantic_hits,
                args,
            )
            ordered_hits, ranking_components = self._rank_hits_with_trace(
                candidate_pool,
                args,
            )
            selected_hits = ordered_hits[: args.limit]
            products = self._hydrate(args, selected_hits, semantic_state="active")
            match_status: MatchStatus = "matching_products" if products else "not_found"
            result = self._result(
                args,
                products,
                match_status=match_status,
                strict_total=filtered_count,
                total=filtered_count if products else 0,
                argument_corrections=argument_corrections,
                ranking_components=ranking_components,
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
                    original_args=original_args,
                    argument_corrections=argument_corrections,
                    facet_mapping=facet_mapping,
                    candidate_generation_lanes=candidate_generation_lanes,
                    ranking_components=ranking_components,
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
            argument_corrections=argument_corrections,
            ranking_components=alternatives.ranking_components,
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
                original_args=original_args,
                argument_corrections=argument_corrections,
                facet_mapping=facet_mapping,
                candidate_generation_lanes=alternatives.candidate_generation_lanes,
                ranking_components=alternatives.ranking_components,
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
                candidate_generation_lanes=[],
                ranking_components={},
            )

        vector = self._embed_query(args)
        candidate_limit = (
            DEFAULT_DIRECTIONAL_SEMANTIC_CANDIDATES
            if self._effective_ranking_objectives(args)
            else self.sort_candidate_limit
            if args.sort != "relevance"
            else self.relevance_candidate_limit
        )
        target = min(args.limit, ALTERNATIVE_LIMIT)
        selected: list[tuple[VectorSearchHit, tuple[str, ...]]] = []
        selected_ids: set[str] = set()
        semantic_by_id: dict[str, VectorSearchHit] = {}
        candidate_generation_lanes: list[dict[str, Any]] = []
        ranking_components: dict[str, dict[str, Any]] = {}

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
            candidate_pool, lanes = self._directional_candidate_pool(eligible, stage_args)
            eligible_ids = {hit.product_id for hit in eligible}
            candidate_pool = [
                hit for hit in candidate_pool if hit.product_id in eligible_ids
            ]
            for lane in lanes:
                candidate_generation_lanes.append(
                    {"alternative_stage": index + 1, **lane}
                )
            ordered_stage, stage_components = self._rank_hits_with_trace(
                candidate_pool,
                stage_args,
            )
            ranking_components.update(stage_components)
            for hit in ordered_stage:
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
            candidate_generation_lanes=candidate_generation_lanes,
            ranking_components=ranking_components,
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
        if args.semantic_plan_compiled:
            return args.model_copy(update=updates)
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

    def _constraint_conflicts(
        self,
        args: ProductSearchArguments,
        product: dict[str, Any],
    ) -> list[str]:
        conflicts: list[str] = []
        if args.category_id and product.get("category", {}).get("id") != args.category_id:
            conflicts.append(f"Kateqoriya fərqlidir: {product.get('category', {}).get('name')}")
        if args.max_price is not None and float(product["price"]["sale"]) > args.max_price:
            conflicts.append(f"Qiymət büdcəni keçir: {product['price']['sale']} AZN")
        if args.min_price is not None and float(product["price"]["sale"]) < args.min_price:
            conflicts.append(f"Qiymət minimumdan aşağıdır: {product['price']['sale']} AZN")
        if args.in_stock is not None:
            actual_in_stock = product.get("stock", {}).get("status") == "in_stock"
            if actual_in_stock != args.in_stock:
                conflicts.append(
                    "Məhsul stokda deyil" if args.in_stock else "Məhsul stokdadır"
                )
        for field in (
            "brand",
            "model_family",
            "color_code",
            "storage_gb",
            "ram_gb",
            "btu",
            "screen_size_in",
            "connectivity",
            "active_noise_cancellation",
        ):
            if getattr(args, field, None) is None or self._product_matches_field(args, product, field):
                continue
            difference = self._format_difference(product, field)
            if difference and difference not in conflicts:
                conflicts.append(difference)
        for attribute_filter in args.attribute_filters:
            actual = product.get("attributes", {}).get(attribute_filter.field)
            if self._attribute_matches(attribute_filter, actual):
                continue
            difference = self._format_difference(product, attribute_filter.field)
            if difference and difference not in conflicts:
                conflicts.append(difference)
        if args.semantic_filter_expression is not None and not expression_matches(
            args.semantic_filter_expression, product
        ):
            semantic_conflict_count = len(conflicts)
            for predicate in iter_predicates(args.semantic_filter_expression):
                if predicate.strength != "hard":
                    continue
                single = predicate.model_copy()
                expression = args.semantic_filter_expression.model_construct(
                    kind="predicate",
                    predicate=single,
                )
                if expression_matches(expression, product):
                    continue
                conflict = f"{predicate.field} şərti ödənmir"
                if conflict not in conflicts:
                    conflicts.append(conflict)
            if len(conflicts) == semantic_conflict_count:
                conflicts.append("Verilən sərt məntiqi şərt ödənmir")
        return conflicts

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
                "catalog_checksum": self.catalog.manifest.get("checksums", {}).get("products_sha256"),
                "field_capability_checksum": self.catalog.field_capability_checksum,
                "field_capability_count": len(self.catalog.field_capabilities()),
                "categories": sorted(category_counts),
                "role": "full_product_hydration",
            },
            "semantic_qdrant": {
                "configured": self.semantic_enabled,
                "ready": self.semantic_enabled,
                "collection": getattr(self.store, "collection_name", None),
                "alternative_min_score": self.alternative_min_score,
                "entity_resolution_min_score": self.entity_resolution_min_score,
                "entity_resolution_margin": self.entity_resolution_margin,
                "directional_ranking_enabled": self.directional_ranking_enabled,
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

    def _sort_hits_for_args(
        self,
        hits: Sequence[VectorSearchHit],
        args: ProductSearchArguments,
    ) -> list[VectorSearchHit]:
        return self._rank_hits_with_trace(hits, args)[0]

    def _rank_hits_with_trace(
        self,
        hits: Sequence[VectorSearchHit],
        args: ProductSearchArguments,
    ) -> tuple[list[VectorSearchHit], dict[str, dict[str, Any]]]:
        ordered = self._sort_hits(hits, args.sort)
        objectives = self._effective_ranking_objectives(args)
        if not objectives and args.semantic_preference_expression is None:
            return ordered, {}

        semantic_rank = {hit.product_id: rank for rank, hit in enumerate(ordered)}
        objective_scores: dict[tuple[str, str], dict[str, float]] = {}
        for objective in objectives:
            values = {
                hit.product_id: self._numeric_payload_value(hit.payload, objective.field)
                for hit in hits
            }
            objective_scores[(objective.field, objective.direction)] = self._percentile_scores(
                values,
                direction=objective.direction,
            )

        preference_count = (
            args.semantic_preference_expression.predicate_count()
            if args.semantic_preference_expression is not None
            else 0
        )
        components: dict[str, dict[str, Any]] = {}
        for hit in hits:
            explicit_weighted = []
            inferred_weighted = []
            objective_details = []
            for objective in objectives:
                score = objective_scores[(objective.field, objective.direction)].get(
                    hit.product_id,
                    0.0,
                )
                weight = 2.0 if objective.priority == "primary" else 1.0
                target = inferred_weighted if objective.origin == "inferred" else explicit_weighted
                target.append((score, weight))
                objective_details.append(
                    {
                        "field": objective.field,
                        "direction": objective.direction,
                        "priority": objective.priority,
                        "origin": objective.origin,
                        "value": self._numeric_payload_value(hit.payload, objective.field),
                        "desirability": round(score, 6),
                    }
                )

            explicit_score = self._weighted_average(explicit_weighted)
            inferred_score = self._weighted_average(inferred_weighted)
            binary_score = (
                preference_score(args.semantic_preference_expression, hit.payload)
                / preference_count
                if preference_count
                else None
            )
            soft_values = [value for value in (inferred_score, binary_score) if value is not None]
            soft_score = sum(soft_values) / len(soft_values) if soft_values else None
            semantic_score = max(0.0, min(1.0, float(hit.score)))
            groups = [
                (explicit_score, 0.80),
                (semantic_score, 0.15),
                (soft_score, 0.05),
            ]
            active = [(score, weight) for score, weight in groups if score is not None]
            total_weight = sum(weight for _, weight in active) or 1.0
            total_score = sum(float(score) * weight for score, weight in active) / total_weight
            components[hit.product_id] = {
                "total_score": round(total_score, 6),
                "explicit_directional_score": (
                    round(explicit_score, 6) if explicit_score is not None else None
                ),
                "semantic_score": round(semantic_score, 6),
                "soft_score": round(soft_score, 6) if soft_score is not None else None,
                "objectives": objective_details,
            }

        ranked = sorted(
            hits,
            key=lambda hit: (
                -float(components[hit.product_id]["total_score"]),
                semantic_rank.get(hit.product_id, len(hits) + 1),
                hit.product_id,
            ),
        )
        if not self.directional_ranking_enabled and objectives:
            return ordered, {
                product_id: {**value, "shadow_only": True}
                for product_id, value in components.items()
            }
        return ranked, components

    @staticmethod
    def _effective_ranking_objectives(
        args: ProductSearchArguments,
    ) -> list[RankingObjective]:
        objectives = list(args.semantic_ranking_objectives)
        sort_objectives = {
            "price_asc": ("sale_price", "minimize"),
            "price_desc": ("sale_price", "maximize"),
            "rating_desc": ("rating", "maximize"),
        }
        sort_goal = sort_objectives.get(args.sort)
        if sort_goal and not any(
            item.field in {sort_goal[0], "price" if sort_goal[0] == "sale_price" else sort_goal[0]}
            for item in objectives
        ):
            objectives.append(
                RankingObjective(
                    field=sort_goal[0],
                    direction=sort_goal[1],
                    priority="primary",
                    origin="explicit",
                    evidence_text=args.query,
                )
            )
        return objectives

    @staticmethod
    def _numeric_payload_value(payload: dict[str, Any], field: str) -> float | None:
        key = "sale_price" if field == "price" else field
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _percentile_scores(
        values: dict[str, float | None],
        *,
        direction: str,
    ) -> dict[str, float]:
        present = sorted({value for value in values.values() if value is not None})
        if not present:
            return {product_id: 0.0 for product_id in values}
        if len(present) == 1:
            base = {present[0]: 1.0}
        else:
            base = {
                value: (index + 1) / len(present)
                for index, value in enumerate(present)
            }
        return {
            product_id: (
                0.0
                if value is None
                else base[value]
                if direction == "maximize"
                else (len(present) + 1) / len(present) - base[value]
            )
            for product_id, value in values.items()
        }

    @staticmethod
    def _weighted_average(values: list[tuple[float, float]]) -> float | None:
        if not values:
            return None
        total = sum(weight for _, weight in values)
        return sum(score * weight for score, weight in values) / total

    def _directional_candidate_pool(
        self,
        semantic_hits: Sequence[VectorSearchHit],
        args: ProductSearchArguments,
    ) -> tuple[list[VectorSearchHit], list[dict[str, Any]]]:
        objectives = self._effective_ranking_objectives(args)
        lanes: list[dict[str, Any]] = [
            {
                "kind": "semantic",
                "limit": DEFAULT_DIRECTIONAL_SEMANTIC_CANDIDATES,
                "candidate_count": len(semantic_hits),
                "product_ids": [hit.product_id for hit in semantic_hits],
            }
        ]
        by_id = {hit.product_id: hit for hit in semantic_hits}
        if not objectives or not self.directional_ranking_enabled or self.store is None:
            if objectives and not self.directional_ranking_enabled:
                lanes.append({"kind": "directional", "mode": "shadow", "queried": False})
            return list(by_id.values()), lanes

        ordered_search = getattr(self.store, "ordered_candidates", None)
        if not callable(ordered_search):
            lanes.append(
                {
                    "kind": "directional",
                    "mode": "active",
                    "queried": False,
                    "reason": "store_capability_unavailable",
                }
            )
            return list(by_id.values()), lanes

        seen_objectives: set[tuple[str, str]] = set()
        for objective in objectives:
            key = (objective.field, objective.direction)
            if key in seen_objectives:
                continue
            seen_objectives.add(key)
            ordered_hits = ordered_search(
                args,
                field=objective.field,
                direction=objective.direction,
                candidate_limit=DEFAULT_DIRECTIONAL_FIELD_CANDIDATES,
            )
            lanes.append(
                {
                    "kind": "field_order",
                    "field": objective.field,
                    "direction": objective.direction,
                    "limit": DEFAULT_DIRECTIONAL_FIELD_CANDIDATES,
                    "candidate_count": len(ordered_hits),
                    "product_ids": [hit.product_id for hit in ordered_hits],
                }
            )
            for hit in ordered_hits:
                by_id.setdefault(hit.product_id, hit)
        return list(by_id.values()), lanes

    def _result(
        self,
        args: ProductSearchArguments,
        products: Sequence[dict[str, Any]],
        *,
        match_status: MatchStatus,
        strict_total: int,
        total: int,
        relaxed_fields: Sequence[str] = (),
        differences: dict[str, list[str]] | None = None,
        requested_item: dict[str, Any] | None = None,
        constraint_conflicts: Sequence[str] = (),
        argument_corrections: Sequence[dict[str, Any]] = (),
        ranking_components: dict[str, dict[str, Any]] | None = None,
    ) -> ProductSearchResult:
        item_differences = differences or {}
        item_ranking = ranking_components or {}
        items = []
        for product in products:
            item = ProductCatalog.to_result(product)
            product_differences = item_differences.get(product["product_id"], [])
            ranking_reasons = [
                self._ranking_reason(objective)
                for objective in item_ranking.get(product["product_id"], {}).get(
                    "objectives",
                    [],
                )
                if objective.get("value") is not None
            ]
            if product_differences or ranking_reasons:
                item = item.model_copy(
                    update={
                        "differences": product_differences,
                        "ranking_reasons": ranking_reasons,
                    }
                )
            items.append(item)
        requested_result = ProductCatalog.to_result(requested_item) if requested_item else None
        display_product_ids = [item.product_id for item in items[:ALTERNATIVE_LIMIT]]
        recommended_product_id = display_product_ids[0] if display_product_ids else None
        if recommended_product_id is None and requested_result is not None:
            recommended_product_id = requested_result.product_id
        return ProductSearchResult(
            match_status=match_status,
            requested_label=self._requested_label(args),
            strict_total=strict_total,
            total=total,
            applied_filters=ProductCatalog.applied_filters(args),
            relaxed_fields=list(relaxed_fields),
            items=items,
            requested_item=requested_result,
            constraint_conflicts=list(constraint_conflicts),
            argument_corrections=list(argument_corrections),
            recommended_product_id=recommended_product_id,
            display_product_ids=display_product_ids,
            ranking_applied=bool(item_ranking) and self.directional_ranking_enabled,
            ranking_objectives=self._effective_ranking_objectives(args),
        )

    @staticmethod
    def _ranking_reason(objective: dict[str, Any]) -> str:
        direction = (
            "daha yüksək dəyər üstün tutuldu"
            if objective.get("direction") == "maximize"
            else "daha aşağı dəyər üstün tutuldu"
        )
        return f"{objective.get('field')}: {objective.get('value')} ({direction})"

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
        original_args: ProductSearchArguments | None = None,
        argument_corrections: Sequence[dict[str, Any]] = (),
        facet_mapping: Sequence[dict[str, Any]] = (),
        constraint_conflicts: Sequence[str] = (),
        candidate_generation_lanes: Sequence[dict[str, Any]] = (),
        ranking_components: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_ids = set(hydrated_ids)
        return {
            "mode": "qdrant_only_v2",
            "query": args.query,
            "original_arguments": (original_args or args).model_dump(mode="json"),
            "canonical_arguments": args.model_dump(mode="json"),
            "argument_corrections": list(argument_corrections),
            "facet_mapping": list(facet_mapping),
            "filters": ProductCatalog.applied_filters(args),
            "sort": args.sort,
            "ranking_mode": (
                "active"
                if self.directional_ranking_enabled
                and self._effective_ranking_objectives(args)
                else "shadow"
                if self._effective_ranking_objectives(args)
                else "none"
            ),
            "ranking_objectives": [
                objective.model_dump(mode="json")
                for objective in self._effective_ranking_objectives(args)
            ],
            "candidate_generation_lanes": list(candidate_generation_lanes),
            "ranking_components": ranking_components or {},
            "qdrant_checked": True,
            "retrieval_executed": True,
            "filtered_count": filtered_count,
            "exact_product_ids": [hit.product_id for hit in exact_unfiltered],
            "matching_exact_product_ids": [hit.product_id for hit in exact_filtered],
            "exact_filter_conflict": exact_filter_conflict,
            "constraint_conflicts": list(constraint_conflicts),
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
        original_args: ProductSearchArguments | None = None,
        argument_corrections: Sequence[dict[str, Any]] = (),
        facet_mapping: Sequence[dict[str, Any]] = (),
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
            original_args=original_args,
            argument_corrections=argument_corrections,
            facet_mapping=facet_mapping,
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
