from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.tools.document_search import DocumentSearchBackendError, DocumentSearchTool
from app.tools.product_search import ProductSearchBackendError, ProductSearchTool
from app.tools.schemas import (
    DocumentSearchArguments,
    ProductQueryPlan,
    ProductSearchArguments,
    RequiredFilterField,
)


class UnknownToolError(ValueError):
    pass


class ToolArgumentsError(ValueError):
    pass


@dataclass(frozen=True)
class ToolExecution:
    result: dict[str, Any]
    debug_trace: dict[str, Any] | None = None


class ToolRegistry:
    def __init__(
        self,
        product_search: ProductSearchTool,
        timeout_seconds: float,
        *,
        document_search: DocumentSearchTool | None = None,
    ) -> None:
        self.product_search = product_search
        self.document_search = document_search
        self.timeout_seconds = timeout_seconds

    def specs(self) -> list[dict[str, Any]]:
        schema = ProductQueryPlan.model_json_schema()
        schema.get("properties", {}).pop("context_product_ids", None)
        schema.get("properties", {}).pop("context_memory", None)
        catalog = getattr(getattr(self.product_search, "backend", None), "catalog", None)
        semantic_fields = getattr(catalog, "semantic_fields", ())
        if semantic_fields:
            definitions = schema.get("$defs", {})
            for definition_name in ("SemanticPredicate", "FactQuestion"):
                field_schema = (
                    definitions.get(definition_name, {})
                    .get("properties", {})
                    .get("field")
                )
                if isinstance(field_schema, dict):
                    field_schema["enum"] = list(semantic_fields)
            value_schema = (
                definitions.get("SemanticPredicate", {})
                .get("properties", {})
                .get("value")
            )
            if isinstance(value_schema, dict):
                facet_hints: list[str] = []
                hint_size = 0
                for field in semantic_fields:
                    values = tuple(
                        getattr(catalog, "facet_values", lambda _field: ())(field)
                    )
                    if not values or len(values) > 30:
                        continue
                    hint = f"{field}={json.dumps(values, ensure_ascii=False)}"
                    if hint_size + len(hint) > 4000:
                        break
                    facet_hints.append(hint)
                    hint_size += len(hint)
                if not facet_hints:
                    facet_hints = []
                existing = str(value_schema.get("description", "")).rstrip()
                if facet_hints:
                    value_schema["description"] = (
                        f"{existing}. For a field listed below, map the user's intended meaning to "
                        "one of that field's canonical catalog values; do not copy a translated surface "
                        f"form. Catalog values: {'; '.join(facet_hints)}"
                    )
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "product_search",
                    "description": (
                        "Create one complete semantic product query plan from the current user message "
                        "and relevant conversation context. Preserve selected and superseded entities, "
                        "intent, logical relationships, hard constraints, preferences, fact questions, "
                        "recommendation intent, and genuine ambiguity. Use recursive expression operators "
                        "only for their semantic meaning. evidence_text must be an exact span from the "
                        "current user message. Use selection_expression for relationships among product "
                        "entities and filter_expression for catalog-field constraints. Preserve the user's "
                        "facet granularity; never widen a named family or model into its parent brand or "
                        "category. A fact question and a filtering constraint are mutually exclusive. Emit "
                        "raw product mentions only; never invent catalog IDs. Entity choice expressions "
                        "must not be duplicated as filter predicates; filters contain only independent "
                        "constraints common to the selection branches. When one branch unambiguously "
                        "establishes the shared product type of fallback/OR branches, preserve that common "
                        "type as a canonical category_id constraint instead of widening another branch. "
                        "For every discover request whose product type is unambiguous from the message or "
                        "context, a canonical category_id predicate is required in the plan. A clear broad "
                        "brand/family/facet discovery is valid without category, budget, or extra "
                        "preferences and must not request clarification merely to narrow optional scope. "
                        "For exact lookup, do not add an inferred category predicate whose evidence_text "
                        "would require expanding or reconstructing the user's current words; entity "
                        "resolution already validates the product type. "
                        "The system message may contain verified session memory with opaque memory_id "
                        "values. Use memory_action=replace for an independent goal, merge for a contextual "
                        "revision, and preserve for a read-only continuation. Copy every used memory_id "
                        "into referenced_memory_ids and the relevant entity/predicate/fact memory_refs; "
                        "put explicitly discarded state in removed_memory_ids and add a matching "
                        "memory_removals entry whose evidence_text is the exact current-message span that "
                        "supports the removal. "
                        "A pending_intent memory item represents a previous unresolved or unavailable "
                        "entity; when continuing or replacing it, reference its memory_id on that inherited "
                        "entity and use the current message only for the new meaning. When its product type "
                        "is unambiguous, preserve that type as a canonical category predicate grounded by "
                        "the pending memory reference. "
                        "Never invent a memory ID. "
                        "Current-message meaning still needs the narrowest exact evidence_text span; "
                        "inherited catalog meaning additionally needs its verified memory_refs."
                        " Never invent helpful default constraints such as availability, budget, color, "
                        "or specifications. Each predicate evidence span must specifically support that "
                        "field, operator, and value rather than using the whole message as generic evidence."
                    ),
                    "parameters": schema,
                },
            }
        ]
        if self.document_search is not None:
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "document_search",
                        "description": (
                            "Yüklənmiş mağaza qaydaları və xidmət Markdown sənədlərində semantic "
                            "axtarış edir. Kredit, çatdırılma, zəmanət, geri qaytarma, quraşdırma "
                            "və digər mağaza siyasəti suallarında istifadə et. Məhsul qiyməti, stok və "
                            "texniki xüsusiyyət üçün bu tool-u deyil, product_search istifadə et."
                        ),
                        "parameters": DocumentSearchArguments.model_json_schema(),
                    },
                }
            )
        return specs

    async def execute(self, name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        execution = await self.execute_with_trace(name, raw_arguments)
        return execution.result

    async def execute_with_trace(
        self,
        name: str,
        raw_arguments: dict[str, Any],
    ) -> ToolExecution:
        if name == "document_search":
            return await self._execute_document_search(raw_arguments)
        if name != self.product_search.name:
            raise UnknownToolError(f"Naməlum tool: {name}")
        if "operation" in raw_arguments or "entities" in raw_arguments:
            try:
                semantic_arguments = ProductQueryPlan.model_validate(raw_arguments)
            except ValidationError as exc:
                return self._semantic_plan_invalid(raw_arguments, exc)
            try:
                semantic_execution = await asyncio.wait_for(
                    self.product_search.execute_with_trace(semantic_arguments),
                    timeout=self.timeout_seconds,
                )
            except ProductSearchBackendError as exc:
                return ToolExecution(
                    result={"status": "error", "code": exc.code, "message": exc.message},
                    debug_trace=exc.debug_trace,
                )
            return ToolExecution(
                result=semantic_execution.result.model_dump(mode="json"),
                debug_trace=semantic_execution.debug_trace,
            )
        normalized_arguments = dict(raw_arguments)
        input_corrections: list[dict[str, Any]] = []
        required_fields = normalized_arguments.get("required_filter_fields")
        if isinstance(required_fields, list):
            adapter = TypeAdapter(RequiredFilterField)
            valid_required_fields = []
            for field in required_fields:
                try:
                    valid_required_fields.append(adapter.validate_python(field))
                except ValidationError:
                    input_corrections.append(
                        {
                            "field": "required_filter_fields",
                            "action": "removed_non_relaxable_field",
                            "original": field,
                        }
                    )
            normalized_arguments["required_filter_fields"] = list(
                dict.fromkeys(valid_required_fields)
            )
        try:
            arguments = ProductSearchArguments.model_validate(normalized_arguments)
        except ValidationError as exc:
            raise ToolArgumentsError("product_search arqumentləri etibarsızdır") from exc
        try:
            execution = await asyncio.wait_for(
                self.product_search.execute_with_trace(arguments),
                timeout=self.timeout_seconds,
            )
        except ProductSearchBackendError as exc:
            debug_trace = exc.debug_trace
            if debug_trace is not None and input_corrections:
                debug_trace = {
                    **debug_trace,
                    "input_argument_corrections": input_corrections,
                    "argument_corrections": [
                        *input_corrections,
                        *debug_trace.get("argument_corrections", []),
                    ],
                }
            return ToolExecution(
                result={
                    "status": "error",
                    "code": exc.code,
                    "message": exc.message,
                    "argument_corrections": input_corrections,
                },
                debug_trace=debug_trace,
            )
        result = execution.result
        if input_corrections:
            result = result.model_copy(
                update={
                    "argument_corrections": [
                        *input_corrections,
                        *result.argument_corrections,
                    ]
                }
            )
        debug_trace = execution.debug_trace
        if debug_trace is not None and input_corrections:
            debug_trace = {
                **debug_trace,
                "input_argument_corrections": input_corrections,
                "argument_corrections": [
                    *input_corrections,
                    *debug_trace.get("argument_corrections", []),
                ],
            }
        return ToolExecution(
            result=result.model_dump(mode="json"),
            debug_trace=debug_trace,
        )

    @staticmethod
    def _semantic_plan_invalid(
        raw_arguments: dict[str, Any],
        exc: ValidationError,
    ) -> ToolExecution:
        clarification = {
            "reason": "semantic_plan_invalid",
            "question": "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?",
        }
        result = {
            "status": "success",
            "match_status": "clarification_required",
            "requested_label": None,
            "strict_total": 0,
            "total": 0,
            "applied_filters": {},
            "relaxed_fields": [],
            "items": [],
            "requested_item": None,
            "constraint_conflicts": [],
            "argument_corrections": [],
            "recommended_product_id": None,
            "display_product_ids": [],
            "operation": raw_arguments.get("operation", "discover"),
            "resolved_entities": [],
            "entity_results": [],
            "canonical_query_hash": None,
            "clarification": clarification,
            "unavailable_requested_values": [],
        }
        return ToolExecution(
            result=result,
            debug_trace={
                "mode": "semantic_plan_v1",
                "semantic_plan_invalid": True,
                "raw_semantic_plan": raw_arguments,
                # Pydantic may place a live ValueError object in ctx; debug traces are JSONB.
                "validation_errors": exc.errors(
                    include_url=False,
                    include_context=False,
                ),
                "clarification": clarification,
                "match_status": "clarification_required",
                "retrieval_executed": False,
            },
        )

    async def _execute_document_search(
        self,
        raw_arguments: dict[str, Any],
    ) -> ToolExecution:
        if self.document_search is None:
            raise UnknownToolError("Naməlum tool: document_search")
        try:
            arguments = DocumentSearchArguments.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolArgumentsError("document_search arqumentləri etibarsızdır") from exc
        try:
            execution = await asyncio.wait_for(
                self.document_search.execute_with_trace(arguments),
                timeout=self.timeout_seconds,
            )
        except DocumentSearchBackendError as exc:
            return ToolExecution(
                result={"status": "error", "code": exc.code, "message": exc.message},
                debug_trace=exc.debug_trace,
            )
        return ToolExecution(
            result=execution.result.model_dump(mode="json"),
            debug_trace=execution.debug_trace,
        )

    def debug_source_state(self) -> dict[str, Any]:
        source_state = self.product_search.debug_source_state()
        if self.document_search is not None:
            source_state["documents"] = self.document_search.debug_source_state()
        return source_state
