from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from app.tools.catalog import CatalogBackend, EntityResolution, normalize_text
from app.tools.schemas import (
    BOOLEAN_ATTRIBUTE_FIELDS,
    NUMERIC_ATTRIBUTE_FIELDS,
    AttributeFilter,
    FactQuestion,
    ProductQueryPlan,
    ProductSearchArguments,
    SemanticExpression,
    SemanticPredicate,
)

SEMANTIC_PLAN_VERSION = "product-query-plan-v3-typed-memory"
PRICE_FIELDS = frozenset({"price", "sale_price"})
STOCK_FIELDS = frozenset({"stock_status", "in_stock"})
TOP_LEVEL_FIELDS = frozenset(
    {
        "product_id",
        "sku",
        "model",
        "name",
        "category_id",
        "brand",
        "model_family",
        "color_code",
        "warranty_months",
        "rating",
    }
)


@dataclass(frozen=True)
class SemanticPlanCompilation:
    plan: ProductQueryPlan
    canonical_plan: dict[str, Any]
    canonical_hash: str
    evidence_validation: list[dict[str, Any]]
    resolutions: tuple[EntityResolution, ...]
    arguments: tuple[ProductSearchArguments, ...]
    clarification: dict[str, Any] | None
    facet_mapping: tuple[dict[str, Any], ...] = ()
    unavailable_requested_values: tuple[dict[str, Any], ...] = ()
    deterministic_empty: bool = False


class SemanticPlanValidationError(ValueError):
    pass


def compile_semantic_plan(
    plan: ProductQueryPlan,
    catalog: CatalogBackend,
    *,
    resolution_overrides: dict[str, str] | None = None,
) -> SemanticPlanCompilation:
    overrides = resolution_overrides or {}
    source_plan = plan
    plan, facet_mapping, grounding_issue, unavailable_values = _ground_plan_predicates(
        plan,
        catalog,
    )
    plan = _canonicalize_selected_entities(plan)
    active_entities = [entity for entity in plan.entities if entity.state == "selected"]
    resolutions = tuple(
        _resolve_superseding_entity(
            entity,
            plan,
            catalog,
            EntityResolution(
                entity.entity_id,
                entity.raw_text,
                "resolved",
                overrides[entity.entity_id],
                (),
                "semantic_unique_candidate",
            )
            if entity.entity_id in overrides
            and catalog.product_by_id(overrides[entity.entity_id]) is not None
            else _resolve_entity(entity, catalog),
        )
        for entity in active_entities
    )
    evidence_validation = validate_evidence(source_plan, resolutions=resolutions)
    invalid_evidence = [item for item in evidence_validation if not item["valid"]]
    memory_issue = validate_memory_references(plan, resolutions=resolutions)
    canonical = plan.model_dump(mode="json", exclude_none=True)
    canonical_hash = hashlib.sha256(
        json.dumps(
            {
                "plan_version": SEMANTIC_PLAN_VERSION,
                "catalog_schema_version": catalog.manifest.get("dataset_version"),
                "catalog_checksum": catalog.manifest.get("checksums", {}).get("schema_sha256"),
                "plan": canonical,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    invalid_fields = sorted(
        {
            predicate.field
            for expression in _plan_expressions(plan)
            for predicate in iter_predicates(expression)
            if not catalog.supports_field(predicate.field)
        }
        | {
            question.field
            for question in plan.fact_questions
            if not catalog.supports_field(question.field)
        }
    )
    if invalid_evidence or invalid_fields or memory_issue:
        reasons = []
        if invalid_evidence:
            reasons.append("evidence_not_grounded")
        if invalid_fields:
            reasons.append("unsupported_catalog_field")
        if memory_issue:
            reasons.append("invalid_memory_reference")
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            (),
            (),
            {
                "reason": ",".join(reasons),
                "question": "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?",
                "invalid_fields": invalid_fields,
                "memory_issue": memory_issue,
            },
            tuple(facet_mapping),
            tuple(unavailable_values),
        )
    if grounding_issue is not None:
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            (),
            (),
            grounding_issue,
            tuple(facet_mapping),
            tuple(unavailable_values),
        )
    _validate_predicate_types(plan)
    if plan.needs_clarification:
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            (),
            (),
            {
                "reason": "model_requested_clarification",
                "question": plan.clarification_question,
            },
            tuple(facet_mapping),
            tuple(unavailable_values),
        )

    if _has_blocking_unavailable_filter(plan, unavailable_values):
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            resolutions,
            (),
            None,
            tuple(facet_mapping),
            tuple(unavailable_values),
            True,
        )

    invalid_context_entities = [
        entity.entity_id
        for entity in active_entities
        if entity.context_product_id
        and (
            entity.context_product_id not in set(plan.context_product_ids)
            or catalog.product_by_id(entity.context_product_id) is None
        )
    ]
    if invalid_context_entities:
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            (),
            (),
            {
                "reason": "invalid_context_reference",
                "question": "Hansı əvvəlki məhsulu nəzərdə tutursunuz?",
                "entity_ids": invalid_context_entities,
            },
            tuple(facet_mapping),
            tuple(unavailable_values),
        )
    ambiguous = [resolution for resolution in resolutions if resolution.status == "ambiguous"]
    if ambiguous:
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            resolutions,
            (),
            {
                "reason": "ambiguous_entity",
                "question": "Hansı dəqiq məhsulu nəzərdə tutursunuz?",
                "entity_ids": [item.entity_id for item in ambiguous],
            },
            tuple(facet_mapping),
            tuple(unavailable_values),
        )

    selection_filter, unsupported_selection = _selection_filter_expression(
        plan.selection_expression
    )
    if unsupported_selection:
        return SemanticPlanCompilation(
            plan,
            canonical,
            canonical_hash,
            evidence_validation,
            resolutions,
            (),
            {
                "reason": "mixed_selection_expression",
                "question": "Məhsul seçimlərini və şərtləri ayrıca dəqiqləşdirə bilərsiniz?",
            },
            tuple(facet_mapping),
            tuple(unavailable_values),
        )
    effective_filter = _conjoin_expressions(plan.filter_expression, selection_filter)
    resolved_selection_filter = (
        None
        if plan.operation == "compare"
        else _resolved_selection_filter(
            plan.selection_expression,
            active_entities,
            resolutions,
        )
    )
    effective_filter = _conjoin_expressions(effective_filter, resolved_selection_filter)
    branches = _fallback_branches(effective_filter)
    arguments: list[ProductSearchArguments] = []
    if plan.operation == "compare":
        for entity, resolution in zip(active_entities, resolutions, strict=True):
            arguments.append(
                _arguments_for_plan(
                    plan,
                    filter_expression=branches[0],
                    entity=entity,
                    resolution=resolution,
                )
            )
    elif resolved_selection_filter is not None or _entities_represented_by_filter(
        plan,
        active_entities,
    ):
        for branch in branches:
            arguments.append(
                _arguments_for_plan(
                    plan,
                    filter_expression=branch,
                    entity=None,
                    resolution=None,
                )
            )
    else:
        selection_branches = _selection_branches(
            plan.selection_expression,
            active_entities,
            resolutions,
        )
        if len(selection_branches) > 1:
            for index, (entity, resolution) in enumerate(selection_branches):
                arguments.append(
                    _arguments_for_plan(
                        plan,
                        filter_expression=branches[min(index, len(branches) - 1)],
                        entity=entity,
                        resolution=resolution,
                    )
                )
        else:
            entity, resolution = selection_branches[0]
            for branch in branches:
                arguments.append(
                    _arguments_for_plan(
                        plan,
                        filter_expression=branch,
                        entity=entity,
                        resolution=resolution,
                    )
                )
    return SemanticPlanCompilation(
        plan,
        canonical,
        canonical_hash,
        evidence_validation,
        resolutions,
        tuple(arguments),
        None,
        tuple(facet_mapping),
        tuple(unavailable_values),
    )


def validate_evidence(
    plan: ProductQueryPlan,
    *,
    resolutions: tuple[EntityResolution, ...] = (),
) -> list[dict[str, Any]]:
    query = normalize_text(plan.query)
    memory_items = _context_memory_items(plan)
    known_memory_ids = set(memory_items)
    resolution_by_id = {item.entity_id: item for item in resolutions}
    entity_reference_details: dict[str, dict[str, Any]] = {}
    for entity in plan.entities:
        valid, legacy_refs, anchors = _entity_memory_reference_match(
            entity,
            resolution_by_id.get(entity.entity_id),
            memory_items,
        )
        entity_reference_details[f"entity:{entity.entity_id}"] = {
            "typed_memory_valid": valid,
            "legacy_predicate_entity_refs": legacy_refs,
            "canonical_memory_anchors": anchors,
        }
    values: list[tuple[str, str, list[str]]] = []
    values.extend(
        (f"entity:{entity.entity_id}", entity.evidence_text, entity.memory_refs)
        for entity in plan.entities
    )
    for index, expression in enumerate(_plan_expressions(plan)):
        values.extend(
            (
                f"predicate:{index}:{offset}",
                predicate.evidence_text,
                predicate.memory_refs,
            )
            for offset, predicate in enumerate(iter_predicates(expression))
        )
    values.extend(
        (f"fact_question:{index}", question.evidence_text, question.memory_refs)
        for index, question in enumerate(plan.fact_questions)
    )
    values.extend(
        (f"memory_removal:{index}", removal.evidence_text, [])
        for index, removal in enumerate(plan.memory_removals)
    )
    result = []
    for source, evidence, memory_refs in values:
        current_message_valid = (
            bool(normalize_text(evidence)) and normalize_text(evidence) in query
        )
        details = entity_reference_details.get(source, {})
        memory_valid = (
            bool(details.get("typed_memory_valid"))
            if source.startswith("entity:")
            else bool(memory_refs)
            and all(memory_id in known_memory_ids for memory_id in memory_refs)
        )
        result.append({
            "source": source,
            "evidence_text": evidence,
            "memory_refs": list(memory_refs),
            "evidence_source": (
                "current_message"
                if current_message_valid
                else "session_memory" if memory_valid else "invalid"
            ),
            "valid": current_message_valid or memory_valid,
            "memory_reference_kinds": [
                memory_items.get(memory_id, {}).get("kind", "unknown")
                for memory_id in memory_refs
            ],
            **details,
        })
    return result


def validate_memory_references(
    plan: ProductQueryPlan,
    *,
    resolutions: tuple[EntityResolution, ...] = (),
) -> dict[str, Any] | None:
    items = _context_memory_items(plan)
    known_ids = set(items)
    nested_refs: set[str] = set()
    for entity in plan.entities:
        nested_refs.update(entity.memory_refs)
    for expression in _plan_expressions(plan):
        for predicate in iter_predicates(expression):
            nested_refs.update(predicate.memory_refs)
    for question in plan.fact_questions:
        nested_refs.update(question.memory_refs)
    declared_refs = set(plan.referenced_memory_ids)
    removed_refs = set(plan.removed_memory_ids)
    invalid_ids = sorted((declared_refs | removed_refs | nested_refs) - known_ids)
    undeclared_nested = sorted(nested_refs - (declared_refs | removed_refs))
    grounded_removals = {item.memory_id for item in plan.memory_removals}
    unanchored_removals = sorted(removed_refs - grounded_removals)
    downgraded: list[str] = []
    mutated_inherited_predicates: list[str] = []
    invalid_inherited_entities: list[str] = []
    legacy_predicate_entity_refs: list[str] = []
    query = normalize_text(plan.query)
    resolution_by_id = {item.entity_id: item for item in resolutions}
    for entity in plan.entities:
        normalized_evidence = normalize_text(entity.evidence_text)
        if normalized_evidence and normalized_evidence in query:
            continue
        valid_reference, legacy_refs, _ = _entity_memory_reference_match(
            entity,
            resolution_by_id.get(entity.entity_id),
            items,
        )
        legacy_predicate_entity_refs.extend(legacy_refs)
        if not valid_reference:
            invalid_inherited_entities.append(entity.entity_id)
    for expression in _plan_expressions(plan):
        for predicate in iter_predicates(expression):
            normalized_evidence = normalize_text(predicate.evidence_text)
            current_evidence = bool(normalized_evidence) and normalized_evidence in query
            for memory_id in predicate.memory_refs:
                item = items.get(memory_id, {})
                if (
                    item.get("kind") == "predicate"
                    and item.get("strength") == "hard"
                    and predicate.strength != "hard"
                ):
                    downgraded.append(memory_id)
                if (
                    not current_evidence
                    and item.get("kind") == "predicate"
                    and not _predicate_matches_memory(predicate, item)
                ):
                    mutated_inherited_predicates.append(memory_id)
    invalid_inherited_fact_questions = []
    for question in plan.fact_questions:
        normalized = normalize_text(question.evidence_text)
        if normalized and normalized in query:
            continue
        referenced_questions = [
            items[memory_id]
            for memory_id in question.memory_refs
            if items.get(memory_id, {}).get("kind") == "fact_question"
        ]
        if not any(_fact_question_matches_memory(question, item) for item in referenced_questions):
            invalid_inherited_fact_questions.append(question.field)
    action_without_memory = (
        plan.memory_action in {"merge", "preserve"}
        and not known_ids
        and bool(declared_refs or nested_refs or removed_refs)
    )
    if (
        not invalid_ids
        and not undeclared_nested
        and not unanchored_removals
        and not downgraded
        and not mutated_inherited_predicates
        and not invalid_inherited_entities
        and not invalid_inherited_fact_questions
        and not action_without_memory
    ):
        return None
    return {
        "invalid_ids": invalid_ids,
        "undeclared_nested_refs": undeclared_nested,
        "unanchored_removals": unanchored_removals,
        "hard_constraint_downgrades": sorted(set(downgraded)),
        "mutated_inherited_predicates": sorted(set(mutated_inherited_predicates)),
        "invalid_inherited_entities": sorted(set(invalid_inherited_entities)),
        "invalid_inherited_fact_questions": sorted(set(invalid_inherited_fact_questions)),
        "legacy_predicate_entity_refs": sorted(set(legacy_predicate_entity_refs)),
        "memory_action": plan.memory_action,
    }


def _entity_memory_reference_match(
    entity: Any,
    resolution: EntityResolution | None,
    items: dict[str, dict[str, Any]],
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Validate inherited entity references against their canonical catalog meaning.

    V3 entity anchors are preferred. V2 predicate-backed entities are accepted only when
    local catalog resolution proves an exact field/value match.
    """

    legacy_refs: list[str] = []
    anchors: list[dict[str, Any]] = []
    matched = False
    for memory_id in entity.memory_refs:
        item = items.get(memory_id, {})
        kind = item.get("kind")
        if kind == "entity":
            anchor_kind = item.get("anchor_kind") or "product"
            if (
                anchor_kind == "product"
                and resolution is not None
                and resolution.product_id
                and resolution.product_id == item.get("product_id")
            ):
                matched = True
            elif (
                anchor_kind == "facet"
                and resolution is not None
                and resolution.status == "constraint_entity"
                and resolution.constraint_field == item.get("facet_field")
                and _canonical_value_equal(
                    resolution.constraint_value,
                    item.get("facet_value"),
                )
            ):
                matched = True
            anchors.append({
                "memory_id": memory_id,
                "anchor_kind": anchor_kind,
                "product_id": item.get("product_id"),
                "facet_field": item.get("facet_field"),
                "facet_value": item.get("facet_value"),
            })
        elif (
            kind == "predicate"
            and resolution is not None
            and resolution.status == "constraint_entity"
            and resolution.constraint_field == item.get("field")
            and _canonical_value_equal(resolution.constraint_value, item.get("value"))
        ):
            matched = True
            legacy_refs.append(memory_id)
            anchors.append({
                "memory_id": memory_id,
                "anchor_kind": "legacy_predicate",
                "facet_field": item.get("field"),
                "facet_value": item.get("value"),
            })
        elif kind == "pending_intent":
            pending_values = {
                normalize_text(str(raw_entity))
                for raw_entity in item.get("raw_entities", [])
            }
            if normalize_text(entity.raw_text) in pending_values:
                matched = True
    return matched, legacy_refs, anchors


def _canonical_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return normalize_text(left) == normalize_text(right)
    return left == right


def _has_blocking_unavailable_filter(
    plan: ProductQueryPlan,
    unavailable_values: list[dict[str, Any]],
) -> bool:
    unavailable = {
        (str(item.get("field")), json.dumps(item.get("value"), sort_keys=True))
        for item in unavailable_values
        if item.get("reason") == "unmapped_required_value"
    }
    if not unavailable or plan.filter_expression is None:
        return False
    return _expression_is_deterministically_empty(plan.filter_expression, unavailable)


def _expression_is_deterministically_empty(
    expression: SemanticExpression,
    unavailable: set[tuple[str, str]],
) -> bool:
    if expression.kind == "predicate" and expression.predicate is not None:
        return (
            expression.predicate.field,
            json.dumps(expression.predicate.value, sort_keys=True),
        ) in unavailable
    if expression.kind == "all_of":
        return any(
            _expression_is_deterministically_empty(child, unavailable)
            for child in expression.expressions or []
        )
    if expression.kind == "any_of":
        children = expression.expressions or []
        return bool(children) and all(
            _expression_is_deterministically_empty(child, unavailable)
            for child in children
        )
    if expression.kind == "fallback":
        return bool(expression.primary and expression.secondary) and all(
            _expression_is_deterministically_empty(child, unavailable)
            for child in (expression.primary, expression.secondary)
        )
    # Negation of an unavailable equality is non-blocking; preferences never filter.
    return False


def _predicate_matches_memory(
    predicate: SemanticPredicate,
    memory_item: dict[str, Any],
) -> bool:
    return all(
        (
            predicate.field == memory_item.get("field"),
            predicate.operator == memory_item.get("operator"),
            predicate.value == memory_item.get("value"),
            predicate.strength == memory_item.get("strength"),
            predicate.unit == memory_item.get("unit"),
        )
    )


def _fact_question_matches_memory(
    question: Any,
    memory_item: dict[str, Any],
) -> bool:
    return all(
        (
            question.field == memory_item.get("field"),
            question.operator == memory_item.get("operator"),
            question.value == memory_item.get("value"),
            question.unit == memory_item.get("unit"),
        )
    )


def _context_memory_items(plan: ProductQueryPlan) -> dict[str, dict[str, Any]]:
    memory = plan.context_memory if isinstance(plan.context_memory, dict) else {}
    product_state = (
        memory.get("confirmed_state")
        if isinstance(memory.get("confirmed_state"), dict)
        else memory.get("product") if isinstance(memory.get("product"), dict) else {}
    )
    result: dict[str, dict[str, Any]] = {}
    _add_context_state_items(result, product_state, scope="confirmed")
    pending = (
        memory.get("pending_intent")
        if isinstance(memory.get("pending_intent"), dict)
        else memory.get("pending_clarification")
    )
    if isinstance(pending, dict) and pending.get("memory_id"):
        result[str(pending["memory_id"])] = {"kind": "pending_intent", **pending}
        pending_state = pending.get("state")
        if isinstance(pending_state, dict):
            _add_context_state_items(result, pending_state, scope="pending")
    return result


def _add_context_state_items(
    result: dict[str, dict[str, Any]],
    state: dict[str, Any],
    *,
    scope: str,
) -> None:
    for item in state.get("entities", []):
        if isinstance(item, dict) and item.get("memory_id"):
            result[str(item["memory_id"])] = {"kind": "entity", "scope": scope, **item}
    for field in ("hard_constraints", "preferences"):
        for item in state.get(field, []):
            if isinstance(item, dict) and item.get("memory_id"):
                result[str(item["memory_id"])] = {
                    "kind": "predicate",
                    "scope": scope,
                    **item,
                }
    for item in state.get("fact_questions", []):
        if isinstance(item, dict) and item.get("memory_id"):
            result[str(item["memory_id"])] = {
                "kind": "fact_question",
                "scope": scope,
                **item,
            }


def iter_predicates(expression: SemanticExpression) -> list[SemanticPredicate]:
    if expression.kind == "predicate":
        return [expression.predicate] if expression.predicate else []
    predicates: list[SemanticPredicate] = []
    for child in expression.expressions or []:
        predicates.extend(iter_predicates(child))
    for child in (expression.expression, expression.primary, expression.secondary):
        if child is not None:
            predicates.extend(iter_predicates(child))
    return predicates


def expression_matches(expression: SemanticExpression | None, product: dict[str, Any]) -> bool:
    if expression is None:
        return True
    if expression.kind == "predicate":
        return _predicate_matches(expression.predicate, product) if expression.predicate else False
    if expression.kind == "all_of":
        return all(expression_matches(child, product) for child in expression.expressions or [])
    if expression.kind == "any_of":
        return any(expression_matches(child, product) for child in expression.expressions or [])
    if expression.kind == "not":
        return not expression_matches(expression.expression, product)
    if expression.kind == "fallback":
        return expression_matches(expression.primary, product) or expression_matches(
            expression.secondary, product
        )
    if expression.kind == "prefer":
        return expression_matches(expression.expression, product)
    return True


def preference_score(expression: SemanticExpression | None, product: dict[str, Any]) -> int:
    if expression is None:
        return 0
    if expression.kind == "predicate":
        return int(bool(expression.predicate) and _predicate_matches(expression.predicate, product))
    if expression.kind in {"all_of", "any_of"}:
        return sum(preference_score(child, product) for child in expression.expressions or [])
    if expression.kind in {"not", "prefer"}:
        return int(expression_matches(expression, product))
    if expression.kind == "fallback":
        return max(
            preference_score(expression.primary, product),
            preference_score(expression.secondary, product),
        )
    return 0


def _arguments_for_plan(
    plan: ProductQueryPlan,
    *,
    filter_expression: SemanticExpression | None,
    entity: Any,
    resolution: EntityResolution | None,
) -> ProductSearchArguments:
    if (
        resolution is not None
        and resolution.status == "constraint_entity"
        and resolution.constraint_field
        and resolution.constraint_value is not None
    ):
        entity_constraint = SemanticExpression(
            kind="predicate",
            predicate=SemanticPredicate(
                field=resolution.constraint_field,
                operator="eq",
                value=resolution.constraint_value,
                strength="hard",
                evidence_text=entity.evidence_text,
                memory_refs=entity.memory_refs,
            ),
        )
        filter_expression = _conjoin_expressions(filter_expression, entity_constraint)
    updates: dict[str, Any] = {
        "query": plan.query,
        "search_intent": "lookup" if plan.operation == "lookup" else "discover",
        "requested_fields": [_requested_field(question) for question in plan.fact_questions],
        "sort": plan.sort,
        "limit": min(plan.limit, 3),
        "semantic_filter_expression": filter_expression,
        "semantic_preference_expression": plan.preference_expression,
        "semantic_plan_compiled": True,
    }
    updates["requested_fields"] = list(dict.fromkeys(updates["requested_fields"]))
    if resolution and resolution.product_id:
        updates["product_id"] = resolution.product_id
        updates["excluded_product_ids"] = [resolution.product_id]
    elif entity is not None and entity.identifier_type != "auto" and (
        resolution is None or resolution.status != "constraint_entity"
    ):
        identifier_type = entity.identifier_type
        if identifier_type == "product_id":
            updates["product_id"] = entity.raw_text
        elif identifier_type == "sku":
            updates["sku"] = entity.raw_text
        elif identifier_type == "model_family":
            updates["model_family"] = entity.raw_text
        else:
            updates["model"] = entity.raw_text
    _copy_conjunctive_filters(filter_expression, updates)
    if "required_filter_fields" in updates:
        updates["required_filter_fields"] = list(
            dict.fromkeys(updates["required_filter_fields"])
        )
    return ProductSearchArguments.model_validate(updates)


def _copy_conjunctive_filters(
    expression: SemanticExpression | None,
    updates: dict[str, Any],
) -> None:
    if expression is None:
        return
    if expression.kind == "all_of":
        for child in expression.expressions or []:
            _copy_conjunctive_filters(child, updates)
        return
    if expression.kind != "predicate" or expression.predicate is None:
        return
    predicate = expression.predicate
    if predicate.strength != "hard":
        return
    field, operator, value = predicate.field, predicate.operator, predicate.value
    if field in PRICE_FIELDS:
        if operator in {"lt", "lte"} and isinstance(value, (int, float)):
            updates["max_price"] = float(value)
        elif operator in {"gt", "gte"} and isinstance(value, (int, float)):
            updates["min_price"] = float(value)
        return
    if field in STOCK_FIELDS and operator == "eq":
        updates["in_stock"] = value if isinstance(value, bool) else value == "in_stock"
        return
    top_level = {
        "category_id", "brand", "model_family", "color_code", "storage_gb", "ram_gb",
        "btu", "screen_size_in", "connectivity", "active_noise_cancellation",
    }
    if field in top_level and operator == "eq":
        updates[field] = value
        if field not in {"category_id"}:
            updates.setdefault("required_filter_fields", []).append(field)
        return
    if field in NUMERIC_ATTRIBUTE_FIELDS and operator in {"eq", "gte", "lte"}:
        updates.setdefault("attribute_filters", []).append(
            AttributeFilter(field=field, operator=operator, value=value)
        )
        updates.setdefault("required_filter_fields", []).append(field)


def _fallback_branches(expression: SemanticExpression | None) -> tuple[SemanticExpression | None, ...]:
    if expression is None:
        return (None,)
    if expression.kind == "fallback":
        return (
            *_fallback_branches(expression.primary),
            *_fallback_branches(expression.secondary),
        )[:8]
    if expression.expressions is not None:
        child_branches = [_fallback_branches(child) for child in expression.expressions]
        return tuple(
            expression.model_copy(update={"expressions": list(children)})
            for children in product(*child_branches)
        )[:8]
    if expression.expression is not None:
        return tuple(
            expression.model_copy(update={"expression": child})
            for child in _fallback_branches(expression.expression)
            if child is not None
        )[:8]
    return (expression,)


def _selection_filter_expression(
    expression: SemanticExpression | None,
) -> tuple[SemanticExpression | None, bool]:
    """Project predicate-only selection trees into the executable filter contract.

    This is structural canonicalization, not natural-language interpretation. Entity-only
    selection trees continue through entity resolution. Mixing entity references and catalog
    predicates inside the same logical subtree is unsafe to project, so it requires clarification.
    """
    if expression is None:
        return None, False
    has_predicate = bool(iter_predicates(expression))
    has_entity_ref = _contains_entity_ref(expression)
    if has_predicate and has_entity_ref:
        return None, True
    return (expression, False) if has_predicate else (None, False)


def _contains_entity_ref(expression: SemanticExpression) -> bool:
    if expression.kind == "entity_ref":
        return True
    if any(_contains_entity_ref(child) for child in expression.expressions or []):
        return True
    return any(
        child is not None and _contains_entity_ref(child)
        for child in (expression.expression, expression.primary, expression.secondary)
    )


def _conjoin_expressions(
    left: SemanticExpression | None,
    right: SemanticExpression | None,
) -> SemanticExpression | None:
    if left is None:
        return right
    if right is None:
        return left
    expressions: list[SemanticExpression] = []
    for expression in (left, right):
        if expression.kind == "all_of" and expression.expressions is not None:
            expressions.extend(expression.expressions)
        else:
            expressions.append(expression)
    return SemanticExpression(kind="all_of", expressions=expressions)


def _selection_branches(
    expression: SemanticExpression | None,
    entities: list[Any],
    resolutions: tuple[EntityResolution, ...],
) -> list[tuple[Any, EntityResolution | None]]:
    by_id = {
        entity.entity_id: (entity, resolution)
        for entity, resolution in zip(entities, resolutions, strict=True)
    }
    if expression is not None and expression.kind == "entity_ref":
        selected = by_id.get(expression.entity_id)
        if selected is not None:
            return [selected]
    if expression is not None and expression.kind == "fallback":
        refs = []
        for child in (expression.primary, expression.secondary):
            if child is not None and child.kind == "entity_ref" and child.entity_id in by_id:
                refs.append(by_id[child.entity_id])
        if refs:
            return refs
    if entities:
        return [(entities[-1], resolutions[-1])]
    return [(None, None)]


def _resolved_selection_filter(
    expression: SemanticExpression | None,
    entities: list[Any],
    resolutions: tuple[EntityResolution, ...],
) -> SemanticExpression | None:
    if expression is None or expression.kind not in {"all_of", "any_of"}:
        return None
    by_id = {
        entity.entity_id: (entity, resolution)
        for entity, resolution in zip(entities, resolutions, strict=True)
    }

    def resolve(child: SemanticExpression) -> SemanticExpression | None:
        if child.kind == "entity_ref":
            selected = by_id.get(child.entity_id)
            if selected is None:
                return None
            entity, resolution = selected
            if resolution.product_id:
                field, value = "product_id", resolution.product_id
            elif resolution.constraint_field and resolution.constraint_value is not None:
                field, value = resolution.constraint_field, resolution.constraint_value
            else:
                return None
            return SemanticExpression(
                kind="predicate",
                predicate=SemanticPredicate(
                    field=field,
                    operator="eq",
                    value=value,
                    strength="hard",
                    evidence_text=entity.evidence_text,
                ),
            )
        if child.kind in {"all_of", "any_of"}:
            resolved_children = [resolve(item) for item in child.expressions or []]
            if any(item is None for item in resolved_children):
                return None
            return SemanticExpression(
                kind=child.kind,
                expressions=[item for item in resolved_children if item is not None],
            )
        return None

    return resolve(expression)


def _entities_represented_by_filter(
    plan: ProductQueryPlan,
    entities: list[Any],
) -> bool:
    if plan.filter_expression is None or not entities:
        return False
    predicates = iter_predicates(plan.filter_expression)
    for entity in entities:
        matching = False
        for predicate in predicates:
            values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
            normalized_values = {
                normalize_text(value) for value in values if isinstance(value, str)
            }
            same_value = normalize_text(entity.raw_text) in normalized_values
            inherited_same_meaning = bool(
                set(entity.memory_refs) & set(predicate.memory_refs)
            )
            same_family_field = (
                entity.identifier_type == "model_family"
                and predicate.field == "model_family"
            )
            if same_value or (same_family_field and inherited_same_meaning):
                matching = True
                break
        if not matching:
            return False
    return True


def _canonicalize_selected_entities(plan: ProductQueryPlan) -> ProductQueryPlan:
    selected_ids = {entity.entity_id for entity in plan.entities if entity.state == "selected"}

    def prune(expression: SemanticExpression | None) -> SemanticExpression | None:
        if expression is None:
            return None
        if expression.kind == "entity_ref":
            return expression if expression.entity_id in selected_ids else None
        if expression.kind in {"all_of", "any_of"}:
            children = [
                item
                for child in expression.expressions or []
                if (item := prune(child)) is not None
            ]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return expression.model_copy(update={"expressions": children})
        if expression.kind in {"not", "prefer"}:
            child = prune(expression.expression)
            return expression.model_copy(update={"expression": child}) if child else None
        if expression.kind == "fallback":
            primary, secondary = prune(expression.primary), prune(expression.secondary)
            if primary is None:
                return secondary
            if secondary is None:
                return primary
            return expression.model_copy(update={"primary": primary, "secondary": secondary})
        return expression

    canonical_selection = prune(plan.selection_expression)
    if canonical_selection == plan.selection_expression:
        return plan
    return plan.model_copy(update={"selection_expression": canonical_selection})


def _requested_field(question: FactQuestion) -> str:
    aliases = {
        "sale_price": "price",
        "in_stock": "stock_status",
        "color_code": "color",
    }
    return aliases.get(question.field, question.field)


def _plan_expressions(plan: ProductQueryPlan) -> list[SemanticExpression]:
    return [
        expression
        for expression in (
            plan.selection_expression,
            plan.filter_expression,
            plan.preference_expression,
        )
        if expression is not None
    ]


def _ground_plan_predicates(
    plan: ProductQueryPlan,
    catalog: CatalogBackend,
) -> tuple[
    ProductQueryPlan,
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    mappings: list[dict[str, Any]] = []
    Grounded = tuple[
        SemanticExpression | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]

    def ground_expression(expression: SemanticExpression | None) -> Grounded:
        if expression is None:
            return None, [], []
        if expression.kind == "predicate" and expression.predicate is not None:
            predicate = expression.predicate
            raw_values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
            grounded_values: list[Any] = []
            grounded_fields: list[str] = []
            local_issues: list[dict[str, Any]] = []
            for raw_value in raw_values:
                if not isinstance(raw_value, str):
                    grounded_values.append(raw_value)
                    grounded_fields.append(predicate.field)
                    continue
                resolution = catalog.resolve_predicate_value(predicate.field, raw_value)
                canonical_field = predicate.field
                if resolution.status == "unmapped":
                    namespace_matches = catalog.resolve_facet_namespaces(raw_value)
                    if len(namespace_matches) == 1:
                        resolution = namespace_matches[0]
                        canonical_field = resolution.field
                mappings.append(
                    {
                        "field": predicate.field,
                        "canonical_field": canonical_field,
                        "original": raw_value,
                        "canonical": resolution.canonical_value,
                        "status": resolution.status,
                        "candidates": list(resolution.candidates),
                        "evidence_text": predicate.evidence_text,
                    }
                )
                if resolution.status == "ambiguous":
                    local_issues.append(
                        {
                            "field": predicate.field,
                            "value": raw_value,
                            "reason": "ambiguous_facet_value",
                            "evidence_text": predicate.evidence_text,
                        }
                    )
                elif resolution.status == "unmapped" and predicate.strength == "hard":
                    local_issues.append(
                        {
                            "field": predicate.field,
                            "value": raw_value,
                            "reason": "unmapped_required_value",
                            "operator": predicate.operator,
                            "evidence_text": predicate.evidence_text,
                        }
                    )
                grounded_values.append(resolution.canonical_value or raw_value)
                grounded_fields.append(canonical_field)
            unique_fields = set(grounded_fields)
            if len(unique_fields) > 1:
                if predicate.operator in {"in", "not_in"}:
                    child_operator = "eq" if predicate.operator == "in" else "not_eq"
                    child_kind = "any_of" if predicate.operator == "in" else "all_of"
                    return (
                        SemanticExpression(
                            kind=child_kind,
                            expressions=[
                                SemanticExpression(
                                    kind="predicate",
                                    predicate=predicate.model_copy(
                                        update={
                                            "field": field,
                                            "operator": child_operator,
                                            "value": value,
                                        }
                                    ),
                                )
                                for field, value in zip(
                                    grounded_fields,
                                    grounded_values,
                                    strict=True,
                                )
                            ],
                        ),
                        local_issues,
                        [],
                    )
                local_issues.append(
                    {
                        "field": predicate.field,
                        "value": predicate.value,
                        "reason": "mixed_facet_namespaces",
                        "evidence_text": predicate.evidence_text,
                    }
                )
            grounded_value: Any = grounded_values if isinstance(predicate.value, list) else grounded_values[0]
            return (
                expression.model_copy(
                    update={
                        "predicate": predicate.model_copy(
                            update={
                                "field": grounded_fields[0],
                                "value": grounded_value,
                            }
                        )
                    }
                ),
                local_issues,
                [],
            )

        if expression.expressions is not None:
            children = [ground_expression(child) for child in expression.expressions]
            if expression.kind == "any_of":
                valid_children = [
                    grounded
                    for grounded, child_issues, _ in children
                    if grounded is not None and not child_issues
                ]
                blocking_issues = [
                    issue
                    for _, child_issues, _ in children
                    for issue in child_issues
                    if issue.get("reason") != "unmapped_required_value"
                ]
                unavailable = [
                    *(
                        unavailable_item
                        for _, _, child_unavailable in children
                        for unavailable_item in child_unavailable
                    ),
                    *(
                        issue
                        for _, child_issues, _ in children
                        for issue in child_issues
                        if issue.get("reason") == "unmapped_required_value"
                    ),
                ]
                if valid_children and not blocking_issues:
                    grounded_expression = (
                        valid_children[0]
                        if len(valid_children) == 1
                        else expression.model_copy(update={"expressions": valid_children})
                    )
                    return grounded_expression, [], unavailable
                child_issues = [
                    issue for _, issues, _ in children for issue in issues
                ]
                grounded_children = [
                    grounded for grounded, _, _ in children if grounded is not None
                ]
                return (
                    expression.model_copy(update={"expressions": grounded_children}),
                    child_issues,
                    unavailable,
                )

            grounded_children = [
                grounded for grounded, _, _ in children if grounded is not None
            ]
            child_issues = [issue for _, issues, _ in children for issue in issues]
            unavailable = [item for _, _, values in children for item in values]
            return (
                expression.model_copy(update={"expressions": grounded_children}),
                child_issues,
                unavailable,
            )

        updates: dict[str, Any] = {}
        issues: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        for field in ("expression", "primary", "secondary"):
            child = getattr(expression, field)
            if child is not None:
                grounded, child_issues, child_unavailable = ground_expression(child)
                updates[field] = grounded
                issues.extend(child_issues)
                unavailable.extend(child_unavailable)
        return (
            expression.model_copy(update=updates) if updates else expression,
            issues,
            unavailable,
        )

    selection, selection_issues, selection_unavailable = ground_expression(
        plan.selection_expression
    )
    filters, filter_issues, filter_unavailable = ground_expression(plan.filter_expression)
    preferences, preference_issues, preference_unavailable = ground_expression(
        plan.preference_expression
    )
    raw_issues = [*selection_issues, *filter_issues, *preference_issues]
    deterministic_unavailable = [
        issue
        for issue in raw_issues
        if issue.get("reason") == "unmapped_required_value"
        and issue.get("operator") in {"eq", "in"}
    ]
    issues = [issue for issue in raw_issues if issue not in deterministic_unavailable]
    unavailable_values = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in [
                *selection_unavailable,
                *filter_unavailable,
                *preference_unavailable,
                *deterministic_unavailable,
            ]
        }.values()
    )
    unavailable_values = [
        {key: value for key, value in item.items() if key != "operator"}
        for item in unavailable_values
    ]

    grounded_plan = plan.model_copy(
        update={
            "selection_expression": selection,
            "filter_expression": filters,
            "preference_expression": preferences,
        }
    )
    issue = None
    if issues:
        issue = {
            "reason": issues[0]["reason"],
            "question": "Şərtin hansı kataloq dəyərini nəzərdə tutduğunu dəqiqləşdirə bilərsiniz?",
            "issues": issues,
        }
    return grounded_plan, mappings, issue, unavailable_values


def _resolve_entity(entity: Any, catalog: CatalogBackend) -> EntityResolution:
    # A replacement entity may refer to old memory for context, but the old product ID must
    # never override its new raw meaning. Supersession is semantic-plan structure, not wording.
    if not entity.context_product_id or entity.supersedes_entity_id:
        return catalog.resolve_entity(entity)
    resolved = catalog.resolve_entity(
        entity.model_copy(
            update={
                "raw_text": entity.context_product_id,
                "identifier_type": "product_id",
            }
        )
    )
    return replace(resolved, raw_text=entity.raw_text, reason="session_context_candidate")


def _resolve_superseding_entity(
    entity: Any,
    plan: ProductQueryPlan,
    catalog: CatalogBackend,
    resolution: EntityResolution,
) -> EntityResolution:
    """Use an explicit semantic supersession link to disambiguate a shortened replacement.

    The model decides that one entity supersedes another. This helper only compares catalog-backed
    model tokens; it does not interpret words such as negation or correction markers.
    """
    if resolution.status != "ambiguous" or not entity.supersedes_entity_id:
        return resolution
    previous = next(
        (
            item
            for item in plan.entities
            if item.entity_id == entity.supersedes_entity_id
        ),
        None,
    )
    if previous is None:
        return resolution
    previous_resolution = _resolve_entity(previous, catalog)
    if not previous_resolution.product_id:
        return resolution
    previous_product = catalog.product_by_id(previous_resolution.product_id)
    if previous_product is None:
        return resolution
    previous_tokens = _product_identity_tokens(previous_product)
    ranked: list[tuple[float, str]] = []
    for candidate in resolution.candidates:
        product = catalog.product_by_id(candidate.product_id)
        if product is None:
            continue
        tokens = _product_identity_tokens(product)
        union = previous_tokens | tokens
        score = len(previous_tokens & tokens) / len(union) if union else 0.0
        ranked.append((score, candidate.product_id))
    ranked.sort(reverse=True)
    if not ranked:
        return resolution
    best_score, best_id = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score <= 0 or best_score - second_score < 0.1:
        return resolution
    return replace(
        resolution,
        status="resolved",
        product_id=best_id,
        reason="supersession_catalog_context",
    )


def _product_identity_tokens(product: dict[str, Any]) -> set[str]:
    value = " ".join(
        str(product.get(field) or "")
        for field in ("brand", "model_family", "model")
    )
    return {token for token in normalize_text(value).split() if token}


def _validate_predicate_types(plan: ProductQueryPlan) -> None:
    if plan.filter_expression is not None and any(
        predicate.strength != "hard" for predicate in iter_predicates(plan.filter_expression)
    ):
        raise SemanticPlanValidationError("filter_expression may contain only hard predicates")
    if plan.preference_expression is not None and any(
        predicate.strength != "preference"
        for predicate in iter_predicates(plan.preference_expression)
    ):
        raise SemanticPlanValidationError(
            "preference_expression may contain only preference predicates"
        )
    for expression in _plan_expressions(plan):
        for predicate in iter_predicates(expression):
            numeric = predicate.field in PRICE_FIELDS or predicate.field in (
                set(NUMERIC_ATTRIBUTE_FIELDS) | {"rating", "warranty_months"}
            )
            values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
            if numeric and any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            ):
                raise SemanticPlanValidationError(
                    f"numeric value is required for field {predicate.field}"
                )
            if predicate.field in (set(BOOLEAN_ATTRIBUTE_FIELDS) | {"in_stock"}) and any(
                not isinstance(value, bool) for value in values
            ):
                raise SemanticPlanValidationError(
                    f"boolean value is required for field {predicate.field}"
                )
            if predicate.operator in {"lt", "lte", "gt", "gte"} and (
                not numeric
                or isinstance(predicate.value, bool)
                or not isinstance(predicate.value, (int, float))
            ):
                raise SemanticPlanValidationError(
                    f"numeric comparison is invalid for field {predicate.field}"
                )
            if predicate.operator in {"in", "not_in"} and not isinstance(predicate.value, list):
                raise SemanticPlanValidationError(f"{predicate.operator} requires a list value")
            if predicate.field in PRICE_FIELDS and predicate.unit not in {None, "AZN"}:
                raise SemanticPlanValidationError("price unit must match catalog currency")


def _predicate_matches(predicate: SemanticPredicate | None, product: dict[str, Any]) -> bool:
    if predicate is None:
        return False
    actual = _product_value(product, predicate.field)
    expected = predicate.value
    if predicate.operator in {"eq", "not_eq"}:
        matched = _equal(actual, expected)
        return matched if predicate.operator == "eq" else not matched
    if predicate.operator in {"in", "not_in"}:
        values = expected if isinstance(expected, list) else []
        matched = any(_equal(actual, value) for value in values)
        return matched if predicate.operator == "in" else not matched
    try:
        left, right = float(actual), float(expected)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return {
        "lt": left < right,
        "lte": left <= right,
        "gt": left > right,
        "gte": left >= right,
    }[predicate.operator]


def _product_value(product: dict[str, Any], field: str) -> Any:
    if field == "category_id":
        return product.get("category", {}).get("id", product.get("category_id"))
    if field == "color_code":
        return product.get("color", {}).get("code", product.get("color_code"))
    if field in PRICE_FIELDS:
        return product.get("price", {}).get("sale", product.get("sale_price"))
    if field in STOCK_FIELDS:
        if field == "in_stock":
            if "stock" in product:
                return product.get("stock", {}).get("status") == "in_stock"
            return product.get("in_stock")
        status = product.get("stock", {}).get("status", product.get("stock_status"))
        if status is None and isinstance(product.get("in_stock"), bool):
            status = "in_stock" if product["in_stock"] else "out_of_stock"
        return status
    if field in TOP_LEVEL_FIELDS:
        return product.get(field)
    return product.get("attributes", {}).get(field, product.get(field))


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return normalize_text(left) == normalize_text(right)
    return left == right
