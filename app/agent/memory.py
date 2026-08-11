from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

MemoryAction = Literal["replace", "merge", "preserve"]


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryEntity(MemoryModel):
    memory_id: str
    entity_id: str
    raw_text: str = Field(max_length=200)
    anchor_kind: Literal["product", "facet"] = "product"
    product_id: str | None = None
    facet_field: str | None = Field(default=None, max_length=100)
    facet_value: str | None = Field(default=None, max_length=200)
    name: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=120)
    model_family: str | None = Field(default=None, max_length=120)
    brand: str | None = Field(default=None, max_length=120)
    category_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_anchor(self) -> MemoryEntity:
        if self.anchor_kind == "product" and not self.product_id:
            raise ValueError("product memory anchors require product_id")
        if self.anchor_kind == "facet" and not (self.facet_field and self.facet_value):
            raise ValueError("facet memory anchors require facet_field and facet_value")
        return self


class MemoryPredicate(MemoryModel):
    memory_id: str
    field: str = Field(max_length=100)
    operator: str = Field(max_length=20)
    value: str | int | float | bool | list[str] | list[int] | list[float]
    strength: Literal["hard", "preference"]
    unit: str | None = Field(default=None, max_length=30)


class MemoryFactQuestion(MemoryModel):
    memory_id: str
    field: str = Field(max_length=100)
    operator: str | None = Field(default=None, max_length=20)
    value: str | int | float | bool | list[str] | list[int] | list[float] | None = None
    unit: str | None = Field(default=None, max_length=30)


class ProductMemoryState(MemoryModel):
    operation: str | None = Field(default=None, max_length=30)
    entities: list[MemoryEntity] = Field(default_factory=list, max_length=3)
    hard_constraints: list[MemoryPredicate] = Field(default_factory=list, max_length=20)
    preferences: list[MemoryPredicate] = Field(default_factory=list, max_length=20)
    selection_expression: dict[str, Any] | None = None
    fact_questions: list[MemoryFactQuestion] = Field(default_factory=list, max_length=20)
    last_fact_fields: list[str] = Field(default_factory=list, max_length=20)
    display_product_ids: list[str] = Field(default_factory=list, max_length=3)
    recommended_product_id: str | None = None
    last_match_status: str | None = Field(default=None, max_length=40)
    constraint_conflicts: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_constraint_limit(self) -> ProductMemoryState:
        if len(self.hard_constraints) + len(self.preferences) > 20:
            raise ValueError("session memory supports at most 20 constraints and preferences")
        return self


class PendingMemoryIntent(MemoryModel):
    memory_id: str | None = None
    reason: str = Field(max_length=120)
    question: str | None = Field(default=None, max_length=300)
    operation: str | None = Field(default=None, max_length=30)
    raw_entities: list[str] = Field(default_factory=list, max_length=3)
    state: ProductMemoryState = Field(default_factory=ProductMemoryState)


class DocumentMemoryState(MemoryModel):
    topic: str = Field(max_length=300)
    source_ids: list[str] = Field(default_factory=list, max_length=5)
    last_match_status: str | None = Field(default=None, max_length=40)


class SessionMemory(MemoryModel):
    version: int = 3
    revision: int = Field(default=0, ge=0)
    last_request_id: str | None = None
    continuation_summary: str = Field(default="", max_length=1500)
    summary_source_request_id: str | None = None
    confirmed_state: ProductMemoryState = Field(
        default_factory=ProductMemoryState,
        validation_alias=AliasChoices("confirmed_state", "product"),
    )
    document: DocumentMemoryState | None = None
    pending_intent: PendingMemoryIntent | None = Field(
        default=None,
        validation_alias=AliasChoices("pending_intent", "pending_clarification"),
    )

    @property
    def pending_clarification(self) -> PendingMemoryIntent | None:
        """Compatibility accessor for v1 callers and stored debug fixtures."""
        return self.pending_intent

    @property
    def product(self) -> ProductMemoryState:
        """Compatibility accessor for v1 callers while v2 persists confirmed_state."""
        return self.confirmed_state


@dataclass(frozen=True)
class MemoryUpdate:
    memory: SessionMemory
    transition: dict[str, Any]


def load_session_memory(context: dict[str, Any] | None) -> SessionMemory:
    raw = (context or {}).get("memory")
    if not isinstance(raw, dict):
        return SessionMemory()
    try:
        loaded = SessionMemory.model_validate(raw)
        if loaded.version < 3:
            loaded = loaded.model_copy(update={"version": 3})
        return loaded
    except ValidationError:
        return SessionMemory()


def memory_context_payload(memory: SessionMemory) -> dict[str, Any]:
    """Return only the bounded, verified state that may be shown to the model."""
    product = _product_context_payload(memory.product)
    pending = memory.pending_intent
    return {
        "version": memory.version,
        "continuation_summary": memory.continuation_summary,
        "confirmed_state": product,
        "pending_intent": (
            {
                "memory_id": pending.memory_id,
                "reason": pending.reason,
                "question": pending.question,
                "operation": pending.operation,
                "raw_entities": list(pending.raw_entities),
                "state": _product_context_payload(pending.state),
            }
            if pending is not None
            else None
        ),
        "document": (
            {
                "topic": memory.document.topic,
                "source_ids": list(memory.document.source_ids),
                "last_match_status": memory.document.last_match_status,
            }
            if memory.document is not None
            else None
        ),
    }


def _product_context_payload(product: ProductMemoryState) -> dict[str, Any]:
    return {
        "operation": product.operation,
        "entities": [
            {
                "memory_id": item.memory_id,
                "anchor_kind": item.anchor_kind,
                "product_id": item.product_id,
                "facet_field": item.facet_field,
                "facet_value": item.facet_value,
                "name": item.name,
                "model": item.model,
                "model_family": item.model_family,
                "brand": item.brand,
                "category_id": item.category_id,
            }
            for item in product.entities
        ],
        "hard_constraints": [
            item.model_dump(mode="json", exclude_none=True)
            for item in product.hard_constraints
        ],
        "preferences": [
            item.model_dump(mode="json", exclude_none=True)
            for item in product.preferences
        ],
        "selection_expression": product.selection_expression,
        "fact_questions": [
            item.model_dump(mode="json", exclude_none=True)
            for item in product.fact_questions
        ],
        "last_fact_fields": list(product.last_fact_fields),
        "display_product_ids": list(product.display_product_ids),
        "recommended_product_id": product.recommended_product_id,
        "last_match_status": product.last_match_status,
        "constraint_conflicts": list(product.constraint_conflicts),
    }


def memory_reference_map(memory: SessionMemory) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    _add_state_references(references, memory.product, scope="confirmed")
    if memory.pending_intent is not None:
        _add_state_references(references, memory.pending_intent.state, scope="pending")
    if memory.pending_intent and memory.pending_intent.memory_id:
        references[memory.pending_intent.memory_id] = {
            "kind": "pending_intent",
            **memory.pending_intent.model_dump(mode="json", exclude_none=True),
        }
    return references


def _add_state_references(
    references: dict[str, dict[str, Any]],
    state: ProductMemoryState,
    *,
    scope: str,
) -> None:
    for entity in state.entities:
        references[entity.memory_id] = {
            "kind": "entity",
            "scope": scope,
            **entity.model_dump(mode="json", exclude_none=True),
        }
    for predicate in [*state.hard_constraints, *state.preferences]:
        references[predicate.memory_id] = {
            "kind": "predicate",
            "scope": scope,
            **predicate.model_dump(mode="json", exclude_none=True),
        }
    for question in state.fact_questions:
        references[question.memory_id] = {
            "kind": "fact_question",
            "scope": scope,
            **question.model_dump(mode="json", exclude_none=True),
        }


def semantic_memory_hash(memory: SessionMemory) -> str:
    payload = memory_context_payload(memory)
    payload.pop("continuation_summary", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def update_session_memory(
    before: SessionMemory,
    *,
    request_id: str,
    product_plan: dict[str, Any] | None,
    product_result: dict[str, Any] | None,
    document_arguments: dict[str, Any] | None,
    document_result: dict[str, Any] | None,
    max_bytes: int,
) -> MemoryUpdate:
    before_references = memory_reference_map(before)
    before_ids = set(before_references)
    action: MemoryAction = "preserve"
    after = before.model_copy(deep=True)

    if (
        product_result
        and product_result.get("status") == "success"
        and product_plan
        and _product_result_is_memory_safe(product_result)
    ):
        requested_action = product_plan.get("memory_action", "replace")
        action = requested_action if requested_action in {"replace", "merge", "preserve"} else "replace"
        after = _update_product_memory(
            before,
            request_id=request_id,
            plan=product_plan,
            result=product_result,
            action=action,
        )
        after = after.model_copy(
            update={
                "continuation_summary": _build_product_continuation_summary(
                    after,
                    action=action,
                ),
                "summary_source_request_id": request_id,
            }
        )
    elif document_result and document_result.get("status") == "success" and document_arguments:
        action = "merge"
        after = _update_document_memory(
            before,
            request_id=request_id,
            arguments=document_arguments,
            result=document_result,
        )
        after = after.model_copy(
            update={
                "continuation_summary": _build_document_continuation_summary(after),
                "summary_source_request_id": request_id,
            }
        )

    after = _fit_memory(after, max_bytes=max_bytes)
    changed = _memory_semantically_changed(before, after)
    if changed and after.revision <= before.revision:
        after = after.model_copy(update={"revision": before.revision + 1})
    if not changed:
        after = before.model_copy(deep=True)
        action = "preserve"

    after_references = memory_reference_map(after)
    after_ids = set(after_references)
    changed_ids = sorted(
        (after_ids - before_ids)
        | {
            memory_id
            for memory_id in before_ids & after_ids
            if before_references[memory_id] != after_references[memory_id]
        }
    )
    if before.product != after.product and not changed_ids:
        changed_ids.append("product_state")
    if before.pending_intent != after.pending_intent:
        changed_ids.append("pending_intent")
    if before.document != after.document:
        changed_ids.append("document_state")
    summary_replaced = before.continuation_summary != after.continuation_summary
    context_source = _context_source(after)
    size_bytes = len(_canonical_json(after.model_dump(mode="json", exclude_none=True)).encode("utf-8"))
    return MemoryUpdate(
        memory=after,
        transition={
            "revision_before": before.revision,
            "revision_after": after.revision,
            "action": action,
            "changed_ids": list(dict.fromkeys(changed_ids)),
            "removed_ids": sorted(before_ids - after_ids),
            "size_bytes": size_bytes,
            "context_source": context_source,
            "summary_replaced": summary_replaced,
            "summary_size_chars": len(after.continuation_summary),
            "confirmed_state_changed": before.product != after.product,
            "pending_state_changed": before.pending_intent != after.pending_intent,
        },
    )


def _product_result_is_memory_safe(result: dict[str, Any]) -> bool:
    clarification = result.get("clarification")
    if not isinstance(clarification, dict):
        return True
    reason = str(clarification.get("reason") or "")
    unsafe_reasons = {
        "invalid_memory_reference",
        "evidence_not_grounded",
        "unsupported_catalog_field",
        "invalid_context_reference",
        "semantic_plan_invalid",
        "invalid_semantic_plan",
        "invalid_tool_arguments",
    }
    return not any(item in reason.split(",") for item in unsafe_reasons)


def preserved_transition(memory: SessionMemory) -> dict[str, Any]:
    size_bytes = len(_canonical_json(memory.model_dump(mode="json", exclude_none=True)).encode("utf-8"))
    return {
        "revision_before": memory.revision,
        "revision_after": memory.revision,
        "action": "preserve",
        "changed_ids": [],
        "removed_ids": [],
        "size_bytes": size_bytes,
        "context_source": _context_source(memory),
        "summary_replaced": False,
        "summary_size_chars": len(memory.continuation_summary),
        "confirmed_state_changed": False,
        "pending_state_changed": False,
    }


def _update_product_memory(
    before: SessionMemory,
    *,
    request_id: str,
    plan: dict[str, Any],
    result: dict[str, Any],
    action: MemoryAction,
) -> SessionMemory:
    clarification = result.get("clarification")
    match_status = str(result.get("match_status") or "not_found")
    raw_entities = _grounded_pending_entities(plan)
    current = _product_state_from_run(plan, result)
    merge_base = (
        before.pending_intent.state
        if before.pending_intent is not None and action == "merge"
        else before.product
    )
    if isinstance(clarification, dict) or match_status in {"not_found", "alternatives"}:
        if action == "replace":
            pending_state = current
        elif action == "preserve":
            pending_state = merge_base.model_copy(
                deep=True,
                update={
                    "fact_questions": current.fact_questions or merge_base.fact_questions,
                    "last_fact_fields": current.last_fact_fields or merge_base.last_fact_fields,
                    "display_product_ids": current.display_product_ids,
                    "recommended_product_id": current.recommended_product_id,
                    "last_match_status": current.last_match_status,
                    "constraint_conflicts": current.constraint_conflicts,
                },
            )
        else:
            pending_state = _merge_product_state(merge_base, current, plan)
        pending_identity = {
            "operation": str(plan.get("operation") or "discover")[:30],
            "raw_entities": raw_entities,
            "state": pending_state.model_dump(mode="json", exclude_none=True),
        }
        pending = PendingMemoryIntent(
            memory_id=_memory_id("pending", pending_identity),
            reason=str((clarification or {}).get("reason") or match_status)[:120],
            question=(
                str((clarification or {}).get("question"))[:300]
                if (clarification or {}).get("question")
                else None
            ),
            operation=str(plan.get("operation") or "discover")[:30],
            raw_entities=raw_entities,
            state=pending_state,
        )
        return before.model_copy(
            deep=True,
            update={
                "last_request_id": request_id,
                "confirmed_state": before.product,
                "pending_intent": pending,
            },
        )

    if action == "replace":
        product = current
    elif action == "preserve":
        product = before.product.model_copy(
            deep=True,
            update={
                "fact_questions": current.fact_questions or before.product.fact_questions,
                "last_fact_fields": current.last_fact_fields or before.product.last_fact_fields,
                "display_product_ids": current.display_product_ids,
                "recommended_product_id": current.recommended_product_id,
                "last_match_status": current.last_match_status,
                "constraint_conflicts": current.constraint_conflicts,
            },
        )
    else:
        product = _merge_product_state(merge_base, current, plan)
    return before.model_copy(
        deep=True,
        update={
            "last_request_id": request_id,
            "confirmed_state": product,
            "pending_intent": None,
        },
    )


def _product_state_from_run(
    plan: dict[str, Any],
    result: dict[str, Any],
) -> ProductMemoryState:
    products: dict[str, dict[str, Any]] = {}
    for item in [*result.get("items", []), result.get("requested_item")]:
        if isinstance(item, dict) and item.get("product_id"):
            products[str(item["product_id"])] = item
    resolution_by_entity = {
        str(item.get("entity_id")): item
        for item in result.get("resolved_entities", [])
        if isinstance(item, dict) and item.get("entity_id")
    }
    entities: list[MemoryEntity] = []
    for item in plan.get("entities", []):
        if not isinstance(item, dict) or item.get("state", "selected") != "selected":
            continue
        entity_id = str(item.get("entity_id") or "entity")
        resolution = resolution_by_entity.get(entity_id, {})
        product_id = resolution.get("product_id") or item.get("context_product_id")
        facet_field = resolution.get("constraint_field")
        facet_value = resolution.get("constraint_value")
        anchor_kind: Literal["product", "facet"]
        if product_id:
            anchor_kind = "product"
        elif resolution.get("status") == "constraint_entity" and facet_field and facet_value is not None:
            anchor_kind = "facet"
        else:
            # Ambiguous and unresolved mentions are not durable verified references.
            continue
        product = products.get(str(product_id), {}) if product_id else {}
        canonical = {
            "entity_id": entity_id,
            "raw_text": str(item.get("raw_text") or "")[:200],
            "product_id": str(product_id) if product_id else None,
            "anchor_kind": anchor_kind,
            "facet_field": str(facet_field)[:100] if facet_field else None,
            "facet_value": str(facet_value)[:200] if facet_value is not None else None,
        }
        memory_identity = (
            {"kind": "product", "product_id": str(product_id)}
            if product_id
            else {
                "kind": "facet",
                "field": canonical["facet_field"],
                "value": canonical["facet_value"],
            }
        )
        facet_metadata = {
            str(facet_field): str(facet_value)
            for facet_field in (facet_field,)
            if facet_field in {"model", "model_family", "brand", "category_id"}
        }
        entities.append(
            MemoryEntity(
                memory_id=_memory_id("entity", memory_identity),
                **canonical,
                name=(str(product.get("name"))[:200] if product.get("name") else None),
                model=(
                    str(product.get("model") or facet_metadata.get("model"))[:120]
                    if product.get("model") or facet_metadata.get("model")
                    else None
                ),
                model_family=(
                    str(product.get("model_family") or facet_metadata.get("model_family"))[:120]
                    if product.get("model_family") or facet_metadata.get("model_family")
                    else None
                ),
                brand=(
                    str(product.get("brand") or facet_metadata.get("brand"))[:120]
                    if product.get("brand") or facet_metadata.get("brand")
                    else None
                ),
                category_id=(
                    str(
                        product.get("category_id")
                        or (
                            (product.get("category") or {}).get("id")
                            if isinstance(product.get("category"), dict)
                            else ""
                        )
                        or facet_metadata.get("category_id")
                    )[:100]
                    or None
                ),
            )
        )
    hard = _memory_predicates(plan.get("filter_expression"), "hard")
    preferences = _memory_predicates(plan.get("preference_expression"), "preference")
    fact_questions = _memory_fact_questions(plan.get("fact_questions", []))
    fact_fields = [item.field for item in fact_questions]
    display_ids = [str(item) for item in result.get("display_product_ids", []) if item][:3]
    hard = hard[:20]
    preferences = preferences[: max(0, 20 - len(hard))]
    return ProductMemoryState(
        operation=str(plan.get("operation") or "discover")[:30],
        entities=entities[:3],
        hard_constraints=hard,
        preferences=preferences,
        selection_expression=_strip_expression(plan.get("selection_expression")),
        fact_questions=fact_questions,
        last_fact_fields=list(dict.fromkeys(fact_fields)),
        display_product_ids=display_ids,
        recommended_product_id=(
            str(result.get("recommended_product_id"))
            if result.get("recommended_product_id")
            else None
        ),
        last_match_status=str(result.get("match_status") or "not_found")[:40],
        constraint_conflicts=[str(item)[:300] for item in result.get("constraint_conflicts", [])][:20],
    )


def _grounded_pending_entities(plan: dict[str, Any]) -> list[str]:
    """Keep only current-message evidence in pending intent memory.

    Confirmed inherited state already lives in ``before.product``. Reconstructed entity text that
    is absent from the current message must not become a new durable memory fact after a
    clarification.
    """
    query = " ".join(str(plan.get("query") or "").casefold().split())
    grounded: list[str] = []
    for item in plan.get("entities", []):
        if not isinstance(item, dict):
            continue
        evidence = str(item.get("evidence_text") or "").strip()
        normalized_evidence = " ".join(evidence.casefold().split())
        if not normalized_evidence or normalized_evidence not in query:
            continue
        raw_text = str(item.get("raw_text") or "").strip()
        normalized_raw = " ".join(raw_text.casefold().split())
        grounded.append(
            (raw_text if normalized_raw and normalized_raw in query else evidence)[:200]
        )
    return list(dict.fromkeys(grounded))[:3]


def _merge_product_state(
    before: ProductMemoryState,
    current: ProductMemoryState,
    plan: dict[str, Any],
) -> ProductMemoryState:
    removed = {str(item) for item in plan.get("removed_memory_ids", [])}
    entities = [item for item in before.entities if item.memory_id not in removed]
    constraints = [item for item in before.hard_constraints if item.memory_id not in removed]
    preferences = [item for item in before.preferences if item.memory_id not in removed]

    entities = _merge_entities(entities, current.entities, limit=3)
    constraints = _merge_predicates(constraints, current.hard_constraints, limit=20)
    preferences = _merge_predicates(
        preferences,
        current.preferences,
        limit=max(0, 20 - len(constraints)),
    )
    return ProductMemoryState(
        operation=current.operation or before.operation,
        entities=entities,
        hard_constraints=constraints,
        preferences=preferences,
        selection_expression=current.selection_expression or before.selection_expression,
        fact_questions=current.fact_questions or before.fact_questions,
        last_fact_fields=current.last_fact_fields or before.last_fact_fields,
        display_product_ids=current.display_product_ids,
        recommended_product_id=current.recommended_product_id,
        last_match_status=current.last_match_status,
        constraint_conflicts=current.constraint_conflicts,
    )


def _update_document_memory(
    before: SessionMemory,
    *,
    request_id: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> SessionMemory:
    source_ids = [
        str(item.get("document_id") or item.get("chunk_id"))
        for item in result.get("chunks", [])
        if isinstance(item, dict) and (item.get("document_id") or item.get("chunk_id"))
    ][:5]
    document = DocumentMemoryState(
        topic=str(arguments.get("query") or "")[:300],
        source_ids=list(dict.fromkeys(source_ids)),
        last_match_status=str(result.get("match_status") or "not_found")[:40],
    )
    return before.model_copy(
        deep=True,
        update={
            "last_request_id": request_id,
            "document": document,
        },
    )


def _build_product_continuation_summary(
    memory: SessionMemory,
    *,
    action: MemoryAction,
) -> str:
    pending = memory.pending_intent
    state = pending.state if pending is not None else memory.product
    parts = ["İstifadəçinin aktiv məhsul məqsədi"]
    if state.operation:
        parts[0] += f" {state.operation} əməliyyatıdır"
    else:
        parts[0] += " davam edir"

    entities = [
        item.name or item.model or item.model_family or item.raw_text
        for item in state.entities
        if item.name or item.model or item.model_family or item.raw_text
    ]
    if entities:
        parts.append(f"Aktiv seçim: {_summary_join(entities[:3])}")
    if state.hard_constraints:
        parts.append(
            "Məcburi şərtlər: "
            + "; ".join(_summary_predicate(item) for item in state.hard_constraints[:20])
        )
    if state.preferences:
        parts.append(
            "Üstünlüklər: "
            + "; ".join(_summary_predicate(item) for item in state.preferences[:10])
        )
    if state.last_fact_fields:
        parts.append(f"Son soruşulan fakt sahələri: {', '.join(state.last_fact_fields[:10])}")

    status = state.last_match_status
    if status == "not_found":
        parts.append("Kataloqda uyğun məhsul və etibarlı alternativ tapılmadı")
    elif status == "alternatives":
        parts.append("Dəqiq seçim tapılmadı və yaxın alternativlər göstərildi")
    elif status == "clarification_required":
        parts.append("Sorğunun davamı üçün istifadəçi aydınlaşdırması gözlənilir")
    elif status == "exact_conflict":
        parts.append("Dəqiq məhsul mövcuddur, lakin məcburi şərtlərlə konflikt var")
    elif status == "exact_match":
        parts.append("Dəqiq məhsul kataloqda təsdiqləndi")
    elif status == "matching_products":
        parts.append("Kataloqda şərtlərə uyğun məhsullar tapıldı")

    if pending is not None:
        parts.append("Bu pending məqsəd növbəti məhsul davamında qorunmalıdır")
    elif action == "replace":
        parts.append("Bu məqsəd əvvəlki məhsul məqsədini əvəz etdi")
    elif action == "merge":
        parts.append("Cari dəyişiklik əvvəlki məhsul məqsədi ilə birləşdirildi")
    else:
        parts.append("Təsdiqlənmiş məhsul vəziyyəti qorundu")
    summary = ". ".join(part.rstrip(".") for part in parts if part).strip()
    if summary and not summary.endswith("."):
        summary += "."
    return summary[:1500]


def _build_document_continuation_summary(memory: SessionMemory) -> str:
    document = memory.document
    if document is None:
        return memory.continuation_summary
    summary = (
        f"İstifadəçinin son sənəd mövzusu: {document.topic}. "
        f"Sənəd nəticəsinin statusu: {document.last_match_status or 'müəyyən deyil'}."
    )
    if _has_product_state(memory.product):
        summary += " Əvvəlki təsdiqlənmiş məhsul məqsədi ayrıca qorunur."
    return summary[:1500]


def _summary_predicate(predicate: MemoryPredicate) -> str:
    labels = {
        "category_id": "kateqoriya",
        "brand": "brend",
        "model_family": "məhsul ailəsi",
        "model": "model",
        "color_code": "rəng",
        "price": "qiymət",
        "sale_price": "qiymət",
        "in_stock": "stok",
        "stock_status": "stok",
        "storage_gb": "yaddaş",
        "ram_gb": "RAM",
    }
    operators = {
        "eq": "=",
        "not_eq": "≠",
        "lt": "<",
        "lte": "≤",
        "gt": ">",
        "gte": "≥",
        "in": "daxildir",
        "not_in": "daxil deyil",
    }
    value = predicate.value
    if isinstance(value, list):
        rendered = ", ".join(str(item) for item in value)
    else:
        rendered = str(value)
    unit = f" {predicate.unit}" if predicate.unit else ""
    return (
        f"{labels.get(predicate.field, predicate.field)} "
        f"{operators.get(predicate.operator, predicate.operator)} {rendered}{unit}"
    )


def _summary_join(values: list[str]) -> str:
    clean = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if len(clean) <= 1:
        return clean[0] if clean else ""
    return ", ".join(clean[:-1]) + f" və {clean[-1]}"


def _has_product_state(state: ProductMemoryState) -> bool:
    return bool(
        state.operation
        or state.entities
        or state.hard_constraints
        or state.preferences
        or state.display_product_ids
    )


def _context_source(memory: SessionMemory) -> Literal["confirmed", "pending", "none"]:
    if memory.pending_intent is not None:
        return "pending"
    if _has_product_state(memory.product):
        return "confirmed"
    return "none"


def _memory_fact_questions(value: Any) -> list[MemoryFactQuestion]:
    questions: list[MemoryFactQuestion] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not item.get("field"):
            continue
        canonical = {
            "field": str(item.get("field"))[:100],
            "operator": (str(item.get("operator"))[:20] if item.get("operator") else None),
            "value": item.get("value"),
            "unit": (str(item.get("unit"))[:30] if item.get("unit") else None),
        }
        questions.append(
            MemoryFactQuestion(
                memory_id=_memory_id("fact", canonical),
                **canonical,
            )
        )
    return questions[:20]


def _memory_predicates(expression: Any, expected_strength: str) -> list[MemoryPredicate]:
    result: list[MemoryPredicate] = []
    for predicate in _iter_predicate_dicts(expression):
        if predicate.get("strength") != expected_strength:
            continue
        canonical = {
            "field": str(predicate.get("field") or "")[:100],
            "operator": str(predicate.get("operator") or "eq")[:20],
            "value": predicate.get("value"),
            "strength": expected_strength,
            "unit": predicate.get("unit"),
        }
        result.append(
            MemoryPredicate(
                memory_id=_memory_id("predicate", canonical),
                **canonical,
            )
        )
    return result


def _iter_predicate_dicts(expression: Any) -> list[dict[str, Any]]:
    if not isinstance(expression, dict):
        return []
    result: list[dict[str, Any]] = []
    if expression.get("kind") == "predicate" and isinstance(expression.get("predicate"), dict):
        result.append(expression["predicate"])
    for child in expression.get("expressions") or []:
        result.extend(_iter_predicate_dicts(child))
    for field in ("expression", "primary", "secondary"):
        result.extend(_iter_predicate_dicts(expression.get(field)))
    return result


def _strip_expression(expression: Any) -> dict[str, Any] | None:
    if not isinstance(expression, dict):
        return None
    cleaned = {
        key: value
        for key, value in expression.items()
        if key not in {"evidence_text", "memory_refs"}
    }
    if isinstance(cleaned.get("predicate"), dict):
        cleaned["predicate"] = {
            key: value
            for key, value in cleaned["predicate"].items()
            if key not in {"evidence_text", "memory_refs"}
        }
    for field in ("expression", "primary", "secondary"):
        if field in cleaned:
            cleaned[field] = _strip_expression(cleaned[field])
    if "expressions" in cleaned:
        cleaned["expressions"] = [
            item for child in cleaned["expressions"] if (item := _strip_expression(child))
        ]
    return cleaned


def _merge_by_id(existing: list[Any], current: list[Any], *, limit: int) -> list[Any]:
    by_id = {item.memory_id: item for item in existing}
    for item in current:
        by_id[item.memory_id] = item
    return list(by_id.values())[-limit:]


def _merge_entities(
    existing: list[MemoryEntity],
    current: list[MemoryEntity],
    *,
    limit: int,
) -> list[MemoryEntity]:
    by_anchor: dict[tuple[str, str, str], MemoryEntity] = {}
    for item in [*existing, *current]:
        key = (
            item.anchor_kind,
            str(item.product_id or item.facet_field or "").casefold(),
            str(item.facet_value or "").casefold(),
        )
        by_anchor[key] = item
    return list(by_anchor.values())[-limit:]


def _merge_predicates(
    existing: list[MemoryPredicate],
    current: list[MemoryPredicate],
    *,
    limit: int,
) -> list[MemoryPredicate]:
    current_fields = {item.field for item in current}
    retained = [item for item in existing if item.field not in current_fields]
    return _merge_by_id(retained, current, limit=limit)


def _fit_memory(memory: SessionMemory, *, max_bytes: int) -> SessionMemory:
    current = memory.model_copy(deep=True)
    if _memory_size(current) > max_bytes and current.continuation_summary:
        overflow = _memory_size(current) - max_bytes
        keep = max(0, len(current.continuation_summary) - overflow - 16)
        shortened = current.continuation_summary[:keep].rstrip()
        current = current.model_copy(
            update={"continuation_summary": f"{shortened}…" if shortened else ""}
        )
    while _memory_size(current) > max_bytes and current.product.last_fact_fields:
        current.product.last_fact_fields.pop()
    if current.pending_intent is not None:
        while (
            _memory_size(current) > max_bytes
            and current.pending_intent.state.last_fact_fields
        ):
            current.pending_intent.state.last_fact_fields.pop()
    while _memory_size(current) > max_bytes and current.product.fact_questions:
        current.product.fact_questions.pop()
    if current.pending_intent is not None:
        while _memory_size(current) > max_bytes and current.pending_intent.state.fact_questions:
            current.pending_intent.state.fact_questions.pop()
    while _memory_size(current) > max_bytes and current.product.preferences:
        current.product.preferences.pop()
    if current.pending_intent is not None:
        while _memory_size(current) > max_bytes and current.pending_intent.state.preferences:
            current.pending_intent.state.preferences.pop()
    while _memory_size(current) > max_bytes and current.product.constraint_conflicts:
        current.product.constraint_conflicts.pop()
    if current.pending_intent is not None:
        while (
            _memory_size(current) > max_bytes
            and current.pending_intent.state.constraint_conflicts
        ):
            current.pending_intent.state.constraint_conflicts.pop()
    if _memory_size(current) > max_bytes:
        current.confirmed_state = current.product.model_copy(
            update={"selection_expression": None}
        )
    if _memory_size(current) > max_bytes and current.pending_intent is not None:
        current.pending_intent.state = current.pending_intent.state.model_copy(
            update={"selection_expression": None}
        )
    if _memory_size(current) > max_bytes:
        raise ValueError("Session memory exceeds the configured size limit")
    return current


def _memory_semantically_changed(before: SessionMemory, after: SessionMemory) -> bool:
    left = before.model_dump(mode="json", exclude={"revision", "last_request_id"})
    right = after.model_dump(mode="json", exclude={"revision", "last_request_id"})
    return left != right


def _memory_size(memory: SessionMemory) -> int:
    return len(_canonical_json(memory.model_dump(mode="json", exclude_none=True)).encode("utf-8"))


def _memory_id(kind: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]
    return f"mem_{kind}_{digest}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
