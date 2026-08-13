from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent.explanation import build_decision_explanation
from app.agent.memory import (
    SessionMemory,
    load_session_memory,
    memory_context_payload,
    memory_reference_map,
    semantic_memory_hash,
    update_session_memory,
)
from app.config import PROJECT_ROOT
from app.retrieval.semantic_plan import compile_semantic_plan
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductQueryPlan


def _catalog() -> ProductCatalog:
    catalog = ProductCatalog(PROJECT_ROOT / "data" / "catalog" / "products.jsonl")
    catalog.load()
    return catalog


def _exact_plan(*, memory_action: str = "replace") -> dict:
    return {
        "query": "iPhone 17 Pro",
        "operation": "lookup",
        "memory_action": memory_action,
        "entities": [
            {
                "entity_id": "phone",
                "raw_text": "iPhone 17 Pro",
                "state": "selected",
                "evidence_text": "iPhone 17 Pro",
                "identifier_type": "model",
            }
        ],
        "selection_expression": {"kind": "entity_ref", "entity_id": "phone"},
    }


def _exact_result() -> dict:
    return {
        "status": "success",
        "match_status": "exact_match",
        "items": [
            {
                "product_id": "prd_smartphones_001",
                "name": "Apple iPhone 17 Pro",
                "model": "iPhone 17 Pro",
                "model_family": "iPhone",
                "brand": "Apple",
                "category": {"id": "smartphones"},
            }
        ],
        "resolved_entities": [
            {"entity_id": "phone", "product_id": "prd_smartphones_001"}
        ],
        "display_product_ids": ["prd_smartphones_001"],
        "recommended_product_id": "prd_smartphones_001",
        "constraint_conflicts": [],
    }


def _seed_memory() -> SessionMemory:
    update = update_session_memory(
        SessionMemory(),
        request_id="req-1",
        product_plan=_exact_plan(),
        product_result=_exact_result(),
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )
    return update.memory


def test_product_run_creates_bounded_versioned_memory() -> None:
    update = update_session_memory(
        SessionMemory(),
        request_id="req-1",
        product_plan=_exact_plan(),
        product_result=_exact_result(),
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert update.memory.version == 4
    assert update.memory.revision == 1
    assert update.memory.product.entities[0].product_id == "prd_smartphones_001"
    assert update.transition["action"] == "replace"
    assert update.transition["revision_after"] == 1
    assert update.transition["size_bytes"] <= 8192
    assert 0 < len(update.memory.continuation_summary) <= 1500
    assert update.memory.summary_source_request_id == "req-1"
    assert "confirmed_state" in update.memory.model_dump(mode="json")
    assert "product" not in update.memory.model_dump(mode="json")


def test_model_context_omits_request_ids_and_raw_user_entity_text() -> None:
    memory = _seed_memory()

    payload = memory_context_payload(memory)
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "last_request_id" not in payload
    assert "raw_text" not in encoded
    assert "iPhone 17 Pro" in encoded
    assert "prd_smartphones_001" in encoded


def test_resolved_product_memory_id_is_stable_across_rephrasing() -> None:
    before = _seed_memory()
    original_id = before.product.entities[0].memory_id
    plan = _exact_plan(memory_action="merge")
    plan["query"] = "Apple iPhone 17 Pro modelini deyirəm"
    plan["entities"][0]["raw_text"] = "Apple iPhone 17 Pro"
    plan["entities"][0]["evidence_text"] = "Apple iPhone 17 Pro"

    update = update_session_memory(
        before,
        request_id="req-rephrased",
        product_plan=plan,
        product_result=_exact_result(),
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert len(update.memory.product.entities) == 1
    assert update.memory.product.entities[0].memory_id == original_id


def test_semantic_memory_hash_ignores_revision_and_request_id() -> None:
    memory = _seed_memory()
    changed_metadata = memory.model_copy(
        update={"revision": memory.revision + 7, "last_request_id": "another-request"}
    )

    assert semantic_memory_hash(memory) == semantic_memory_hash(changed_metadata)


def test_clarification_keeps_confirmed_product_state_and_records_pending_intent() -> None:
    before = _seed_memory()
    update = update_session_memory(
        before,
        request_id="req-2",
        product_plan={
            "query": "o biri olsun",
            "operation": "lookup",
            "memory_action": "merge",
            "entities": [
                {
                    "entity_id": "unknown",
                    "raw_text": "o biri",
                    "evidence_text": "o biri",
                }
            ],
        },
        product_result={
            "status": "success",
            "match_status": "clarification_required",
            "clarification": {
                "reason": "ambiguous_entity",
                "question": "Hansı məhsulu nəzərdə tutursunuz?",
            },
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert update.memory.product == before.product
    assert update.memory.pending_clarification is not None
    assert update.memory.pending_clarification.reason == "ambiguous_entity"
    assert update.memory.revision == before.revision + 1


def test_invalid_evidence_clarification_preserves_memory_without_pending_intent() -> None:
    before = _seed_memory()
    update = update_session_memory(
        before,
        request_id="req-pending-safe",
        product_plan={
            "query": "ya da qirmizi",
            "operation": "discover",
            "memory_action": "merge",
            "entities": [
                {
                    "entity_id": "iphone_red",
                    "raw_text": "Qara 128 GB iPhone və ya qirmizi",
                    "evidence_text": "qirmizi",
                }
            ],
        },
        product_result={
            "status": "success",
            "match_status": "clarification_required",
            "clarification": {
                "reason": "evidence_not_grounded",
                "question": "Sorğunu dəqiqləşdirə bilərsiniz?",
            },
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert update.memory.product == before.product
    assert update.memory.pending_clarification is None
    assert update.memory.revision == before.revision
    assert update.transition["action"] == "preserve"


def test_memory_backed_or_follow_up_keeps_valid_branch_and_reports_missing_color() -> None:
    catalog = _catalog()
    first_plan = {
        "query": "Qara 128 GB iPhone göstər",
        "operation": "discover",
        "filter_expression": {
            "kind": "all_of",
            "expressions": [
                {
                    "kind": "predicate",
                    "predicate": {
                        "field": "category_id",
                        "operator": "eq",
                        "value": "smartphones",
                        "strength": "hard",
                        "evidence_text": "iPhone",
                    },
                },
                {
                    "kind": "predicate",
                    "predicate": {
                        "field": "model_family",
                        "operator": "eq",
                        "value": "iPhone",
                        "strength": "hard",
                        "evidence_text": "iPhone",
                    },
                },
                {
                    "kind": "predicate",
                    "predicate": {
                        "field": "storage_gb",
                        "operator": "eq",
                        "value": 128,
                        "unit": "GB",
                        "strength": "hard",
                        "evidence_text": "128 GB",
                    },
                },
                {
                    "kind": "predicate",
                    "predicate": {
                        "field": "color_code",
                        "operator": "eq",
                        "value": "black",
                        "strength": "hard",
                        "evidence_text": "Qara",
                    },
                },
            ],
        },
    }
    memory = update_session_memory(
        SessionMemory(),
        request_id="req-black",
        product_plan=first_plan,
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    refs = {item.field: item.memory_id for item in memory.product.hard_constraints}
    follow_up = ProductQueryPlan.model_validate(
        {
            "query": "ya da qirmizi",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": list(refs.values()),
            "filter_expression": {
                "kind": "all_of",
                "expressions": [
                    *[
                        {
                            "kind": "predicate",
                            "predicate": {
                                "field": item.field,
                                "operator": item.operator,
                                "value": item.value,
                                "strength": item.strength,
                                "unit": item.unit,
                                "evidence_text": "əvvəlki şərt",
                                "memory_refs": [item.memory_id],
                            },
                        }
                        for item in memory.product.hard_constraints
                        if item.field != "color_code"
                    ],
                    {
                        "kind": "any_of",
                        "expressions": [
                            {
                                "kind": "predicate",
                                "predicate": {
                                    "field": "color_code",
                                    "operator": "eq",
                                    "value": "black",
                                    "strength": "hard",
                                    "evidence_text": "Qara",
                                    "memory_refs": [refs["color_code"]],
                                },
                            },
                            {
                                "kind": "predicate",
                                "predicate": {
                                    "field": "color_code",
                                    "operator": "eq",
                                    "value": "red",
                                    "strength": "hard",
                                    "evidence_text": "qirmizi",
                                },
                            },
                        ],
                    },
                ],
            },
            "context_memory": memory_context_payload(memory),
        }
    )

    compilation = compile_semantic_plan(follow_up, catalog)

    assert compilation.clarification is None
    assert compilation.arguments[0].color_code == "black"
    assert compilation.arguments[0].storage_gb == 128
    assert compilation.unavailable_requested_values[0]["value"] == "red"
    assert {item["evidence_source"] for item in compilation.evidence_validation} == {
        "current_message",
        "session_memory",
    }


def test_model_family_is_persisted_as_facet_anchor_and_reused_by_follow_up() -> None:
    catalog = _catalog()
    first_plan = ProductQueryPlan.model_validate(
        {
            "query": "Qara 128 GB iPhone goster",
            "operation": "discover",
            "entities": [
                {
                    "entity_id": "iphone_family",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                    "identifier_type": "model_family",
                }
            ],
            "filter_expression": {
                "kind": "all_of",
                "expressions": [
                    _predicate("category_id", "smartphones", "iPhone"),
                    _predicate("model_family", "iPhone", "iPhone"),
                    _predicate("storage_gb", 128, "128 GB", unit="GB"),
                    _predicate("color_code", "black", "Qara"),
                ],
            },
        }
    )
    first_compilation = compile_semantic_plan(first_plan, catalog)
    assert first_compilation.clarification is None
    assert first_compilation.resolutions[0].status == "constraint_entity"
    resolution = first_compilation.resolutions[0]
    first_result = {
        "status": "success",
        "match_status": "matching_products",
        "items": [],
        "resolved_entities": [
            {
                "entity_id": resolution.entity_id,
                "status": resolution.status,
                "product_id": resolution.product_id,
                "constraint_field": resolution.constraint_field,
                "constraint_value": resolution.constraint_value,
            }
        ],
        "display_product_ids": [],
        "constraint_conflicts": [],
    }
    memory = update_session_memory(
        SessionMemory(),
        request_id="req-black-iphone",
        product_plan=first_plan.model_dump(mode="json", exclude_none=True),
        product_result=first_result,
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory

    assert memory.version == 4
    assert len(memory.product.entities) == 1
    anchor = memory.product.entities[0]
    assert anchor.anchor_kind == "facet"
    assert (anchor.facet_field, anchor.facet_value) == ("model_family", "iPhone")

    refs = {item.field: item.memory_id for item in memory.product.hard_constraints}
    follow_up = ProductQueryPlan.model_validate(
        {
            "query": "bes qirmizi var?",
            "operation": "discover",
            "memory_action": "merge",
            "entities": [
                {
                    "entity_id": "iphone_family",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                    "identifier_type": "model_family",
                    "memory_refs": [anchor.memory_id],
                }
            ],
            "referenced_memory_ids": [anchor.memory_id, *refs.values()],
            "removed_memory_ids": [refs["color_code"]],
            "memory_removals": [
                {"memory_id": refs["color_code"], "evidence_text": "qirmizi"}
            ],
            "filter_expression": {
                "kind": "all_of",
                "expressions": [
                    _predicate(
                        "category_id",
                        "smartphones",
                        "evvelki sert",
                        memory_refs=[refs["category_id"]],
                    ),
                    _predicate(
                        "model_family",
                        "iPhone",
                        "iPhone",
                        memory_refs=[refs["model_family"]],
                    ),
                    _predicate(
                        "storage_gb",
                        128,
                        "128 GB",
                        unit="GB",
                        memory_refs=[refs["storage_gb"]],
                    ),
                    _predicate("color_code", "red", "qirmizi"),
                ],
            },
            "context_memory": memory_context_payload(memory),
        }
    )
    compilation = compile_semantic_plan(follow_up, catalog)

    assert compilation.clarification is None
    assert compilation.unavailable_requested_values[0]["value"] == "red"
    entity_evidence = next(
        item
        for item in compilation.evidence_validation
        if item["source"] == "entity:iphone_family"
    )
    assert entity_evidence["typed_memory_valid"] is True
    assert entity_evidence["canonical_memory_anchors"][0]["anchor_kind"] == "facet"


def test_v2_predicate_backed_entity_requires_typed_anchor_clarification() -> None:
    catalog = _catalog()
    first_memory = update_session_memory(
        SessionMemory(),
        request_id="req-v2",
        product_plan={
            "query": "Qara 128 GB iPhone",
            "operation": "discover",
            "filter_expression": {
                "kind": "all_of",
                "expressions": [
                    _predicate("model_family", "iPhone", "iPhone"),
                    _predicate("color_code", "black", "Qara"),
                ],
            },
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    old_payload = first_memory.model_dump(mode="json", exclude_none=True)
    old_payload["version"] = 2
    old_payload["confirmed_state"]["entities"] = []
    legacy_memory = load_session_memory({"memory": old_payload})
    refs = {item.field: item.memory_id for item in legacy_memory.product.hard_constraints}

    def follow_up(reference_id: str) -> ProductQueryPlan:
        return ProductQueryPlan.model_validate(
            {
                "query": "bes qirmizi var?",
                "operation": "discover",
                "memory_action": "merge",
                "entities": [
                    {
                        "entity_id": "iphone_family",
                        "raw_text": "iPhone",
                        "evidence_text": "iPhone",
                        "identifier_type": "model_family",
                        "memory_refs": [reference_id],
                    }
                ],
                "referenced_memory_ids": [reference_id],
                "filter_expression": _predicate("color_code", "red", "qirmizi"),
                "context_memory": memory_context_payload(legacy_memory),
            }
        )

    compatible = compile_semantic_plan(follow_up(refs["model_family"]), catalog)
    assert compatible.clarification is not None
    assert "invalid_memory_reference" in compatible.clarification["reason"]
    assert compatible.clarification["memory_issue"]["reference_type_mismatches"] == [
        {
            "source": "entity:iphone_family",
            "memory_id": refs["model_family"],
            "expected_kind": "entity",
            "actual_kind": "predicate",
        }
    ]

    unrelated = compile_semantic_plan(follow_up(refs["color_code"]), catalog)
    assert unrelated.clarification is not None
    assert "invalid_memory_reference" in unrelated.clarification["reason"]
    assert unrelated.clarification["memory_issue"]["invalid_inherited_entities"] == [
        "iphone_family"
    ]


def test_direct_or_failed_run_does_not_change_memory() -> None:
    before = _seed_memory()
    update = update_session_memory(
        before,
        request_id="req-2",
        product_plan=None,
        product_result=None,
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert update.memory == before
    assert update.transition["action"] == "preserve"
    assert update.transition["revision_after"] == before.revision


def test_memory_backed_evidence_is_accepted_only_with_current_session_id() -> None:
    catalog = _catalog()
    memory = _seed_memory()
    memory_id = next(iter(memory_reference_map(memory)))
    payload = {
        "query": "stokdadırmı?",
        "operation": "lookup",
        "memory_action": "preserve",
        "referenced_memory_ids": [memory_id],
        "entities": [
            {
                "entity_id": "phone",
                "raw_text": "iPhone 17 Pro",
                "evidence_text": "iPhone 17 Pro",
                "memory_refs": [memory_id],
                "context_product_id": "prd_smartphones_001",
            }
        ],
        "selection_expression": {"kind": "entity_ref", "entity_id": "phone"},
        "fact_questions": [
            {
                "field": "stock_status",
                "evidence_text": "stokdadırmı",
            }
        ],
        "context_product_ids": ["prd_smartphones_001"],
        "context_memory": memory_context_payload(memory),
    }
    valid = compile_semantic_plan(ProductQueryPlan.model_validate(payload), catalog)

    payload["referenced_memory_ids"] = ["mem_entity_forged"]
    payload["entities"][0]["memory_refs"] = ["mem_entity_forged"]
    forged = compile_semantic_plan(ProductQueryPlan.model_validate(payload), catalog)

    assert valid.clarification is None
    assert valid.evidence_validation[0]["evidence_source"] == "session_memory"
    assert forged.clarification is not None
    assert "invalid_memory_reference" in forged.clarification["reason"]


def test_two_turn_product_replacement_may_reference_and_remove_same_memory_id() -> None:
    catalog = _catalog()
    memory = _seed_memory()
    memory_id = memory.product.entities[0].memory_id
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Yox, Pro Max olsun.",
            "operation": "lookup",
            "memory_action": "merge",
            "referenced_memory_ids": [memory_id],
            "removed_memory_ids": [memory_id],
            "memory_removals": [
                {"memory_id": memory_id, "evidence_text": "Pro Max olsun"}
            ],
            "entities": [
                {
                    "entity_id": "old",
                    "raw_text": "iPhone 17 Pro",
                    "state": "superseded",
                    "evidence_text": "Yox",
                    "identifier_type": "model",
                    "memory_refs": [memory_id],
                    "context_product_id": "prd_smartphones_001",
                },
                {
                    "entity_id": "new",
                    "raw_text": "Pro Max",
                    "state": "selected",
                    "supersedes_entity_id": "old",
                    "evidence_text": "Pro Max",
                    "identifier_type": "model",
                    "memory_refs": [memory_id],
                    "context_product_id": "prd_smartphones_001",
                },
            ],
            "selection_expression": {"kind": "entity_ref", "entity_id": "new"},
            "context_product_ids": ["prd_smartphones_001"],
            "context_memory": memory_context_payload(memory),
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.arguments[0].product_id == "prd_smartphones_002"


def test_hard_memory_constraint_cannot_be_downgraded_to_preference() -> None:
    catalog = _catalog()
    seeded = update_session_memory(
        SessionMemory(),
        request_id="req-hard",
        product_plan={
            "query": "1000 AZN-dən ucuz telefon",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 1000,
                    "unit": "AZN",
                    "strength": "hard",
                    "evidence_text": "1000 AZN-dən ucuz",
                },
            },
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    predicate_id = seeded.product.hard_constraints[0].memory_id
    plan = ProductQueryPlan.model_validate(
        {
            "query": "bu sadəcə üstünlük olsun",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": [predicate_id],
            "preference_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 1000,
                    "unit": "AZN",
                    "strength": "preference",
                    "evidence_text": "sadəcə üstünlük",
                    "memory_refs": [predicate_id],
                },
            },
            "context_memory": memory_context_payload(seeded),
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is not None
    assert predicate_id in compilation.clarification["memory_issue"]["hard_constraint_downgrades"]


def test_inherited_constraint_cannot_change_value_without_current_evidence() -> None:
    catalog = _catalog()
    seeded = update_session_memory(
        SessionMemory(),
        request_id="req-hard",
        product_plan={
            "query": "1000 AZN-dən ucuz telefon",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 1000,
                    "unit": "AZN",
                    "strength": "hard",
                    "evidence_text": "1000 AZN-dən ucuz",
                },
            },
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    predicate_id = seeded.product.hard_constraints[0].memory_id
    plan = ProductQueryPlan.model_validate(
        {
            "query": "telefon göstər",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": [predicate_id],
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 500,
                    "unit": "AZN",
                    "strength": "hard",
                    "evidence_text": "1000 AZN-dən ucuz",
                    "memory_refs": [predicate_id],
                },
            },
            "context_memory": memory_context_payload(seeded),
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is not None
    assert predicate_id in compilation.clarification["memory_issue"]["mutated_inherited_predicates"]


def test_memory_removal_requires_exact_current_message_evidence() -> None:
    with pytest.raises(ValidationError):
        ProductQueryPlan.model_validate(
            {
                "query": "büdcə şərtini çıxar",
                "operation": "discover",
                "memory_action": "merge",
                "removed_memory_ids": ["mem_predicate_123"],
            }
        )


def test_grounded_memory_removal_deletes_only_the_named_constraint() -> None:
    seeded = update_session_memory(
        SessionMemory(),
        request_id="req-hard",
        product_plan={
            "query": "1000 AZN-dən ucuz telefon",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 1000,
                    "unit": "AZN",
                    "strength": "hard",
                    "evidence_text": "1000 AZN-dən ucuz",
                },
            },
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    predicate_id = seeded.product.hard_constraints[0].memory_id
    removal_plan = {
        "query": "büdcə şərtini çıxar",
        "operation": "discover",
        "memory_action": "merge",
        "removed_memory_ids": [predicate_id],
        "memory_removals": [
            {
                "memory_id": predicate_id,
                "evidence_text": "büdcə şərtini çıxar",
            }
        ],
    }

    compilation = compile_semantic_plan(
        ProductQueryPlan.model_validate(
            {**removal_plan, "context_memory": memory_context_payload(seeded)}
        ),
        _catalog(),
    )
    update = update_session_memory(
        seeded,
        request_id="req-remove",
        product_plan=removal_plan,
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert compilation.clarification is None
    assert update.memory.product.hard_constraints == []
    assert predicate_id in update.transition["removed_ids"]


def test_pending_root_marks_follow_up_context_without_grounding_nested_facts() -> None:
    first_plan = {
        "query": "iPhone 19 göstər",
        "operation": "lookup",
        "entities": [
            {
                "entity_id": "missing",
                "raw_text": "iPhone 19",
                "state": "selected",
                "evidence_text": "iPhone 19",
                "identifier_type": "model",
            }
        ],
        "selection_expression": {"kind": "entity_ref", "entity_id": "missing"},
    }
    memory = update_session_memory(
        SessionMemory(),
        request_id="req-missing",
        product_plan=first_plan,
        product_result={
            "status": "success",
            "match_status": "alternatives",
            "items": [
                {"product_id": "prd_smartphones_003", "name": "Apple iPhone 17"}
            ],
            "display_product_ids": ["prd_smartphones_003"],
            "recommended_product_id": "prd_smartphones_003",
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    assert memory.pending_clarification is not None
    pending_id = memory.pending_clarification.memory_id
    assert pending_id
    assert pending_id in memory_reference_map(memory)

    follow_up = ProductQueryPlan.model_validate(
        {
            "query": "Onda Samsung göstər.",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": [pending_id],
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "brand",
                    "operator": "eq",
                    "value": "Samsung",
                    "strength": "hard",
                    "evidence_text": "Samsung",
                },
            },
            "context_memory": memory_context_payload(memory),
        }
    )

    compilation = compile_semantic_plan(follow_up, _catalog())

    assert compilation.clarification is None
    assert compilation.arguments[0].brand == "Samsung"
    assert compilation.arguments[0].category_id is None


def test_pending_root_cannot_substitute_for_predicate_memory_id() -> None:
    pending = update_session_memory(
        SessionMemory(),
        request_id="req-pending-root",
        product_plan={"query": "olmayan model", "operation": "lookup"},
        product_result={
            "status": "success",
            "match_status": "not_found",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    assert pending.pending_intent is not None
    pending_id = pending.pending_intent.memory_id
    assert pending_id
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Samsung göstər",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": [pending_id],
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "brand",
                    "operator": "eq",
                    "value": "Samsung",
                    "strength": "hard",
                    "evidence_text": "Samsung",
                    "memory_refs": [pending_id],
                },
            },
            "context_memory": memory_context_payload(pending),
        }
    )

    compilation = compile_semantic_plan(plan, _catalog())

    assert compilation.clarification is not None
    assert compilation.clarification["memory_issue"]["reference_type_mismatches"] == [
        {
            "source": "predicate:brand",
            "memory_id": pending_id,
            "expected_kind": "predicate",
            "actual_kind": "pending_intent",
        }
    ]


def test_not_found_pending_state_keeps_failed_constraint_for_later_revision() -> None:
    seed_plan = {
        "query": "1000 AZN-dən ucuz Samsung telefon göstər",
        "operation": "discover",
        "memory_action": "replace",
        "filter_expression": {
            "kind": "all_of",
            "expressions": [
                _predicate("category_id", "smartphones", "telefon"),
                _predicate("brand", "Samsung", "Samsung"),
                _predicate("price", 1000, "1000 AZN", operator="lt", unit="AZN"),
            ],
        },
    }
    confirmed = update_session_memory(
        SessionMemory(),
        request_id="req-seed",
        product_plan=seed_plan,
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    confirmed_refs = {item.field: item.memory_id for item in confirmed.product.hard_constraints}
    white_plan = {
        "query": "Ağ olsun",
        "operation": "discover",
        "memory_action": "merge",
        "referenced_memory_ids": list(confirmed_refs.values()),
        "filter_expression": {
            "kind": "all_of",
            "expressions": [
                _predicate(
                    item.field,
                    item.value,
                    "əvvəlki şərt",
                    operator=item.operator,
                    unit=item.unit,
                    memory_refs=[item.memory_id],
                )
                for item in confirmed.product.hard_constraints
            ]
            + [_predicate("color_code", "white", "Ağ")],
        },
    }
    pending = update_session_memory(
        confirmed,
        request_id="req-white",
        product_plan=white_plan,
        product_result={
            "status": "success",
            "match_status": "not_found",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory

    assert pending.product == confirmed.product
    assert pending.pending_intent is not None
    pending_fields = {
        item.field: item for item in pending.pending_intent.state.hard_constraints
    }
    assert set(pending_fields) == {"category_id", "brand", "price", "color_code"}
    assert "rəng = white" in pending.continuation_summary
    assert pending_fields["color_code"].memory_id in memory_reference_map(pending)

    revised_plan = {
        "query": "Büdcəni 2000 AZN et",
        "operation": "discover",
        "memory_action": "merge",
        "referenced_memory_ids": [item.memory_id for item in pending_fields.values()],
        "removed_memory_ids": [pending_fields["price"].memory_id],
        "memory_removals": [
            {
                "memory_id": pending_fields["price"].memory_id,
                "evidence_text": "Büdcəni 2000 AZN et",
            }
        ],
        "filter_expression": {
            "kind": "all_of",
            "expressions": [
                *[
                    _predicate(
                        item.field,
                        item.value,
                        "pending şərt",
                        operator=item.operator,
                        unit=item.unit,
                        memory_refs=[item.memory_id],
                    )
                    for field, item in pending_fields.items()
                    if field != "price"
                ],
                _predicate("price", 2000, "2000 AZN", operator="lt", unit="AZN"),
            ],
        },
    }
    revised = update_session_memory(
        pending,
        request_id="req-budget",
        product_plan=revised_plan,
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory
    revised_fields = {item.field: item.value for item in revised.product.hard_constraints}

    assert revised.pending_intent is None
    assert revised_fields["color_code"] == "white"
    assert revised_fields["price"] == 2000


def test_v1_memory_is_upgraded_and_summary_is_bounded() -> None:
    loaded = load_session_memory(
        {
            "memory": {
                "version": 1,
                "revision": 3,
                "product": {},
                "pending_clarification": {
                    "reason": "not_found",
                    "operation": "discover",
                    "raw_entities": ["future phone"],
                },
            }
        }
    )

    assert loaded.version == 4
    assert loaded.pending_intent is not None
    assert loaded.pending_intent.state == loaded.product
    assert loaded.continuation_summary == ""


def test_pending_intent_keeps_full_fact_question_shape() -> None:
    update = update_session_memory(
        SessionMemory(),
        request_id="req-fact-pending",
        product_plan={
            "query": "Bu model 500 AZN-dən aşağıdırmı?",
            "operation": "lookup",
            "memory_action": "replace",
            "fact_questions": [
                {
                    "field": "price",
                    "operator": "lt",
                    "value": 500,
                    "unit": "AZN",
                    "evidence_text": "500 AZN-dən aşağıdırmı",
                }
            ],
        },
        product_result={
            "status": "success",
            "match_status": "not_found",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    )

    assert update.memory.pending_intent is not None
    question = update.memory.pending_intent.state.fact_questions[0]
    assert (question.field, question.operator, question.value, question.unit) == (
        "price",
        "lt",
        500,
        "AZN",
    )
    assert memory_reference_map(update.memory)[question.memory_id]["kind"] == "fact_question"


def test_directional_ranking_objective_is_typed_and_reusable_in_memory() -> None:
    objective = {
        "field": "display_size_in",
        "direction": "maximize",
        "priority": "primary",
        "origin": "explicit",
        "evidence_text": "böyük ekran",
    }
    first = update_session_memory(
        SessionMemory(),
        request_id="req-ranking",
        product_plan={
            "query": "Böyük ekranlı smartfon göstər",
            "operation": "discover",
            "ranking_objectives": [objective],
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
            "ranking_objectives": [objective],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory

    stored = first.product.ranking_objectives[0]
    reference = memory_reference_map(first)[stored.memory_id]

    assert first.version == 4
    assert reference["kind"] == "ranking_objective"
    assert reference["field"] == "display_size_in"
    assert "display_size_in" in first.continuation_summary
    assert memory_context_payload(first)["confirmed_state"]["ranking_objectives"][0][
        "memory_id"
    ] == stored.memory_id

    reused = update_session_memory(
        first,
        request_id="req-ranking-follow-up",
        product_plan={
            "query": "Bu meyar qalsın",
            "operation": "discover",
            "memory_action": "merge",
            "referenced_memory_ids": [stored.memory_id],
            "ranking_objectives": [
                {
                    **objective,
                    "origin": "memory",
                    "evidence_text": "əvvəlki məqsəd",
                    "memory_refs": [stored.memory_id],
                }
            ],
        },
        product_result={
            "status": "success",
            "match_status": "matching_products",
            "items": [],
            "display_product_ids": [],
            "constraint_conflicts": [],
            "ranking_objectives": [
                {
                    **objective,
                    "origin": "memory",
                    "evidence_text": "əvvəlki məqsəd",
                    "memory_refs": [stored.memory_id],
                }
            ],
        },
        document_arguments=None,
        document_result=None,
        max_bytes=8192,
    ).memory

    assert reused.product.ranking_objectives[0].memory_id == stored.memory_id
    assert reused.product.ranking_objectives[0].origin == "explicit"


def _predicate(
    field: str,
    value: object,
    evidence: str,
    *,
    operator: str = "eq",
    unit: str | None = None,
    memory_refs: list[str] | None = None,
) -> dict:
    predicate = {
        "field": field,
        "operator": operator,
        "value": value,
        "strength": "hard",
        "evidence_text": evidence,
    }
    if unit is not None:
        predicate["unit"] = unit
    if memory_refs:
        predicate["memory_refs"] = memory_refs
    return {"kind": "predicate", "predicate": predicate}


def test_decision_explanation_is_deterministic_and_contains_no_provider_reasoning() -> None:
    transition = {
        "revision_before": 0,
        "revision_after": 1,
        "action": "replace",
        "changed_ids": ["mem_entity_1"],
        "removed_ids": [],
        "size_bytes": 512,
    }
    kwargs = {
        "history_message_count": 2,
        "memory_before_revision": 0,
        "product_plan": _exact_plan(),
        "product_result": _exact_result(),
        "product_retrieval": {"semantic_state": "exact"},
        "document_arguments": None,
        "document_result": None,
        "document_retrieval": None,
        "used_tools": ["product_search"],
        "warnings": [],
        "memory_transition": transition,
    }

    first = build_decision_explanation(**kwargs)
    second = build_decision_explanation(**kwargs)
    encoded = json.dumps(first, ensure_ascii=False).lower()

    assert first == second
    assert first["version"] == 2
    assert first["basis"] == "product_search"
    assert first["narrative"].startswith("İstifadəçi")
    assert "reasoning_details" not in encoded
    assert "system prompt" not in encoded
    assert "raw vector" not in encoded


def test_decision_explanation_uses_resolved_names_instead_of_raw_context_reference() -> None:
    explanation = build_decision_explanation(
        history_message_count=4,
        memory_before_revision=2,
        product_plan={
            "query": "Bunlardan hansının stoku var?",
            "operation": "compare",
            "memory_action": "merge",
            "entities": [
                {
                    "entity_id": "first",
                    "raw_text": "Bunlardan",
                    "state": "selected",
                    "memory_refs": ["mem_entity_first"],
                },
                {
                    "entity_id": "second",
                    "raw_text": "Bunlardan",
                    "state": "selected",
                    "memory_refs": ["mem_entity_second"],
                },
            ],
            "fact_questions": [{"field": "stock_status", "evidence_text": "stoku"}],
        },
        product_result={
            "status": "success",
            "match_status": "exact_match",
            "items": [
                {"product_id": "prd_phone_1", "name": "Apple iPhone 17 Pro"},
                {"product_id": "prd_phone_2", "name": "Samsung Galaxy S25 Ultra"},
            ],
            "resolved_entities": [
                {"entity_id": "first", "product_id": "prd_phone_1"},
                {"entity_id": "second", "product_id": "prd_phone_2"},
            ],
            "display_product_ids": ["prd_phone_1", "prd_phone_2"],
        },
        product_retrieval={"retrieval_executed": True},
        document_arguments=None,
        document_result=None,
        document_retrieval=None,
        used_tools=["product_search"],
        warnings=[],
        memory_transition={
            "revision_before": 2,
            "revision_after": 3,
            "action": "merge",
            "changed_ids": [],
            "removed_ids": [],
            "size_bytes": 700,
        },
    )

    assert "Bunlardan" not in explanation["narrative"]
    assert "Apple iPhone 17 Pro" in explanation["narrative"]
    assert "Samsung Galaxy S25 Ultra" in explanation["narrative"]


def test_clarification_narrative_says_search_did_not_start() -> None:
    explanation = build_decision_explanation(
        history_message_count=2,
        memory_before_revision=1,
        memory_context_enabled=False,
        product_plan={
            "query": "ya da qirmizi",
            "operation": "discover",
            "memory_action": "merge",
            "entities": [
                {
                    "entity_id": "iphone_red",
                    "raw_text": "qirmizi 128 GB iPhone",
                    "state": "selected",
                    "identifier_type": "model_family",
                }
            ],
        },
        product_result={
            "status": "success",
            "match_status": "clarification_required",
            "clarification": {
                "reason": "evidence_not_grounded",
                "question": "Sorğunu dəqiqləşdirə bilərsiniz?",
            },
            "display_product_ids": [],
        },
        product_retrieval={
            "semantic_state": "clarification_required",
            "retrieval_executed": False,
        },
        document_arguments=None,
        document_result=None,
        document_retrieval=None,
        used_tools=["product_search"],
        warnings=[],
        memory_transition={
            "revision_before": 1,
            "revision_after": 2,
            "action": "merge",
            "changed_ids": ["pending_clarification"],
            "removed_ids": [],
            "size_bytes": 1000,
        },
    )

    assert "real məhsul axtarışı başlamadı" in explanation["narrative"]
    assert "məhsul tapılmadı" not in explanation["narrative"].casefold()


def test_invalid_stored_memory_falls_back_to_empty_v2() -> None:
    loaded = load_session_memory({"memory": {"version": 1, "unknown": "field"}})

    assert loaded == SessionMemory()
