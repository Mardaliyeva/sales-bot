from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError
from qdrant_client import models

from app.agent.presentation import build_product_cards
from app.config import PROJECT_ROOT
from app.embeddings.azure import DEFAULT_TEXT_VERSION
from app.evals.semantic_plans import semantic_signature
from app.retrieval.qdrant import QdrantProductSearch
from app.retrieval.semantic_plan import compile_semantic_plan, iter_predicates
from app.tools.catalog import ProductCatalog
from app.tools.product_search import ProductSearchTool
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    ProductEntity,
    ProductQueryPlan,
    ProductSearchArguments,
    SemanticExpression,
)
from app.vectorstores.qdrant import QdrantProductStore, VectorSearchHit


@pytest.fixture(scope="module")
def catalog() -> ProductCatalog:
    loaded = ProductCatalog(PROJECT_ROOT / "data" / "catalog" / "products.jsonl")
    loaded.load()
    return loaded


def test_golden_semantic_plans_are_schema_valid_and_evidence_grounded(
    catalog: ProductCatalog,
) -> None:
    path = PROJECT_ROOT / "data" / "evals" / "semantic_query_plans.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) >= 10
    for case in cases:
        plan = ProductQueryPlan.model_validate(case["plan"])
        compilation = compile_semantic_plan(plan, catalog)
        assert all(item["valid"] for item in compilation.evidence_validation), case["id"]
        assert compilation.canonical_hash


def test_semantic_eval_signature_is_independent_of_entity_id_spelling() -> None:
    left = ProductQueryPlan.model_validate(
        {
            "query": "A",
            "operation": "lookup",
            "entities": [{"entity_id": "chosen", "raw_text": "A", "evidence_text": "A"}],
            "selection_expression": {"kind": "entity_ref", "entity_id": "chosen"},
        }
    )
    right = ProductQueryPlan.model_validate(
        {
            "query": "A",
            "operation": "lookup",
            "entities": [{"entity_id": "e7", "raw_text": "A", "evidence_text": "A"}],
            "selection_expression": {"kind": "entity_ref", "entity_id": "e7"},
        }
    )

    assert semantic_signature(left) == semantic_signature(right)


def test_selection_revision_resolves_only_current_entity(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro fikrindən daşındım, iPhone 17 Pro Max olsun",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "old",
                    "raw_text": "iPhone 17 Pro",
                    "state": "superseded",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "new",
                    "raw_text": "iPhone 17 Pro Max",
                    "state": "selected",
                    "supersedes_entity_id": "old",
                    "evidence_text": "iPhone 17 Pro Max",
                    "identifier_type": "model",
                },
            ],
            "selection_expression": {"kind": "entity_ref", "entity_id": "new"},
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert [item.entity_id for item in compilation.resolutions] == ["new"]
    assert compilation.arguments[0].product_id == "prd_smartphones_002"


def test_shortened_superseding_entity_uses_catalog_identity_context(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro yox, Pro Max olsun",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "old",
                    "raw_text": "iPhone 17 Pro",
                    "state": "superseded",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "new",
                    "raw_text": "Pro Max",
                    "state": "selected",
                    "supersedes_entity_id": "old",
                    "evidence_text": "Pro Max",
                    "identifier_type": "model",
                },
            ],
            "selection_expression": {"kind": "entity_ref", "entity_id": "new"},
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.resolutions[0].reason == "supersession_catalog_context"
    assert compilation.arguments[0].product_id == "prd_smartphones_002"


def test_superseded_entity_refs_are_pruned_from_canonical_selection(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro əvəzinə iPhone 17 Pro Max",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "old",
                    "raw_text": "iPhone 17 Pro",
                    "state": "superseded",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "new",
                    "raw_text": "iPhone 17 Pro Max",
                    "state": "selected",
                    "supersedes_entity_id": "old",
                    "evidence_text": "iPhone 17 Pro Max",
                    "identifier_type": "model",
                },
            ],
            "selection_expression": {
                "kind": "all_of",
                "expressions": [
                    {"kind": "entity_ref", "entity_id": "new"},
                    {"kind": "entity_ref", "entity_id": "old"},
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.plan.selection_expression is not None
    assert compilation.plan.selection_expression.kind == "entity_ref"
    assert compilation.plan.selection_expression.entity_id == "new"
    assert compilation.arguments[0].product_id == "prd_smartphones_002"


def test_typo_resolution_does_not_rewrite_numeric_model_identifier(
    catalog: ProductCatalog,
) -> None:
    typo = catalog.resolve_entity(
        ProductEntity(
            entity_id="typo",
            raw_text="iPhnoe 17 Pro",
            evidence_text="iPhnoe 17 Pro",
            identifier_type="model",
        )
    )
    missing = catalog.resolve_entity(
        ProductEntity(
            entity_id="missing",
            raw_text="iPhone 19",
            evidence_text="iPhone 19",
            identifier_type="model",
        )
    )

    assert typo.status == "resolved"
    assert typo.product_id == "prd_smartphones_001"
    assert missing.status == "unresolved"


def test_fact_question_does_not_compile_to_filter(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro 500-dən aşağıdırmı?",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "phone",
                    "raw_text": "iPhone 17 Pro",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                }
            ],
            "fact_questions": [
                {
                    "field": "price",
                    "operator": "lte",
                    "value": 500,
                    "unit": "AZN",
                    "evidence_text": "500-dən aşağıdırmı",
                }
            ],
        }
    )

    arguments = compile_semantic_plan(plan, catalog).arguments[0]

    assert arguments.requested_fields == ["price"]
    assert arguments.max_price is None
    assert arguments.semantic_filter_expression is None


def test_hard_price_constraint_compiles_to_qdrant_filter(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "500 AZN-dən aşağı məhsul",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "price",
                    "operator": "lte",
                    "value": 500,
                    "unit": "AZN",
                    "strength": "hard",
                    "evidence_text": "500 AZN-dən aşağı",
                },
            },
        }
    )

    arguments = compile_semantic_plan(plan, catalog).arguments[0]
    query_filter = QdrantProductStore.build_filter(arguments)

    assert arguments.max_price == 500
    assert query_filter is not None and query_filter.must
    sale_price_conditions = [
        condition
        for condition in query_filter.must
        if isinstance(condition, models.FieldCondition) and condition.key == "sale_price"
    ]
    assert len(sale_price_conditions) == 2
    assert all(condition.range and condition.range.lte == 500 for condition in sale_price_conditions)


def test_catalog_metadata_canonicalizes_semantic_facet_value(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "4K televizor",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "resolution",
                    "operator": "eq",
                    "value": "4K",
                    "strength": "hard",
                    "evidence_text": "4K",
                },
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.plan.filter_expression is not None
    assert compilation.plan.filter_expression.predicate is not None
    assert compilation.plan.filter_expression.predicate.value == "4K UHD"
    assert compilation.facet_mapping[0]["canonical"] == "4K UHD"


def test_unique_catalog_namespace_corrects_invalid_facet_hint(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone modelləri",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "brand",
                    "operator": "eq",
                    "value": "iPhone",
                    "strength": "hard",
                    "evidence_text": "iPhone",
                },
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.plan.filter_expression is not None
    assert compilation.plan.filter_expression.predicate is not None
    assert compilation.plan.filter_expression.predicate.field == "model_family"
    assert compilation.facet_mapping[0]["canonical_field"] == "model_family"


def test_mixed_namespace_in_predicate_becomes_any_of(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone və Samsung",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "brand",
                    "operator": "in",
                    "value": ["iPhone", "Samsung"],
                    "strength": "hard",
                    "evidence_text": "iPhone və Samsung",
                },
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.plan.filter_expression is not None
    assert compilation.plan.filter_expression.kind == "any_of"
    predicates = iter_predicates(compilation.plan.filter_expression)
    assert [(item.field, item.operator, item.value) for item in predicates] == [
        ("model_family", "eq", "iPhone"),
        ("brand", "eq", "Samsung"),
    ]


def test_unmapped_hard_facet_value_is_deterministic_unavailable(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "XQZ çözünürlük",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "resolution",
                    "operator": "eq",
                    "value": "XQZ",
                    "strength": "hard",
                    "evidence_text": "XQZ",
                },
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.arguments == ()
    assert compilation.clarification is None
    assert compilation.deterministic_empty is True
    assert compilation.unavailable_requested_values == (
        {
            "field": "resolution",
            "value": "XQZ",
            "reason": "unmapped_required_value",
            "evidence_text": "XQZ",
        },
    )


def test_any_of_and_not_compile_without_language_rules() -> None:
    expression = SemanticExpression.model_validate(
        {
            "kind": "any_of",
            "expressions": [
                {
                    "kind": "predicate",
                    "predicate": {
                        "field": "brand",
                        "operator": "eq",
                        "value": "Apple",
                        "strength": "hard",
                        "evidence_text": "Apple",
                    },
                },
                {
                    "kind": "not",
                    "expression": {
                        "kind": "predicate",
                        "predicate": {
                            "field": "in_stock",
                            "operator": "eq",
                            "value": False,
                            "strength": "hard",
                            "evidence_text": "stok",
                        },
                    },
                },
            ],
        }
    )
    query_filter = QdrantProductStore.build_filter(
        ProductSearchArguments(query="semantic", semantic_filter_expression=expression)
    )

    assert query_filter is not None and query_filter.must
    semantic_filter = query_filter.must[0]
    assert isinstance(semantic_filter, models.Filter)
    assert semantic_filter.should and len(semantic_filter.should) == 2


def test_predicate_only_selection_expression_is_canonicalized_to_filter(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Apple və Samsung",
            "operation": "discover",
            "selection_expression": {
                "kind": "any_of",
                "expressions": [
                    {
                        "kind": "predicate",
                        "predicate": {
                            "field": "brand",
                            "operator": "eq",
                            "value": "Apple",
                            "strength": "hard",
                            "evidence_text": "Apple",
                        },
                    },
                    {
                        "kind": "predicate",
                        "predicate": {
                            "field": "brand",
                            "operator": "eq",
                            "value": "Samsung",
                            "strength": "hard",
                            "evidence_text": "Samsung",
                        },
                    },
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.arguments[0].semantic_filter_expression is not None
    assert compilation.arguments[0].semantic_filter_expression.kind == "any_of"


def test_any_of_entity_refs_compile_to_one_catalog_filter(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone və Samsung",
            "operation": "discover",
            "entities": [
                {
                    "entity_id": "family",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                    "identifier_type": "model_family",
                },
                {
                    "entity_id": "brand",
                    "raw_text": "Samsung",
                    "evidence_text": "Samsung",
                },
            ],
            "selection_expression": {
                "kind": "any_of",
                "expressions": [
                    {"kind": "entity_ref", "entity_id": "family"},
                    {"kind": "entity_ref", "entity_id": "brand"},
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert len(compilation.arguments) == 1
    compiled = compilation.arguments[0].semantic_filter_expression
    assert compiled is not None and compiled.kind == "any_of"
    assert [predicate.value for predicate in iter_predicates(compiled)] == [
        "iPhone",
        "Samsung",
    ]


def test_filter_represented_entities_do_not_add_last_entity_constraint(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone və Samsung",
            "operation": "discover",
            "entities": [
                {"entity_id": "left", "raw_text": "iPhone", "evidence_text": "iPhone"},
                {"entity_id": "right", "raw_text": "Samsung", "evidence_text": "Samsung"},
            ],
            "filter_expression": {
                "kind": "any_of",
                "expressions": [
                    {
                        "kind": "predicate",
                        "predicate": {
                            "field": "brand",
                            "operator": "eq",
                            "value": "iPhone",
                            "strength": "hard",
                            "evidence_text": "iPhone",
                        },
                    },
                    {
                        "kind": "predicate",
                        "predicate": {
                            "field": "brand",
                            "operator": "eq",
                            "value": "Samsung",
                            "strength": "hard",
                            "evidence_text": "Samsung",
                        },
                    },
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert len(compilation.arguments) == 1
    compiled = compilation.arguments[0].semantic_filter_expression
    assert compiled is not None and compiled.kind == "any_of"
    assert [predicate.value for predicate in iter_predicates(compiled)] == [
        "iPhone",
        "Samsung",
    ]


def test_compare_does_not_compile_entity_combinator_as_product_filter(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro və Galaxy S25 Ultra müqayisəsi",
            "operation": "compare",
            "entities": [
                {
                    "entity_id": "left",
                    "raw_text": "iPhone 17 Pro",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "right",
                    "raw_text": "Galaxy S25 Ultra",
                    "evidence_text": "Galaxy S25 Ultra",
                    "identifier_type": "model",
                },
            ],
            "selection_expression": {
                "kind": "all_of",
                "expressions": [
                    {"kind": "entity_ref", "entity_id": "left"},
                    {"kind": "entity_ref", "entity_id": "right"},
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert [argument.product_id for argument in compilation.arguments] == [
        "prd_smartphones_001",
        "prd_smartphones_014",
    ]
    assert all(
        argument.semantic_filter_expression is None
        for argument in compilation.arguments
    )


def test_eval_signature_ignores_non_executable_details_after_clarification() -> None:
    empty = ProductQueryPlan(
        query="əvvəlki model",
        operation="lookup",
        needs_clarification=True,
        clarification_question="Hansını nəzərdə tutursunuz?",
    )
    with_unresolved_entity = ProductQueryPlan.model_validate(
        {
            "query": "əvvəlki model",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "previous",
                    "raw_text": "əvvəlki model",
                    "evidence_text": "əvvəlki model",
                }
            ],
            "selection_expression": {"kind": "entity_ref", "entity_id": "previous"},
            "needs_clarification": True,
            "clarification_question": "Hansını nəzərdə tutursunuz?",
        }
    )

    assert semantic_signature(empty) == semantic_signature(with_unresolved_entity)


def test_nested_fallback_compiles_to_ordered_query_branches(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "telefon üçün əvvəl Apple, alınmasa Samsung",
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
                            "evidence_text": "telefon",
                        },
                    },
                    {
                        "kind": "fallback",
                        "primary": {
                            "kind": "predicate",
                            "predicate": {
                                "field": "brand",
                                "operator": "eq",
                                "value": "Apple",
                                "strength": "hard",
                                "evidence_text": "Apple",
                            },
                        },
                        "secondary": {
                            "kind": "predicate",
                            "predicate": {
                                "field": "brand",
                                "operator": "eq",
                                "value": "Samsung",
                                "strength": "hard",
                                "evidence_text": "Samsung",
                            },
                        },
                    },
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)
    brands = [
        next(
            predicate.value
            for predicate in iter_predicates(arguments.semantic_filter_expression)
            if predicate.field == "brand"
        )
        for arguments in compilation.arguments
    ]

    assert brands == ["Apple", "Samsung"]


def test_fallback_entities_resolve_to_catalog_facets_without_product_scan(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone alınmasa Samsung",
            "operation": "discover",
            "entities": [
                {
                    "entity_id": "primary",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                    "identifier_type": "model_family",
                },
                {
                    "entity_id": "secondary",
                    "raw_text": "Samsung",
                    "evidence_text": "Samsung",
                },
            ],
            "selection_expression": {
                "kind": "fallback",
                "primary": {"kind": "entity_ref", "entity_id": "primary"},
                "secondary": {"kind": "entity_ref", "entity_id": "secondary"},
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert [resolution.constraint_field for resolution in compilation.resolutions] == [
        "model_family",
        "brand",
    ]
    assert [
        iter_predicates(argument.semantic_filter_expression)[0].value
        for argument in compilation.arguments
    ] == ["iPhone", "Samsung"]


def test_ambiguous_entity_requires_clarification(catalog: ProductCatalog) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone göstər",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "phone",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                    "identifier_type": "model",
                }
            ],
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.arguments == ()
    assert compilation.clarification is not None
    assert compilation.clarification["reason"] == "ambiguous_entity"


def test_plan_limits_are_enforced() -> None:
    with pytest.raises(ValidationError):
        ProductQueryPlan.model_validate(
            {
                "query": "dörd məhsul",
                "operation": "compare",
                "entities": [
                    {"entity_id": str(index), "raw_text": str(index), "evidence_text": str(index)}
                    for index in range(4)
                ],
            }
        )


def test_production_semantic_runtime_has_no_phrase_marker_branch() -> None:
    production_files = [
        PROJECT_ROOT / "app" / "retrieval" / "semantic_plan.py",
        PROJECT_ROOT / "app" / "retrieval" / "qdrant.py",
        PROJECT_ROOT / "app" / "tools" / "catalog.py",
    ]
    forbidden = ("marker in query", "if phrase in", "phrase_to_operator")

    for path in production_files:
        source = path.read_text(encoding="utf-8").casefold()
        assert all(value not in source for value in forbidden), path


def test_generated_80k_catalog_resolution_uses_prebuilt_indexes() -> None:
    catalog = ProductCatalog(PROJECT_ROOT / "data" / "catalog" / "products.jsonl")
    products = [
        {
            "product_id": f"generated_{index:05d}",
            "sku": f"SKU-{index:05d}",
            "model": f"Model {index:05d}",
            "name": f"Brand Model {index:05d}",
            "attributes": {},
        }
        for index in range(80_000)
    ]
    catalog._build_resolution_indexes(products)  # noqa: SLF001 - scale contract test

    class NoRequestScan(list[dict[str, object]]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("per-request catalog scan is forbidden")

    catalog.products = NoRequestScan(products)
    resolution = catalog.resolve_entity(
        ProductEntity(
            entity_id="target",
            raw_text="Model 79999",
            evidence_text="Model 79999",
            identifier_type="model",
        )
    )
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Model 79999",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "target",
                    "raw_text": "Model 79999",
                    "evidence_text": "Model 79999",
                    "identifier_type": "model",
                }
            ],
        }
    )
    compilation = compile_semantic_plan(plan, catalog)

    assert resolution.status == "resolved"
    assert resolution.product_id == "generated_79999"
    assert compilation.arguments[0].semantic_plan_compiled is True


def test_compare_resolves_and_hydrates_entities_separately(catalog: ProductCatalog) -> None:
    class Embeddings:
        def embed(
            self,
            texts: Sequence[str],
            *,
            text_version: str = DEFAULT_TEXT_VERSION,
            refresh: bool = False,
        ) -> list[list[float]]:
            del text_version, refresh
            return [[1.0, 0.0] for _ in texts]

    class ExactStore:
        collection_name = "semantic-plan-test"

        def count_candidates(self, args: ProductSearchArguments) -> int:
            del args
            return len(catalog.products)

        def exact_candidates(
            self,
            args: ProductSearchArguments,
            *,
            include_structured_filters: bool,
        ) -> list[VectorSearchHit]:
            del include_structured_filters
            product = catalog.product_by_id(args.product_id or "")
            if product is None:
                return []
            return [
                VectorSearchHit(
                    product_id=product["product_id"],
                    score=1.0,
                    payload={
                        "name": product["name"],
                        "sale_price": product["price"]["sale"],
                        "rating": product["rating"],
                    },
                )
            ]

        def search_candidates(
            self,
            vector: list[float],
            args: ProductSearchArguments,
            *,
            candidate_limit: int = 20,
        ) -> list[VectorSearchHit]:
            del vector, args, candidate_limit
            return []

    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro ilə Galaxy S25 Ultra-nı müqayisə et",
            "operation": "compare",
            "entities": [
                {
                    "entity_id": "left",
                    "raw_text": "iPhone 17 Pro",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "right",
                    "raw_text": "Galaxy S25 Ultra",
                    "evidence_text": "Galaxy S25 Ultra",
                    "identifier_type": "model",
                },
            ],
            "selection_expression": {
                "kind": "any_of",
                "expressions": [
                    {"kind": "entity_ref", "entity_id": "left"},
                    {"kind": "entity_ref", "entity_id": "right"},
                ],
            },
        }
    )

    execution = QdrantProductSearch(catalog, Embeddings(), ExactStore()).search_with_trace(plan)
    presentation = build_product_cards(execution.result.model_dump(mode="json"))

    assert execution.result.operation == "compare"
    assert execution.result.display_product_ids == [
        "prd_smartphones_001",
        "prd_smartphones_014",
    ]
    assert execution.result.recommended_product_id is None
    assert presentation is not None
    assert presentation["result_kind"] == "comparison"
    assert presentation["recommended_product_id"] is None


def test_context_reference_accepts_only_server_provided_catalog_candidate(
    catalog: ProductCatalog,
) -> None:
    base = {
        "query": "bu məhsulun qiyməti",
        "operation": "lookup",
        "entities": [
            {
                "entity_id": "focused",
                "raw_text": "bu məhsulun",
                "evidence_text": "bu məhsulun",
                "context_product_id": "prd_smartphones_001",
            }
        ],
        "fact_questions": [{"field": "price", "evidence_text": "qiyməti"}],
    }
    valid_plan = ProductQueryPlan.model_validate(
        {**base, "context_product_ids": ["prd_smartphones_001"]}
    )
    invalid_plan = ProductQueryPlan.model_validate(
        {**base, "context_product_ids": ["prd_smartphones_002"]}
    )

    valid = compile_semantic_plan(valid_plan, catalog)
    invalid = compile_semantic_plan(invalid_plan, catalog)

    assert valid.clarification is None
    assert valid.arguments[0].product_id == "prd_smartphones_001"
    assert invalid.arguments == ()
    assert invalid.clarification and invalid.clarification["reason"] == "invalid_context_reference"


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((0.91, 0.51), "resolved"),
        ((0.91, 0.88), "ambiguous"),
    ],
)
def test_semantic_entity_candidates_require_unique_score_margin(
    catalog: ProductCatalog,
    scores: tuple[float, float],
    expected: str,
) -> None:
    products = catalog.products[:2]

    class Embeddings:
        def embed(
            self,
            texts: Sequence[str],
            *,
            text_version: str = DEFAULT_TEXT_VERSION,
            refresh: bool = False,
        ) -> list[list[float]]:
            del text_version, refresh
            return [[1.0, 0.0] for _ in texts]

    class CandidateStore:
        collection_name = "semantic-entity-test"

        def count_candidates(self, args: ProductSearchArguments) -> int:
            del args
            return len(catalog.products)

        def exact_candidates(
            self,
            args: ProductSearchArguments,
            *,
            include_structured_filters: bool,
        ) -> list[VectorSearchHit]:
            del include_structured_filters
            product = catalog.product_by_id(args.product_id or "")
            if product is None:
                return []
            return [self._hit(product, 1.0)]

        def search_candidates(
            self,
            vector: list[float],
            args: ProductSearchArguments,
            *,
            candidate_limit: int = 20,
        ) -> list[VectorSearchHit]:
            del vector, args, candidate_limit
            return [
                self._hit(product, score)
                for product, score in zip(products, scores, strict=True)
            ]

        @staticmethod
        def _hit(product: dict[str, Any], score: float) -> VectorSearchHit:
            return VectorSearchHit(
                product_id=product["product_id"],
                score=score,
                payload={
                    "name": product["name"],
                    "sale_price": product["price"]["sale"],
                    "rating": product["rating"],
                },
            )

    plan = ProductQueryPlan.model_validate(
        {
            "query": "mənim kod adlı məhsulum",
            "operation": "lookup",
            "entities": [
                {
                    "entity_id": "unknown",
                    "raw_text": "kod adlı məhsulum",
                    "evidence_text": "kod adlı məhsulum",
                }
            ],
        }
    )
    execution = QdrantProductSearch(
        catalog,
        Embeddings(),
        CandidateStore(),
    ).search_with_trace(plan)

    if expected == "resolved":
        assert execution.result.match_status == "exact_match"
        assert execution.result.resolved_entities[0]["product_id"] == products[0]["product_id"]
    else:
        assert execution.result.clarification
        assert execution.result.items == []
        assert execution.debug_trace["semantic_entity_candidates"][0]["decision"] == "ambiguous"


def test_tool_schema_exposes_semantic_plan_and_catalog_fields(catalog: ProductCatalog) -> None:
    class Backend:
        def __init__(self) -> None:
            self.catalog = catalog

    registry = ToolRegistry(
        ProductSearchTool(Backend()),  # type: ignore[arg-type]
        timeout_seconds=1,
    )
    schema = registry.specs()[0]["function"]["parameters"]

    assert "operation" in schema["properties"]
    assert "entities" in schema["properties"]
    assert "brand" not in schema["properties"]
    assert "context_product_ids" not in schema["properties"]
    assert "context_memory" not in schema["properties"]
    assert "memory_action" in schema["properties"]
    assert "memory_removals" in schema["properties"]
    assert "ranking_objectives" in schema["properties"]
    assert "btu" in schema["$defs"]["SemanticPredicate"]["properties"]["field"]["enum"]
    ranking_fields = schema["$defs"]["RankingObjective"]["properties"]["field"]["enum"]
    assert "display_size_in" in ranking_fields
    assert "brand" not in ranking_fields
    value_description = schema["$defs"]["SemanticPredicate"]["properties"]["value"]["description"]
    assert "color_code" in value_description
    assert "blue" in value_description


def test_qualitative_goal_compiles_to_directional_ranking_without_threshold(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Böyük ekranlı smartfon göstər",
            "operation": "discover",
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "category_id",
                    "operator": "eq",
                    "value": "smartphones",
                    "strength": "hard",
                    "evidence_text": "smartfon",
                },
            },
            "ranking_objectives": [
                {
                    "field": "display_size_in",
                    "direction": "maximize",
                    "priority": "primary",
                    "origin": "explicit",
                    "evidence_text": "Böyük ekranlı",
                }
            ],
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.arguments[0].max_price is None
    assert compilation.arguments[0].screen_size_in is None
    assert compilation.arguments[0].semantic_ranking_objectives == plan.ranking_objectives
    assert compilation.field_capability_resolution[0]["supported"] is True


def test_ungrounded_numeric_threshold_requires_clarification(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Ekranı böyük smartfon göstər",
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
                            "evidence_text": "smartfon",
                        },
                    },
                    {
                        "kind": "predicate",
                        "predicate": {
                            "field": "display_size_in",
                            "operator": "gte",
                            "value": 6.5,
                            "unit": "in",
                            "strength": "hard",
                            "value_provenance": "current_message",
                            "evidence_text": "Ekranı böyük",
                        },
                    },
                ],
            },
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.arguments == ()
    assert compilation.clarification is not None
    assert "numeric_value_not_grounded" in compilation.clarification["reason"]
    assert compilation.numeric_provenance[-1]["valid"] is False


def test_unreferenced_hallucinated_entity_is_dropped_when_grounded_plan_remains(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Nənəmin gözü zəifdir, böyük ekranlı smartfon göstər",
            "operation": "discover",
            "entities": [
                {
                    "entity_id": "hallucinated",
                    "raw_text": "iPhone",
                    "evidence_text": "iPhone",
                }
            ],
            "filter_expression": {
                "kind": "predicate",
                "predicate": {
                    "field": "category_id",
                    "operator": "eq",
                    "value": "smartphones",
                    "strength": "hard",
                    "evidence_text": "smartfon",
                },
            },
            "ranking_objectives": [
                {
                    "field": "display_size_in",
                    "direction": "maximize",
                    "priority": "primary",
                    "origin": "explicit",
                    "evidence_text": "böyük ekranlı",
                }
            ],
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.plan.entities == []
    assert compilation.resolutions == ()
    assert compilation.plan_corrections[0]["action"] == "dropped_ungrounded_entity"


def test_multiple_selected_entities_without_relationship_require_clarification(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "iPhone 17 Pro və Galaxy S25 Ultra",
            "operation": "discover",
            "entities": [
                {
                    "entity_id": "left",
                    "raw_text": "iPhone 17 Pro",
                    "evidence_text": "iPhone 17 Pro",
                    "identifier_type": "model",
                },
                {
                    "entity_id": "right",
                    "raw_text": "Galaxy S25 Ultra",
                    "evidence_text": "Galaxy S25 Ultra",
                    "identifier_type": "model",
                },
            ],
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.arguments == ()
    assert compilation.clarification is not None
    assert "unbound_selected_entity" in compilation.clarification["reason"]


def test_explicit_unsupported_ranking_blocks_but_inferred_is_safely_dropped(
    catalog: ProductCatalog,
) -> None:
    base = {
        "query": "Rahat smartfon göstər",
        "operation": "discover",
        "filter_expression": {
            "kind": "predicate",
            "predicate": {
                "field": "category_id",
                "operator": "eq",
                "value": "smartphones",
                "strength": "hard",
                "evidence_text": "smartfon",
            },
        },
    }
    explicit = ProductQueryPlan.model_validate(
        {
            **base,
            "ranking_objectives": [
                {
                    "field": "brand",
                    "direction": "maximize",
                    "origin": "explicit",
                    "evidence_text": "Rahat",
                }
            ],
        }
    )
    inferred = ProductQueryPlan.model_validate(
        {
            **base,
            "ranking_objectives": [
                {
                    "field": "brand",
                    "direction": "maximize",
                    "priority": "inferred",
                    "origin": "inferred",
                    "evidence_text": "Rahat",
                }
            ],
        }
    )

    blocked = compile_semantic_plan(explicit, catalog)
    recovered = compile_semantic_plan(inferred, catalog)

    assert blocked.clarification is not None
    assert "unsupported_ranking_objective" in blocked.clarification["reason"]
    assert recovered.clarification is None
    assert recovered.plan.ranking_objectives == []
    assert recovered.plan_corrections[0]["action"] == "dropped_inferred_ranking_objective"


def test_conflicting_explicit_sort_and_direction_require_clarification(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Ucuzdan bahaya düz, amma ən bahalıya üstünlük ver",
            "operation": "discover",
            "sort": "price_asc",
            "ranking_objectives": [
                {
                    "field": "sale_price",
                    "direction": "maximize",
                    "priority": "primary",
                    "origin": "explicit",
                    "evidence_text": "ən bahalıya üstünlük ver",
                }
            ],
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.arguments == ()
    assert compilation.clarification is not None
    assert "conflicting_ranking_direction" in compilation.clarification["reason"]


def test_any_of_keeps_grounded_branch_and_reports_unavailable_value(
    catalog: ProductCatalog,
) -> None:
    plan = ProductQueryPlan.model_validate(
        {
            "query": "Qara və ya qirmizi 128 GB iPhone göstər",
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
        }
    )

    compilation = compile_semantic_plan(plan, catalog)

    assert compilation.clarification is None
    assert compilation.arguments[0].color_code == "black"
    assert compilation.arguments[0].storage_gb == 128
    assert compilation.unavailable_requested_values == (
        {
            "field": "color_code",
            "value": "red",
            "reason": "unmapped_required_value",
            "evidence_text": "qirmizi",
        },
    )
    assert all(item["valid"] for item in compilation.evidence_validation)


@pytest.mark.asyncio
async def test_invalid_semantic_tool_plan_returns_clarification_without_backend_retry(
    catalog: ProductCatalog,
) -> None:
    class Backend:
        def __init__(self) -> None:
            self.catalog = catalog
            self.calls = 0

        def search(self, arguments: object) -> object:
            del arguments
            self.calls += 1
            raise AssertionError("invalid plan must not reach product retrieval")

    backend = Backend()
    registry = ToolRegistry(
        ProductSearchTool(backend),  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    execution = await registry.execute_with_trace(
        "product_search",
        {
            "query": "məhsul",
            "operation": "lookup",
            "entities": [],
            "memory_action": "merge",
            "removed_memory_ids": ["mem_entity_123"],
        },
    )

    json.dumps(execution.debug_trace)

    assert backend.calls == 0
    assert execution.result["clarification"]["reason"] == "semantic_plan_invalid"
    assert execution.debug_trace and execution.debug_trace["semantic_plan_invalid"] is True
