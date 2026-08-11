from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agent.prompt import SYSTEM_PROMPT
from app.config import PROJECT_ROOT, Settings, get_settings
from app.llm.azure_client import AzureChatClient
from app.retrieval.semantic_plan import (
    compile_semantic_plan,
    expression_matches,
    preference_score,
)
from app.tools.catalog import ProductCatalog, normalize_text
from app.tools.product_search import ProductSearchTool
from app.tools.registry import ToolRegistry
from app.tools.schemas import ProductQueryPlan, SemanticExpression

CASES_PATH = PROJECT_ROOT / "data" / "evals" / "semantic_query_plans.json"
MIN_ACCURACY = 0.95


class _SchemaOnlyBackend:
    def __init__(self, catalog: ProductCatalog) -> None:
        self.catalog = catalog

    def search(self, arguments: object) -> object:
        del arguments
        raise RuntimeError("schema-only backend cannot execute retrieval")


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("semantic plan eval data must be a JSON object list")
    return payload


def semantic_signature(
    plan: ProductQueryPlan,
    catalog: ProductCatalog | None = None,
    arguments: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    if plan.needs_clarification:
        return {
            "operation": plan.operation,
            "needs_clarification": True,
        }
    entity_ids = {
        entity.entity_id: f"entity_{index}"
        for index, entity in enumerate(plan.entities, start=1)
    }

    def expression_signature(expression: SemanticExpression | None) -> Any:
        if expression is None:
            return None
        if expression.kind == "predicate" and expression.predicate is not None:
            predicate = expression.predicate
            return {
                "kind": "predicate",
                "field": predicate.field,
                "operator": predicate.operator,
                "value": predicate.value,
                "strength": predicate.strength,
                "unit": predicate.unit or ("AZN" if predicate.field == "price" else None),
            }
        if expression.kind == "entity_ref":
            return {"kind": "entity_ref", "entity_id": entity_ids.get(expression.entity_id)}
        if expression.expressions is not None:
            return {
                "kind": expression.kind,
                "expressions": [expression_signature(child) for child in expression.expressions],
            }
        if expression.expression is not None:
            return {
                "kind": expression.kind,
                "expression": expression_signature(expression.expression),
            }
        return {
            "kind": expression.kind,
            "primary": expression_signature(expression.primary),
            "secondary": expression_signature(expression.secondary),
        }

    selected_entities = [entity for entity in plan.entities if entity.state == "selected"]
    selection_signature = expression_signature(plan.selection_expression)
    if selection_signature is None and len(selected_entities) == 1:
        selection_signature = {
            "kind": "entity_ref",
            "entity_id": entity_ids[selected_entities[0].entity_id],
        }
    filter_signature = expression_signature(plan.filter_expression)
    if (
        plan.selection_expression is not None
        and _signature_has_predicate(selection_signature)
        and not _signature_has_entity_ref(selection_signature)
    ):
        if filter_signature is None:
            filter_signature = selection_signature
        else:
            filter_signature = {
                "kind": "all_of",
                "expressions": [filter_signature, selection_signature],
            }
        selection_signature = None
    if plan.operation == "compare":
        selection_signature = None
    entities_signature = [
        {
            "entity_id": entity_ids[entity.entity_id],
            "raw_text": normalize_text(entity.raw_text),
            "state": entity.state,
            "supersedes_entity_id": entity_ids.get(entity.supersedes_entity_id),
            "identifier_type": (
                entity.identifier_type
                if entity.identifier_type in {"product_id", "sku"}
                else "catalog_text"
            ),
        }
        for entity in plan.entities
    ]
    if (
        plan.operation == "discover"
        and plan.selection_expression is not None
        and plan.selection_expression.kind in {"all_of", "any_of"}
    ):
        entities_signature = []
        selection_signature = None
    if (
        plan.operation == "discover"
        and plan.filter_expression is not None
        and plan.filter_expression.kind != "fallback"
        and plan.selection_expression is None
    ):
        entities_signature = []
        selection_signature = None
    result = {
        "operation": plan.operation,
        "entities": (
            sorted(entities_signature, key=lambda item: (item["raw_text"], item["state"]))
            if plan.operation == "compare"
            else entities_signature
        ),
        "selection_expression": selection_signature,
        "filter_expression": filter_signature,
        "preference_expression": expression_signature(plan.preference_expression),
        "fact_questions": [
            {
                "field": question.field,
                "operator": question.operator,
                "value": question.value,
                "unit": question.unit or ("AZN" if question.field == "price" else None),
            }
            for question in plan.fact_questions
        ],
        "recommendation_requested": (
            plan.recommendation_requested if plan.operation == "compare" else False
        ),
        "needs_clarification": plan.needs_clarification,
    }
    if catalog is not None:
        branch_arguments = arguments or ()
        if branch_arguments:
            result["filter_expression"] = [
                sorted(
                    str(product["product_id"])
                    for product in catalog.products
                    if (argument.product_id is None or product["product_id"] == argument.product_id)
                    and expression_matches(argument.semantic_filter_expression, product)
                )
                for argument in branch_arguments
            ]
        else:
            result["filter_expression"] = sorted(
                str(product["product_id"])
                for product in catalog.products
                if expression_matches(plan.filter_expression, product)
            )
        result["preference_expression"] = [
            [str(product["product_id"]), preference_score(plan.preference_expression, product)]
            for product in catalog.products
        ]
    return result


def _signature_has_predicate(expression: Any) -> bool:
    if isinstance(expression, dict):
        return expression.get("kind") == "predicate" or any(
            _signature_has_predicate(value) for value in expression.values()
        )
    if isinstance(expression, list):
        return any(_signature_has_predicate(value) for value in expression)
    return False


def _signature_has_entity_ref(expression: Any) -> bool:
    if isinstance(expression, dict):
        return expression.get("kind") == "entity_ref" or any(
            _signature_has_entity_ref(value) for value in expression.values()
        )
    if isinstance(expression, list):
        return any(_signature_has_entity_ref(value) for value in expression)
    return False


async def run_eval(
    settings: Settings,
    case_ids: set[str] | None = None,
    *,
    verbose: bool = False,
) -> int:
    catalog = ProductCatalog(settings.product_catalog_path)
    catalog.load()
    registry = ToolRegistry(
        ProductSearchTool(_SchemaOnlyBackend(catalog)),  # type: ignore[arg-type]
        timeout_seconds=settings.tool_timeout_seconds,
    )
    cases = load_cases()
    if case_ids:
        cases = [case for case in cases if case.get("id") in case_ids]
    client = AzureChatClient(settings)
    passed = 0
    try:
        for case in cases:
            case_id = str(case["id"])
            expected = ProductQueryPlan.model_validate(case["plan"])
            response = await client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["utterance"]},
                ],
                tools=registry.specs(),
                tool_choice="required",
                request_id=str(uuid4()),
                model_round=1,
            )
            calls = response.first_choice.message.tool_calls or []
            failure = None
            if len(calls) != 1 or calls[0].function.name != "product_search":
                failure = "missing_product_search_plan"
            else:
                try:
                    raw = json.loads(calls[0].function.arguments)
                    raw["query"] = case["utterance"]
                    actual = ProductQueryPlan.model_validate(raw)
                    actual_compilation = compile_semantic_plan(actual, catalog)
                    expected_compilation = compile_semantic_plan(expected, catalog)
                    actual = actual_compilation.plan
                    expected = expected_compilation.plan
                    actual_signature = semantic_signature(
                        actual,
                        catalog,
                        actual_compilation.arguments,
                    )
                    expected_signature = semantic_signature(
                        expected,
                        catalog,
                        expected_compilation.arguments,
                    )
                    actual_signature["compiled_clarification"] = bool(
                        actual_compilation.clarification
                    )
                    expected_signature["compiled_clarification"] = bool(
                        expected_compilation.clarification
                    )
                    if actual_signature != expected_signature:
                        failure = "semantic_signature_mismatch"
                        if verbose:
                            print(
                                json.dumps(
                                    {
                                        "expected": semantic_signature(expected),
                                        "actual": semantic_signature(actual),
                                        "raw": raw,
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                flush=True,
                            )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    failure = "invalid_semantic_plan"
                    if verbose:
                        print(
                            json.dumps(
                                {
                                    "error": str(exc),
                                    "raw": locals().get("raw"),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            flush=True,
                        )
            passed += int(failure is None)
            print(f"{case_id}: {'PASS' if failure is None else 'FAIL ' + failure}", flush=True)
    finally:
        await client.close()
    accuracy = passed / len(cases) if cases else 0.0
    gate = bool(cases) and accuracy >= MIN_ACCURACY
    print(
        f"Gate: {'KEÇDİ' if gate else 'UĞURSUZ'}; "
        f"accuracy={accuracy:.1%}; passed={passed}/{len(cases)}"
    )
    return 0 if gate else 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Live first-round semantic plan evaluation")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return asyncio.run(
        run_eval(
            get_settings(),
            set(args.case_ids or []),
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
