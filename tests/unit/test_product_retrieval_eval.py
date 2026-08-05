from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from app.evals.product_retrieval import (
    BASELINE_PATH,
    PRODUCTS_PATH,
    RetrievalEvalCase,
    build_canonical_gates,
    build_report,
    evaluate_case,
    load_eval_cases,
    run_evaluation,
    serialize_report,
    validate_expected_products,
)
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments


class StubBackend:
    def __init__(self, returned_ids: list[str], *, total: int) -> None:
        self.returned_ids = returned_ids
        self.total = total

    def search(self, _: ProductSearchArguments) -> Any:
        return SimpleNamespace(
            total=self.total,
            items=[SimpleNamespace(product_id=product_id) for product_id in self.returned_ids],
        )


@pytest.fixture(scope="module")
def catalog() -> ProductCatalog:
    instance = ProductCatalog(PRODUCTS_PATH)
    instance.load()
    return instance


def test_eval_suite_contains_30_canonical_and_30_balanced_challenge_cases(
    catalog: ProductCatalog,
) -> None:
    cases = load_eval_cases()
    assert len(cases) == 60
    assert len({case.query_id for case in cases}) == 60
    assert Counter(case.suite for case in cases) == {"canonical": 30, "challenge": 30}

    challenge_types = Counter(case.type for case in cases if case.suite == "challenge")
    assert challenge_types == {
        "semantic_paraphrase": 6,
        "typo": 6,
        "mixed_language": 6,
        "exact_identifier": 6,
        "negative": 6,
    }
    validate_expected_products(cases, catalog)


def test_case_metrics_identify_hits_false_positives_and_misses() -> None:
    case = RetrievalEvalCase(
        query_id="metric_case",
        suite="challenge",
        type="metric_test",
        query="test query",
        filters={"category_id": "laptops"},
        expected_product_ids=["expected_1", "expected_2"],
    )
    result = evaluate_case(case, StubBackend(["expected_1", "wrong_1"], total=2))

    assert result.hit_product_ids == ["expected_1"]
    assert result.false_positive_ids == ["wrong_1"]
    assert result.missed_product_ids == ["expected_2"]
    assert result.precision_at_5 == 0.5
    assert result.top_5_coverage == 0.5
    assert result.hit_at_5 is True
    assert result.first_hit_rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.outcome == "partial"


def test_empty_case_passes_only_for_an_empty_search_result() -> None:
    case = RetrievalEvalCase(
        query_id="empty_case",
        suite="challenge",
        type="negative",
        query="missing product",
        filters={"brand": "Missing"},
        expected_product_ids=[],
        expect_empty=True,
    )

    empty_result = evaluate_case(case, StubBackend([], total=0))
    wrong_result = evaluate_case(case, StubBackend(["unexpected"], total=1))

    assert empty_result.empty_result_correct is True
    assert empty_result.precision_at_5 == 1.0
    assert empty_result.top_5_coverage == 1.0
    assert empty_result.outcome == "passed"
    assert wrong_result.empty_result_correct is False
    assert wrong_result.outcome == "failed"


def test_current_report_is_deterministic_and_matches_committed_baseline(
    catalog: ProductCatalog,
) -> None:
    cases = load_eval_cases()
    first = serialize_report(build_report(catalog, cases))
    second = serialize_report(build_report(catalog, cases))

    assert first == second
    assert first == BASELINE_PATH.read_text(encoding="utf-8")


def test_canonical_gates_detect_a_metadata_regression(catalog: ProductCatalog) -> None:
    canonical_cases = [case for case in load_eval_cases() if case.suite == "canonical"]
    results = [evaluate_case(case, catalog) for case in canonical_cases]
    assert build_canonical_gates(results)["passed"] is True

    results[0] = results[0].model_copy(
        update={
            "returned_total": 0,
            "total_count_matches": False,
            "hit_product_ids": [],
        }
    )
    gates = build_canonical_gates(results)

    assert gates["passed"] is False
    assert gates["checks"]["metadata_exact"]["passed"] is False


def test_update_baseline_writes_the_deterministic_report() -> None:
    baseline_path = Path("ignored-baseline.json")
    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=1) as write_text,
    ):
        assert run_evaluation(update_baseline=True, baseline_path=baseline_path) == 0

    mkdir.assert_called_once_with(parents=True, exist_ok=True)
    written_report = write_text.call_args.args[0]
    assert json.loads(written_report)["total_cases"] == 60
    assert write_text.call_args.kwargs == {"encoding": "utf-8"}
