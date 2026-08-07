from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from app.evals.product_cases import (
    PRODUCTS_PATH,
    RetrievalEvalCase,
    build_report,
    evaluate_case,
    load_eval_cases,
    serialize_report,
    validate_expected_products,
)
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments


class StubBackend:
    def __init__(
        self,
        returned_ids: list[str],
        *,
        total: int,
        match_status: str = "matching_products",
    ) -> None:
        self.returned_ids = returned_ids
        self.total = total
        self.match_status = match_status

    def search(self, _: ProductSearchArguments) -> Any:
        return SimpleNamespace(
            total=self.total,
            match_status=self.match_status,
            items=[SimpleNamespace(product_id=product_id) for product_id in self.returned_ids],
        )


@pytest.fixture(scope="module")
def catalog() -> ProductCatalog:
    instance = ProductCatalog(PRODUCTS_PATH)
    instance.load()
    return instance


def test_eval_suite_contains_canonical_and_alternative_challenge_cases(
    catalog: ProductCatalog,
) -> None:
    cases = load_eval_cases()
    assert len(cases) == 67
    assert len({case.query_id for case in cases}) == 67
    assert Counter(case.suite for case in cases) == {"canonical": 30, "challenge": 37}
    challenge_types = Counter(case.type for case in cases if case.suite == "challenge")
    assert challenge_types == {
        "semantic_paraphrase": 6,
        "typo": 6,
        "mixed_language": 6,
        "exact_identifier": 6,
        "negative": 6,
        "alternatives": 6,
        "alternative_negative": 1,
    }
    validate_expected_products(cases, catalog)


def test_exact_eval_cases_use_explicit_qdrant_identifier_fields() -> None:
    exact_cases = [case for case in load_eval_cases() if case.type == "exact_identifier"]

    assert len(exact_cases) == 6
    assert all(
        any(key in case.filters for key in ("product_id", "sku", "model"))
        for case in exact_cases
    )


def test_canonical_semantic_labels_cover_all_products_matching_templated_use_case() -> None:
    catalog = ProductCatalog(PRODUCTS_PATH)
    catalog.load()
    semantic_cases = [
        case for case in load_eval_cases() if case.suite == "canonical" and case.type == "semantic"
    ]

    assert len(semantic_cases) == 6
    for case in semantic_cases:
        category_id = case.filters["category_id"]
        category_ids = {
            product["product_id"]
            for product in catalog.products
            if product["category"]["id"] == category_id
        }
        assert set(case.expected_product_ids) == category_ids


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

    assert evaluate_case(case, StubBackend([], total=0)).empty_result_correct is True
    assert evaluate_case(case, StubBackend(["unexpected"], total=1)).outcome == "failed"


def test_alternative_case_checks_status_allowed_ids_and_limit() -> None:
    case = RetrievalEvalCase(
        query_id="alternative_case",
        suite="challenge",
        type="alternatives",
        query="Future Phone 99",
        filters={"category_id": "smartphones", "model": "Future Phone 99"},
        expected_product_ids=["allowed_1", "allowed_2", "allowed_3"],
        expected_match_status="alternatives",
        max_returned_items=3,
    )

    passed = evaluate_case(
        case,
        StubBackend(["allowed_1", "allowed_2"], total=2, match_status="alternatives"),
    )
    failed = evaluate_case(
        case,
        StubBackend(["wrong_1"], total=1, match_status="matching_products"),
    )

    assert passed.outcome == "passed"
    assert passed.match_status_matches is True
    assert passed.returned_count_within_limit is True
    assert failed.outcome == "failed"


def test_generic_report_is_deterministic(catalog: ProductCatalog) -> None:
    cases = load_eval_cases()
    backend = StubBackend([], total=0)
    first = serialize_report(
        build_report(catalog, cases, backend=backend, search_version="test_qdrant")
    )
    second = serialize_report(
        build_report(catalog, cases, backend=backend, search_version="test_qdrant")
    )

    assert first == second
