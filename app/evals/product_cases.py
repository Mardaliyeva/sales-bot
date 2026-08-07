from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.tools.catalog import ProductCatalog
from app.tools.schemas import MatchStatus, ProductSearchArguments, ProductSearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CASES_PATH = PROJECT_ROOT / "data" / "catalog" / "golden_queries.json"
CHALLENGE_CASES_PATH = PROJECT_ROOT / "data" / "evals" / "product_retrieval_challenge.json"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "catalog" / "products.jsonl"
REPORT_SCHEMA_VERSION = 2
TOP_K = 5


class EvalDataError(RuntimeError):
    pass


class SearchBackend(Protocol):
    def search(self, args: ProductSearchArguments) -> ProductSearchResult: ...


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(pattern=r"^[a-z0-9_]+$")
    suite: Literal["canonical", "challenge"]
    type: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=500)
    filters: dict[str, Any] = Field(default_factory=dict)
    expected_product_ids: list[str]
    expect_empty: bool = False
    expected_match_status: MatchStatus | None = None
    max_returned_items: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def validate_expectations(self) -> RetrievalEvalCase:
        if len(self.expected_product_ids) != len(set(self.expected_product_ids)):
            raise ValueError("expected_product_ids təkrarlanmamalıdır")
        if self.expect_empty != (not self.expected_product_ids):
            raise ValueError("expect_empty və expected_product_ids bir-birinə uyğun deyil")
        if self.expected_match_status == "alternatives" and self.expect_empty:
            raise ValueError("Alternativ eval case-i üçün allowed product ID-ləri verilməlidir")
        if self.expected_match_status == "not_found" and not self.expect_empty:
            raise ValueError("not_found eval case-i boş nəticə gözləməlidir")
        if self.expected_match_status == "alternatives" and self.max_returned_items is None:
            self.max_returned_items = 3
        if "query" in self.filters or "limit" in self.filters:
            raise ValueError("query və limit filters daxilində verilə bilməz")
        ProductSearchArguments(query=self.query, limit=TOP_K, **self.filters)
        return self

    def search_arguments(self) -> ProductSearchArguments:
        return ProductSearchArguments(query=self.query, limit=TOP_K, **self.filters)


class CaseResult(BaseModel):
    query_id: str
    suite: Literal["canonical", "challenge"]
    type: str
    query: str
    expected_product_ids: list[str]
    returned_product_ids: list[str]
    hit_product_ids: list[str]
    false_positive_ids: list[str]
    missed_product_ids: list[str]
    expected_total: int
    returned_total: int
    total_count_matches: bool
    expect_empty: bool
    empty_result_correct: bool | None
    precision_at_5: float
    top_5_coverage: float
    hit_at_5: bool
    first_hit_rank: int | None
    reciprocal_rank: float
    outcome: Literal["passed", "partial", "failed"]
    expected_match_status: MatchStatus | None = None
    actual_match_status: MatchStatus = "matching_products"
    match_status_matches: bool = True
    max_returned_items: int | None = None
    returned_count_within_limit: bool = True


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise EvalDataError(f"Eval faylı tapılmadı: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"Eval JSON etibarsızdır: {path}: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise EvalDataError(f"Eval faylı JSON obyektlər siyahısı olmalıdır: {path}")
    return payload


def load_eval_cases(
    canonical_path: Path = CANONICAL_CASES_PATH,
    challenge_path: Path = CHALLENGE_CASES_PATH,
) -> list[RetrievalEvalCase]:
    raw_cases = [
        {
            **item,
            "suite": "canonical",
            "expect_empty": not item.get("expected_product_ids"),
        }
        for item in _read_json_list(canonical_path)
    ]
    raw_cases.extend({**item, "suite": "challenge"} for item in _read_json_list(challenge_path))
    try:
        cases = [RetrievalEvalCase.model_validate(item) for item in raw_cases]
    except ValidationError as exc:
        raise EvalDataError(f"Eval case validasiyası uğursuz oldu: {exc}") from exc
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise EvalDataError("Eval query_id dəyərləri unikal olmalıdır")
    return cases


def validate_expected_products(cases: Sequence[RetrievalEvalCase], catalog: ProductCatalog) -> None:
    known_ids = {product["product_id"] for product in catalog.products}
    for case in cases:
        unknown_ids = sorted(set(case.expected_product_ids) - known_ids)
        if unknown_ids:
            raise EvalDataError(f"{case.query_id} naməlum məhsul ID-ləri ehtiva edir: {unknown_ids}")


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return round(numerator / denominator, 6) if denominator else empty_value


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def evaluate_case(case: RetrievalEvalCase, backend: SearchBackend) -> CaseResult:
    search_result = backend.search(case.search_arguments())
    returned_ids = [item.product_id for item in search_result.items]
    expected = set(case.expected_product_ids)
    returned = set(returned_ids)
    hit_ids = [product_id for product_id in returned_ids if product_id in expected]
    false_positive_ids = [product_id for product_id in returned_ids if product_id not in expected]
    missed_ids = [product_id for product_id in case.expected_product_ids if product_id not in returned]
    first_hit_rank = next(
        (rank for rank, product_id in enumerate(returned_ids, start=1) if product_id in expected),
        None,
    )
    empty_correct = search_result.total == 0 and not returned_ids if case.expect_empty else None
    precision = _safe_ratio(
        len(hit_ids),
        len(returned_ids),
        empty_value=1.0 if case.expect_empty and not returned_ids else 0.0,
    )
    coverage = _safe_ratio(
        len(hit_ids),
        min(TOP_K, len(expected)),
        empty_value=1.0 if case.expect_empty and not returned_ids else 0.0,
    )
    match_status_matches = (
        case.expected_match_status is None
        or search_result.match_status == case.expected_match_status
    )
    returned_count_within_limit = (
        case.max_returned_items is None or len(returned_ids) <= case.max_returned_items
    )
    if case.expected_match_status == "alternatives":
        if match_status_matches and returned_ids and not false_positive_ids and returned_count_within_limit:
            outcome: Literal["passed", "partial", "failed"] = "passed"
        elif hit_ids:
            outcome = "partial"
        else:
            outcome = "failed"
    elif case.expect_empty:
        outcome: Literal["passed", "partial", "failed"] = "passed" if empty_correct else "failed"
    elif coverage == 1.0 and not false_positive_ids:
        outcome = "passed"
    elif hit_ids:
        outcome = "partial"
    else:
        outcome = "failed"
    return CaseResult(
        query_id=case.query_id,
        suite=case.suite,
        type=case.type,
        query=case.query,
        expected_product_ids=case.expected_product_ids,
        returned_product_ids=returned_ids,
        hit_product_ids=hit_ids,
        false_positive_ids=false_positive_ids,
        missed_product_ids=missed_ids,
        expected_total=len(case.expected_product_ids),
        returned_total=search_result.total,
        total_count_matches=search_result.total == len(case.expected_product_ids),
        expect_empty=case.expect_empty,
        empty_result_correct=empty_correct,
        precision_at_5=precision,
        top_5_coverage=coverage,
        hit_at_5=bool(hit_ids),
        first_hit_rank=first_hit_rank,
        reciprocal_rank=round(1 / first_hit_rank, 6) if first_hit_rank is not None else 0.0,
        outcome=outcome,
        expected_match_status=case.expected_match_status,
        actual_match_status=search_result.match_status,
        match_status_matches=match_status_matches,
        max_returned_items=case.max_returned_items,
        returned_count_within_limit=returned_count_within_limit,
    )


def summarize_suite(results: Sequence[CaseResult]) -> dict[str, Any]:
    relevant = [result for result in results if not result.expect_empty]
    empty = [result for result in results if result.expect_empty]
    return {
        "case_count": len(results),
        "passed": sum(result.outcome == "passed" for result in results),
        "partial": sum(result.outcome == "partial" for result in results),
        "failed": sum(result.outcome == "failed" for result in results),
        "exact_total_count_accuracy": _safe_ratio(
            sum(result.total_count_matches for result in results), len(results)
        ),
        "mean_precision_at_5": _mean([result.precision_at_5 for result in relevant]),
        "mean_top_5_coverage": _mean([result.top_5_coverage for result in relevant]),
        "hit_at_5": _safe_ratio(sum(result.hit_at_5 for result in relevant), len(relevant)),
        "mean_reciprocal_rank": _mean([result.reciprocal_rank for result in relevant]),
        "empty_result_accuracy": _safe_ratio(
            sum(result.empty_result_correct is True for result in empty),
            len(empty),
            empty_value=1.0,
        ),
        "empty_case_count": len(empty),
        "match_status_accuracy": _safe_ratio(
            sum(result.match_status_matches for result in results),
            len(results),
        ),
    }


def build_report(
    catalog: ProductCatalog,
    cases: Sequence[RetrievalEvalCase],
    *,
    backend: SearchBackend,
    search_version: str,
) -> dict[str, Any]:
    validate_expected_products(cases, catalog)
    results = [evaluate_case(case, backend) for case in cases]
    canonical = [result for result in results if result.suite == "canonical"]
    challenge = [result for result in results if result.suite == "challenge"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "search_version": search_version,
        "dataset": {
            "dataset_version": catalog.manifest["dataset_version"],
            "product_count": catalog.manifest["product_count"],
            "products_sha256": catalog.manifest["checksums"]["products_sha256"],
        },
        "total_cases": len(results),
        "suites": {
            "canonical": summarize_suite(canonical),
            "challenge": summarize_suite(challenge),
        },
        "cases": [result.model_dump(mode="json") for result in results],
    }


def serialize_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
