from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.evals.product_cases import PRODUCTS_PATH, CaseResult, load_eval_cases
from app.evals.product_semantic import apply_baseline, build_semantic_gates, build_semantic_report
from app.retrieval.semantic import SemanticProductSearch
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments, ProductSearchResult
from app.vectorstores.qdrant import CollectionStatus, VectorSearchHit


class FakeQueryEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0]]


class FakeStore:
    collection_name = "sales_bot_products_semantic_v2"

    def __init__(self, *, total: int, product_ids: list[str], catalog: ProductCatalog) -> None:
        self.total = total
        self.product_ids = product_ids
        self.products = {product["product_id"]: product for product in catalog.products}

    def count_candidates(self, _: ProductSearchArguments) -> int:
        return self.total

    def exact_candidates(self, *_: Any, **__: Any) -> list[VectorSearchHit]:
        return []

    def search_candidates(self, *_: Any, **__: Any) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                product_id=product_id,
                score=1.0,
                payload={
                    "product_id": product_id,
                    "name": self.products[product_id]["name"],
                    "sale_price": self.products[product_id]["price"]["sale"],
                    "rating": self.products[product_id]["rating"],
                },
            )
            for product_id in self.product_ids
        ]


class EmptyBackend:
    def search(self, args: ProductSearchArguments) -> ProductSearchResult:
        return ProductSearchResult(
            match_status="not_found",
            strict_total=0,
            total=0,
            applied_filters={},
            items=[],
        )


def _catalog() -> ProductCatalog:
    catalog = ProductCatalog(PRODUCTS_PATH)
    catalog.load()
    return catalog


def _ready_status() -> CollectionStatus:
    return CollectionStatus(
        exists=True,
        collection_name="sales_bot_products_semantic_v2",
        expected_count=300,
        indexed_count=300,
        vector_size=3072,
        distance="Cosine",
        missing_product_ids=(),
        extra_product_ids=(),
        dataset_versions=("1.0.0",),
        catalog_checksums=("checksum",),
        embedding_text_versions=("name_description_v2",),
        embedding_deployments=("text-embedding-3-large",),
        embedding_dimensions=(3072,),
        metadata_matches=True,
        payload_fields_match=True,
    )


def test_semantic_search_skips_embedding_when_filters_have_no_candidates() -> None:
    catalog = _catalog()
    embeddings = FakeQueryEmbeddings()
    backend = SemanticProductSearch(
        catalog,
        embeddings,
        FakeStore(total=0, product_ids=[], catalog=catalog),
    )

    result = backend.search(
        ProductSearchArguments(query="missing", category_id="laptops", brand="Missing")
    )

    assert result.total == 0
    assert result.items == []
    assert embeddings.calls == []


def test_semantic_search_maps_qdrant_hits_back_to_json_catalog() -> None:
    catalog = _catalog()
    product_ids = [catalog.products[0]["product_id"], catalog.products[1]["product_id"]]
    embeddings = FakeQueryEmbeddings()
    backend = SemanticProductSearch(
        catalog,
        embeddings,
        FakeStore(total=50, product_ids=product_ids, catalog=catalog),
    )

    result = backend.search(ProductSearchArguments(query="gündəlik telefon"))

    assert result.total == 50
    assert [item.product_id for item in result.items] == product_ids
    assert embeddings.calls == [["gündəlik telefon"]]


def _perfect_results() -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in load_eval_cases():
        returned = case.expected_product_ids[:5]
        first_rank = 1 if returned else None
        results.append(
            CaseResult(
                query_id=case.query_id,
                suite=case.suite,
                type=case.type,
                query=case.query,
                expected_product_ids=case.expected_product_ids,
                returned_product_ids=returned,
                hit_product_ids=returned,
                false_positive_ids=[],
                missed_product_ids=case.expected_product_ids[5:],
                expected_total=len(case.expected_product_ids),
                returned_total=len(case.expected_product_ids),
                total_count_matches=True,
                expect_empty=case.expect_empty,
                empty_result_correct=True if case.expect_empty else None,
                precision_at_5=1.0,
                top_5_coverage=1.0,
                hit_at_5=bool(returned),
                first_hit_rank=first_rank,
                reciprocal_rank=1.0 if first_rank else 0.0,
                outcome="passed",
            )
        )
    return results


def test_semantic_gates_require_exact_identifier_top_one() -> None:
    results = _perfect_results()
    assert build_semantic_gates(results, _ready_status())["passed"] is True

    exact_index = next(index for index, result in enumerate(results) if result.type == "exact_identifier")
    results[exact_index] = results[exact_index].model_copy(update={"first_hit_rank": 2})

    gates = build_semantic_gates(results, _ready_status())
    assert gates["passed"] is False
    assert gates["checks"]["challenge_exact_identifier_top_1"]["passed"] is False


def test_semantic_report_is_deterministic_and_has_no_lexical_comparison() -> None:
    catalog = _catalog()
    cases = load_eval_cases()
    report = build_semantic_report(
        catalog,
        cases,
        EmptyBackend(),
        _ready_status(),
        embedding_deployment="text-embedding-3-large",
        embedding_dimensions=3072,
    )
    second = build_semantic_report(
        catalog,
        cases,
        EmptyBackend(),
        _ready_status(),
        embedding_deployment="text-embedding-3-large",
        embedding_dimensions=3072,
    )

    assert report == second
    assert report["search_version"] == "semantic_qdrant_v2"
    assert report["embedding"]["input_fields"] == ["name", "description"]
    assert report["retrieval_policy"] == {
        "alternative_min_score": 0.39,
        "max_alternatives": 3,
    }
    assert "comparison_with_lexical" not in report


def test_apply_baseline_requires_explicit_update() -> None:
    report = {"stable": True}
    path = Path("semantic-baseline.json")
    with patch.object(Path, "is_file", return_value=False):
        assert apply_baseline(report, update_baseline=False, baseline_path=path) == 1

    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=1) as write_text,
    ):
        assert apply_baseline(report, update_baseline=True, baseline_path=path) == 0

    mkdir.assert_called_once_with(parents=True, exist_ok=True)
    write_text.assert_called_once()
