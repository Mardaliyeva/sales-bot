from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.evals.product_retrieval import PRODUCTS_PATH, build_report, load_eval_cases
from app.evals.product_semantic import apply_baseline, build_semantic_gates, build_semantic_report
from app.retrieval.semantic import SemanticProductSearch
from app.tools.catalog import ProductCatalog
from app.tools.schemas import ProductSearchArguments
from app.vectorstores.qdrant import CollectionStatus, VectorSearchHit


class FakeQueryEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0]]


class FakeStore:
    def __init__(self, *, total: int, product_ids: list[str]) -> None:
        self.total = total
        self.product_ids = product_ids

    def count_candidates(self, _: ProductSearchArguments) -> int:
        return self.total

    def search(self, _: list[float], __: ProductSearchArguments) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(product_id=product_id, score=1.0, payload={"product_id": product_id})
            for product_id in self.product_ids
        ]


def _catalog() -> ProductCatalog:
    catalog = ProductCatalog(PRODUCTS_PATH)
    catalog.load()
    return catalog


def _ready_status() -> CollectionStatus:
    return CollectionStatus(
        exists=True,
        collection_name="sales_bot_products_v1",
        expected_count=300,
        indexed_count=300,
        vector_size=3072,
        distance="Cosine",
        missing_product_ids=(),
        extra_product_ids=(),
        dataset_versions=("1.0.0",),
        catalog_checksums=("checksum",),
        embedding_text_versions=("v1",),
        embedding_deployments=("text-embedding-3-large",),
        embedding_dimensions=(3072,),
        metadata_matches=True,
    )


def test_semantic_search_skips_embedding_when_filters_have_no_candidates() -> None:
    embeddings = FakeQueryEmbeddings()
    backend = SemanticProductSearch(_catalog(), embeddings, FakeStore(total=0, product_ids=[]))  # type: ignore[arg-type]

    result = backend.search(
        ProductSearchArguments(query="missing", category_id="laptops", brand="Missing")
    )

    assert result.total == 0
    assert result.items == []
    assert embeddings.calls == []


def test_semantic_search_maps_qdrant_hits_back_to_catalog_products() -> None:
    catalog = _catalog()
    product_ids = [catalog.products[0]["product_id"], catalog.products[1]["product_id"]]
    embeddings = FakeQueryEmbeddings()
    backend = SemanticProductSearch(
        catalog,
        embeddings,
        FakeStore(total=50, product_ids=product_ids),  # type: ignore[arg-type]
    )

    result = backend.search(ProductSearchArguments(query="gündəlik telefon"))

    assert result.total == 50
    assert [item.product_id for item in result.items] == product_ids
    assert embeddings.calls == [["gündəlik telefon"]]


def test_semantic_gates_do_not_require_mrr_one() -> None:
    catalog = _catalog()
    cases = load_eval_cases()
    lexical_report = build_report(catalog, cases)
    results = []
    for case in lexical_report["cases"]:
        if case["type"] == "semantic":
            case = {**case, "first_hit_rank": 2, "reciprocal_rank": 0.5, "hit_at_5": True}
        results.append(case)

    gates = build_semantic_gates(
        [SimpleNamespace(**case) for case in results],  # type: ignore[arg-type]
        _ready_status(),
    )

    assert gates["passed"] is True
    assert "semantic_mrr" not in gates["checks"]


def test_semantic_report_is_deterministic_and_contains_lexical_comparison() -> None:
    catalog = _catalog()
    cases = load_eval_cases()
    report = build_semantic_report(
        catalog,
        cases,
        catalog,
        _ready_status(),
        embedding_deployment="text-embedding-3-large",
        embedding_dimensions=3072,
    )
    second = build_semantic_report(
        catalog,
        cases,
        catalog,
        _ready_status(),
        embedding_deployment="text-embedding-3-large",
        embedding_dimensions=3072,
    )

    assert report == second
    assert report["search_version"] == "semantic_qdrant_v1"
    assert report["embedding"]["dimensions"] == 3072
    assert "exact_identifier" in report["comparison_with_lexical"]
    assert "timestamp" not in report
    assert "endpoint" not in str(report).lower()


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
