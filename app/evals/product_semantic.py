from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import get_settings
from app.embeddings.azure import DEFAULT_TEXT_VERSION, AzureEmbeddingClient, EmbeddingError
from app.evals.product_cases import (
    CaseResult,
    EvalDataError,
    RetrievalEvalCase,
    SearchBackend,
    build_report,
    configure_utf8_output,
    load_eval_cases,
    serialize_report,
)
from app.retrieval.qdrant import ALTERNATIVE_LIMIT, DEFAULT_ALTERNATIVE_MIN_SCORE
from app.retrieval.semantic import SemanticProductSearch
from app.tools.catalog import CatalogLoadError, ProductCatalog
from app.vectorstores.qdrant import CollectionStatus, QdrantProductStore, VectorStoreError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_BASELINE_PATH = (
    PROJECT_ROOT / "data" / "evals" / "baselines" / "semantic_qdrant_v2.json"
)
SEARCH_VERSION = "semantic_qdrant_v2"


def build_semantic_gates(results: Sequence[CaseResult], status: CollectionStatus) -> dict[str, Any]:
    canonical_metadata = [
        result for result in results if result.suite == "canonical" and result.type == "metadata"
    ]
    canonical_semantic = [
        result for result in results if result.suite == "canonical" and result.type == "semantic"
    ]
    canonical_empty = [result for result in canonical_metadata if result.expect_empty]
    challenge_empty = [
        result
        for result in results
        if result.suite == "challenge" and result.type == "negative" and result.expect_empty
    ]
    challenge_alternatives = [
        result
        for result in results
        if result.suite == "challenge"
        and result.type in {"alternatives", "alternative_negative"}
    ]
    challenge_exact = [
        result
        for result in results
        if result.suite == "challenge" and result.type == "exact_identifier"
    ]
    metadata_exact = sum(
        result.total_count_matches
        and not result.false_positive_ids
        and len(result.hit_product_ids) == min(5, result.expected_total)
        for result in canonical_metadata
    )
    exact_top_one = sum(result.first_hit_rank == 1 for result in challenge_exact)
    checks = {
        "index_ready": {"actual": status.ready, "expected": True, "passed": status.ready},
        "canonical_metadata_exact": {
            "actual": metadata_exact,
            "expected": 24,
            "passed": metadata_exact == len(canonical_metadata) == 24,
        },
        "canonical_empty_results": {
            "actual": sum(result.empty_result_correct is True for result in canonical_empty),
            "expected": 6,
            "passed": len(canonical_empty) == 6
            and all(result.empty_result_correct is True for result in canonical_empty),
        },
        "challenge_empty_results": {
            "actual": sum(result.empty_result_correct is True for result in challenge_empty),
            "expected": 6,
            "passed": len(challenge_empty) == 6
            and all(result.empty_result_correct is True for result in challenge_empty),
        },
        "canonical_semantic_hit_at_5": {
            "actual": sum(result.hit_at_5 for result in canonical_semantic),
            "expected": 6,
            "passed": len(canonical_semantic) == 6
            and all(result.hit_at_5 for result in canonical_semantic),
        },
        "challenge_exact_identifier_top_1": {
            "actual": exact_top_one,
            "expected": 6,
            "passed": len(challenge_exact) == 6 and exact_top_one == 6,
        },
        "challenge_alternative_policy": {
            "actual": sum(result.outcome == "passed" for result in challenge_alternatives),
            "expected": 7,
            "passed": len(challenge_alternatives) == 7
            and all(result.outcome == "passed" for result in challenge_alternatives),
        },
    }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def build_semantic_report(
    catalog: ProductCatalog,
    cases: Sequence[RetrievalEvalCase],
    backend: SearchBackend,
    status: CollectionStatus,
    *,
    embedding_deployment: str,
    embedding_dimensions: int,
    alternative_min_score: float = DEFAULT_ALTERNATIVE_MIN_SCORE,
) -> dict[str, Any]:
    report = build_report(
        catalog,
        cases,
        backend=backend,
        search_version=SEARCH_VERSION,
    )
    results = [CaseResult.model_validate(case) for case in report["cases"]]
    report["schema_version"] = 3
    report["embedding"] = {
        "deployment": embedding_deployment,
        "dimensions": embedding_dimensions,
        "text_version": DEFAULT_TEXT_VERSION,
        "input_fields": ["name", "description"],
    }
    report["qdrant"] = {
        "collection_name": status.collection_name,
        "indexed_count": status.indexed_count,
        "vector_size": status.vector_size,
        "distance": status.distance,
        "payload_fields_match": status.payload_fields_match,
        "ready": status.ready,
    }
    report["retrieval_policy"] = {
        "alternative_min_score": alternative_min_score,
        "max_alternatives": ALTERNATIVE_LIMIT,
    }
    report["canonical_gates"] = build_semantic_gates(results, status)
    return report


def render_semantic_report(report: dict[str, Any]) -> str:
    canonical = report["suites"]["canonical"]
    challenge = report["suites"]["challenge"]
    gate_status = "KEÇDİ" if report["canonical_gates"]["passed"] else "UĞURSUZ"
    return "\n".join(
        [
            "Qdrant-only Product Retrieval Baseline",
            f"Dataset: {report['dataset']['dataset_version']} ({report['dataset']['product_count']} məhsul)",
            f"Search: {report['search_version']}",
            (
                f"Embedding: {report['embedding']['deployment']} "
                f"({report['embedding']['dimensions']} dimensions, name + description)"
            ),
            f"Qdrant: {report['qdrant']['collection_name']} ({report['qdrant']['indexed_count']} point)",
            (
                "Canonical: "
                f"{canonical['case_count']} (keçdi={canonical['passed']}, "
                f"qismən={canonical['partial']}, səhv={canonical['failed']})"
            ),
            (
                "Challenge: "
                f"{challenge['case_count']} (keçdi={challenge['passed']}, "
                f"qismən={challenge['partial']}, səhv={challenge['failed']})"
            ),
            f"Semantic acceptance gate: {gate_status}",
        ]
    )


def apply_baseline(
    report: dict[str, Any],
    *,
    update_baseline: bool,
    baseline_path: Path = SEMANTIC_BASELINE_PATH,
) -> int:
    serialized = serialize_report(report)
    try:
        if update_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(serialized, encoding="utf-8")
            print(f"Semantic baseline yeniləndi: {baseline_path}")
            return 0
        if not baseline_path.is_file():
            print(
                "Semantic baseline tapılmadı. Yaratmaq üçün --update-baseline istifadə edin.",
                file=sys.stderr,
            )
            return 1
        current = baseline_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Semantic baseline fayl xətası: {exc}", file=sys.stderr)
        return 1
    if current != serialized:
        print(
            "Semantic baseline fərqi aşkarlandı. Yalnız yoxladıqdan sonra --update-baseline işlədin.",
            file=sys.stderr,
        )
        return 1
    print("Semantic baseline cari nəticə ilə eynidir.")
    return 0


def run_semantic_evaluation(
    *,
    update_baseline: bool = False,
    baseline_path: Path = SEMANTIC_BASELINE_PATH,
) -> int:
    embeddings: AzureEmbeddingClient | None = None
    store: QdrantProductStore | None = None
    try:
        settings = get_settings()
        catalog = ProductCatalog(settings.product_catalog_path)
        catalog.load()
        cases = load_eval_cases()
        store = QdrantProductStore.from_settings(settings)
        status = store.status(
            [product["product_id"] for product in catalog.products],
            expected_dataset_version=catalog.manifest["dataset_version"],
            expected_catalog_checksum=catalog.manifest["checksums"]["products_sha256"],
            expected_embedding_text_version=DEFAULT_TEXT_VERSION,
            expected_embedding_deployment=settings.azure_embedding_model,
            expected_embedding_dimensions=3072,
        )
        if not status.ready:
            raise VectorStoreError(
                "Qdrant v2 indeksi hazır deyil; əvvəl python -m app.indexing.products index işlədin"
            )
        embeddings = AzureEmbeddingClient.from_settings(settings)
        backend = SemanticProductSearch(
            catalog,
            embeddings,
            store,
            alternative_min_score=settings.alternative_min_score,
        )
        report = build_semantic_report(
            catalog,
            cases,
            backend,
            status,
            embedding_deployment=embeddings.deployment,
            embedding_dimensions=embeddings.dimensions,
            alternative_min_score=settings.alternative_min_score,
        )
    except (
        CatalogLoadError,
        EvalDataError,
        EmbeddingError,
        VectorStoreError,
        ValidationError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"Semantic eval xətası: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Semantic eval xətası: uzaq xidmət əməliyyatı uğursuz oldu", file=sys.stderr)
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()
        if store is not None:
            store.close()

    print(render_semantic_report(report))
    if not report["canonical_gates"]["passed"]:
        print("Semantic acceptance gate keçmədi; baseline yenilənmədi.", file=sys.stderr)
        return 1
    return apply_baseline(report, update_baseline=update_baseline, baseline_path=baseline_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qdrant-only product retrieval evaluator")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Cari nəticəni semantic_qdrant_v2 baseline kimi saxla",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output()
    args = parse_args(argv)
    return run_semantic_evaluation(update_baseline=args.update_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
