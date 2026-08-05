from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import get_settings
from app.embeddings.azure import DEFAULT_TEXT_VERSION, AzureEmbeddingClient, EmbeddingError
from app.evals.product_retrieval import BASELINE_PATH as LEXICAL_BASELINE_PATH
from app.evals.product_retrieval import (
    CaseResult,
    EvalDataError,
    RetrievalEvalCase,
    SearchBackend,
    build_report,
    configure_utf8_output,
    load_eval_cases,
    serialize_report,
)
from app.retrieval.semantic import SemanticProductSearch
from app.tools.catalog import CatalogLoadError, ProductCatalog
from app.vectorstores.qdrant import CollectionStatus, QdrantProductStore, VectorStoreError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_BASELINE_PATH = PROJECT_ROOT / "data" / "evals" / "baselines" / "semantic_qdrant_v1.json"
SEARCH_VERSION = "semantic_qdrant_v1"


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return round(numerator / denominator, 6) if denominator else empty_value


def summarize_by_type(cases: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["type"])].append(case)
    summaries: dict[str, dict[str, Any]] = {}
    for case_type in sorted(grouped):
        values = grouped[case_type]
        relevant = [case for case in values if not case["expect_empty"]]
        empty = [case for case in values if case["expect_empty"]]
        summaries[case_type] = {
            "case_count": len(values),
            "mean_precision_at_5": _mean([float(case["precision_at_5"]) for case in relevant]),
            "mean_top_5_coverage": _mean([float(case["top_5_coverage"]) for case in relevant]),
            "hit_at_5": _ratio(sum(bool(case["hit_at_5"]) for case in relevant), len(relevant)),
            "mean_reciprocal_rank": _mean([float(case["reciprocal_rank"]) for case in relevant]),
            "empty_result_accuracy": _ratio(
                sum(case["empty_result_correct"] is True for case in empty),
                len(empty),
                empty_value=1.0,
            ),
        }
    return summaries


def build_lexical_comparison(
    semantic_cases: Sequence[dict[str, Any]],
    lexical_baseline_path: Path = LEXICAL_BASELINE_PATH,
) -> dict[str, Any]:
    try:
        lexical_report = json.loads(lexical_baseline_path.read_text(encoding="utf-8"))
        lexical_cases = lexical_report["cases"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvalDataError("Lexical baseline oxuna bilmədi") from exc
    lexical = summarize_by_type(lexical_cases)
    semantic = summarize_by_type(semantic_cases)
    comparison: dict[str, Any] = {}
    for case_type in sorted(set(lexical) | set(semantic)):
        lexical_metrics = lexical.get(case_type, {})
        semantic_metrics = semantic.get(case_type, {})
        deltas = {
            metric: round(float(semantic_metrics[metric]) - float(lexical_metrics[metric]), 6)
            for metric in (
                "mean_precision_at_5",
                "mean_top_5_coverage",
                "hit_at_5",
                "mean_reciprocal_rank",
                "empty_result_accuracy",
            )
            if metric in lexical_metrics and metric in semantic_metrics
        }
        comparison[case_type] = {
            "lexical": lexical_metrics,
            "semantic": semantic_metrics,
            "delta_semantic_minus_lexical": deltas,
        }
    return comparison


def build_semantic_gates(results: Sequence[CaseResult], status: CollectionStatus) -> dict[str, Any]:
    canonical_metadata = [
        result for result in results if result.suite == "canonical" and result.type == "metadata"
    ]
    canonical_semantic = [
        result for result in results if result.suite == "canonical" and result.type == "semantic"
    ]
    canonical_empty = [result for result in canonical_metadata if result.expect_empty]
    challenge_empty = [
        result for result in results if result.suite == "challenge" and result.expect_empty
    ]
    metadata_exact = sum(
        result.total_count_matches
        and not result.false_positive_ids
        and len(result.hit_product_ids) == min(5, result.expected_total)
        for result in canonical_metadata
    )
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
            "passed": all(result.empty_result_correct is True for result in canonical_empty)
            and len(canonical_empty) == 6,
        },
        "challenge_empty_results": {
            "actual": sum(result.empty_result_correct is True for result in challenge_empty),
            "expected": 6,
            "passed": all(result.empty_result_correct is True for result in challenge_empty)
            and len(challenge_empty) == 6,
        },
        "canonical_semantic_hit_at_5": {
            "actual": sum(result.hit_at_5 for result in canonical_semantic),
            "expected": 6,
            "passed": all(result.hit_at_5 for result in canonical_semantic)
            and len(canonical_semantic) == 6,
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
    lexical_baseline_path: Path = LEXICAL_BASELINE_PATH,
) -> dict[str, Any]:
    report = build_report(catalog, cases, backend=backend, search_version=SEARCH_VERSION)
    results = [CaseResult.model_validate(case) for case in report["cases"]]
    report["schema_version"] = 2
    report["embedding"] = {
        "deployment": embedding_deployment,
        "dimensions": embedding_dimensions,
        "text_version": DEFAULT_TEXT_VERSION,
    }
    report["qdrant"] = {
        "collection_name": status.collection_name,
        "indexed_count": status.indexed_count,
        "vector_size": status.vector_size,
        "distance": status.distance,
        "ready": status.ready,
    }
    report["canonical_gates"] = build_semantic_gates(results, status)
    report["comparison_with_lexical"] = build_lexical_comparison(
        report["cases"],
        lexical_baseline_path,
    )
    return report


def render_semantic_report(report: dict[str, Any]) -> str:
    canonical = report["suites"]["canonical"]
    challenge = report["suites"]["challenge"]
    gate_status = "KEÇDİ" if report["canonical_gates"]["passed"] else "UĞURSUZ"
    return "\n".join(
        [
            "Semantic Product Retrieval Baseline",
            f"Dataset: {report['dataset']['dataset_version']} ({report['dataset']['product_count']} məhsul)",
            f"Search: {report['search_version']}",
            (
                f"Embedding: {report['embedding']['deployment']} "
                f"({report['embedding']['dimensions']} dimensions)"
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
                "Qdrant indeksi hazır deyil; əvvəl python -m app.indexing.products index işlədin"
            )
        embeddings = AzureEmbeddingClient.from_settings(settings)
        backend = SemanticProductSearch(catalog, embeddings, store)
        report = build_semantic_report(
            catalog,
            cases,
            backend,
            status,
            embedding_deployment=embeddings.deployment,
            embedding_dimensions=embeddings.dimensions,
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
    parser = argparse.ArgumentParser(description="Azure və Qdrant semantic retrieval evaluator")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Cari nəticəni semantic_qdrant_v1 baseline kimi saxla",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output()
    args = parse_args(argv)
    return run_semantic_evaluation(update_baseline=args.update_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
