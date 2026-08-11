from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import PROJECT_ROOT, Settings
from app.documents.corpus import DOCUMENT_TEXT_VERSION, DocumentCorpus, DocumentCorpusError
from app.embeddings.azure import AzureEmbeddingClient, EmbeddingError
from app.vectorstores.documents import DocumentSearchHit, QdrantDocumentStore
from app.vectorstores.qdrant import VectorStoreError

DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "evals" / "document_retrieval.json"
DOCUMENT_QUERY_TEXT_VERSION = "document_query_v1"
SEARCH_VERSION = "document_qdrant_v1"
MIN_CASE_COUNT = 30
MIN_POSITIVE_CASES = 20
MIN_CHALLENGE_CASES = 5
MIN_EMPTY_CASES = 5


class DocumentEvalError(RuntimeError):
    pass


class DocumentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,119}$")
    type: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=500)
    expected_chunk_ids: list[str]
    expect_empty: bool = False

    @model_validator(mode="after")
    def validate_expected(self) -> DocumentEvalCase:
        if self.expect_empty and self.expected_chunk_ids:
            raise ValueError("expect_empty=true olduqda expected_chunk_ids boş olmalıdır")
        if not self.expect_empty and not self.expected_chunk_ids:
            raise ValueError("Cavablı case üçün expected_chunk_ids boş ola bilməz")
        if len(self.expected_chunk_ids) != len(set(self.expected_chunk_ids)):
            raise ValueError("expected_chunk_ids təkrarlanmamalıdır")
        return self


def load_cases(path: Path, *, corpus: DocumentCorpus) -> list[DocumentEvalCase]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DocumentEvalError(
            f"Document eval faylı tapılmadı: {path}. Sənədlərə əsasən minimum 30 manual case yaradın."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DocumentEvalError("Document eval JSON etibarsızdır") from exc
    if not isinstance(raw, list):
        raise DocumentEvalError("Document eval faylı JSON array olmalıdır")
    try:
        cases = [DocumentEvalCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise DocumentEvalError("Document eval case schema-sı etibarsızdır") from exc
    ids = [case.query_id for case in cases]
    if len(ids) != len(set(ids)):
        raise DocumentEvalError("Document eval query_id-ləri unikal olmalıdır")
    if len(cases) < MIN_CASE_COUNT:
        raise DocumentEvalError(f"Document eval minimum {MIN_CASE_COUNT} case tələb edir")
    positives = [case for case in cases if not case.expect_empty]
    empty_cases = [case for case in cases if case.expect_empty]
    challenge = [case for case in positives if case.type.casefold() in {"paraphrase", "typo"}]
    if len(positives) < MIN_POSITIVE_CASES:
        raise DocumentEvalError(f"Minimum {MIN_POSITIVE_CASES} cavablı case tələb olunur")
    if len(challenge) < MIN_CHALLENGE_CASES:
        raise DocumentEvalError("Minimum 5 paraphrase/typo case tələb olunur")
    if len(empty_cases) < MIN_EMPTY_CASES:
        raise DocumentEvalError("Minimum 5 cavabsız case tələb olunur")

    available = {chunk.chunk_id for chunk in corpus.chunks}
    unknown = sorted(
        {
            chunk_id
            for case in positives
            for chunk_id in case.expected_chunk_ids
            if chunk_id not in available
        }
    )
    if unknown:
        raise DocumentEvalError(f"Eval daxilində naməlum chunk ID-ləri var: {', '.join(unknown[:5])}")
    positive_counts = {document.document_id: 0 for document in corpus.documents}
    for case in positives:
        for document_id in {chunk_id.split(":", 1)[0] for chunk_id in case.expected_chunk_ids}:
            if document_id in positive_counts:
                positive_counts[document_id] += 1
    uncovered = sorted(document_id for document_id, count in positive_counts.items() if count < 2)
    if uncovered:
        raise DocumentEvalError(
            "Hər sənəd üçün minimum iki müsbət case olmalıdır: " + ", ".join(uncovered)
        )
    return cases


def calibrate_min_score(
    cases: list[DocumentEvalCase],
    candidates: dict[str, list[DocumentSearchHit]],
) -> float:
    negative_scores = [
        hits[0].score
        for case in cases
        if case.expect_empty and (hits := candidates.get(case.query_id, []))
    ]
    if not negative_scores:
        raise DocumentEvalError("Threshold kalibrasiyası üçün nəticəli negative case yoxdur")
    return min(1.0, round(max(negative_scores) + 0.0001, 6))


def build_report(
    *,
    corpus: DocumentCorpus,
    cases: list[DocumentEvalCase],
    candidates: dict[str, list[DocumentSearchHit]],
    collection_name: str,
    embedding_deployment: str,
    embedding_dimensions: int,
    min_score: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive_hits = 0
    reciprocal_rank_total = 0.0
    empty_correct = 0
    covered_documents: set[str] = set()
    positive_count = sum(not case.expect_empty for case in cases)
    empty_count = sum(case.expect_empty for case in cases)
    for case in cases:
        all_hits = candidates.get(case.query_id, [])
        returned = [hit for hit in all_hits if hit.score >= min_score][:5]
        returned_ids = [hit.chunk_id for hit in returned]
        expected = set(case.expected_chunk_ids)
        first_rank = next(
            (index for index, chunk_id in enumerate(returned_ids, start=1) if chunk_id in expected),
            None,
        )
        hit = first_rank is not None
        if not case.expect_empty and hit:
            positive_hits += 1
            reciprocal_rank_total += 1 / first_rank
            covered_documents.update(
                chunk_id.split(":", 1)[0] for chunk_id in expected.intersection(returned_ids)
            )
        if case.expect_empty and not returned_ids:
            empty_correct += 1
        rows.append(
            {
                "query_id": case.query_id,
                "type": case.type,
                "query": case.query,
                "expect_empty": case.expect_empty,
                "expected_chunk_ids": case.expected_chunk_ids,
                "returned_chunk_ids": returned_ids,
                "candidate_scores": [
                    {"chunk_id": hit_item.chunk_id, "score": round(hit_item.score, 6)}
                    for hit_item in all_hits[:5]
                ],
                "hit": hit,
                "first_hit_rank": first_rank,
            }
        )
    hit_at_5 = positive_hits / positive_count if positive_count else 0.0
    mrr = reciprocal_rank_total / positive_count if positive_count else 0.0
    empty_accuracy = empty_correct / empty_count if empty_count else 0.0
    all_documents = {document.document_id for document in corpus.documents}
    gate = (
        hit_at_5 >= 0.95
        and empty_accuracy == 1.0
        and covered_documents == all_documents
    )
    return {
        "dataset": {
            "version": corpus.manifest["dataset_version"],
            "source_checksum": corpus.manifest["source_checksum"],
            "document_count": len(corpus.documents),
            "chunk_count": len(corpus.chunks),
        },
        "search": {
            "version": SEARCH_VERSION,
            "collection_name": collection_name,
            "min_score": min_score,
        },
        "embedding": {
            "deployment": embedding_deployment,
            "dimensions": embedding_dimensions,
            "document_text_version": DOCUMENT_TEXT_VERSION,
            "query_text_version": DOCUMENT_QUERY_TEXT_VERSION,
        },
        "metrics": {
            "case_count": len(cases),
            "positive_count": positive_count,
            "empty_count": empty_count,
            "hit_at_5": round(hit_at_5, 6),
            "mrr": round(mrr, 6),
            "empty_accuracy": round(empty_accuracy, 6),
            "covered_document_count": len(covered_documents),
            "acceptance_gate": gate,
        },
        "cases": rows,
    }


def run(settings: Settings, *, update_baseline: bool) -> int:
    corpus = DocumentCorpus(settings.documents_path)
    corpus.load()
    cases = load_cases(DEFAULT_CASES_PATH, corpus=corpus)
    embeddings = AzureEmbeddingClient.from_settings(settings)
    store = QdrantDocumentStore.from_settings(settings)
    try:
        status = store.status(
            [chunk.chunk_id for chunk in corpus.chunks],
            expected_source_checksum=corpus.manifest["source_checksum"],
            expected_embedding_text_version=DOCUMENT_TEXT_VERSION,
            expected_embedding_deployment=embeddings.deployment,
            expected_embedding_dimensions=embeddings.dimensions,
        )
        if not status.ready:
            raise DocumentEvalError("Document Qdrant collection hazır deyil; əvvəl index və status işlədin")
        vectors = embeddings.embed(
            [case.query for case in cases],
            text_version=DOCUMENT_QUERY_TEXT_VERSION,
        )
        candidates = {
            case.query_id: store.search_candidates(vector, candidate_limit=20)
            for case, vector in zip(cases, vectors, strict=True)
        }
        min_score = calibrate_min_score(cases, candidates)
        report = build_report(
            corpus=corpus,
            cases=cases,
            candidates=candidates,
            collection_name=store.collection_name,
            embedding_deployment=embeddings.deployment,
            embedding_dimensions=embeddings.dimensions,
            min_score=min_score,
        )
        print("Document Retrieval Baseline")
        print(f"Sənədlər: {len(corpus.documents)}, chunk: {len(corpus.chunks)}")
        print(f"Sorğular: {len(cases)}")
        print(f"Hit@5: {report['metrics']['hit_at_5']:.3f}")
        print(f"MRR: {report['metrics']['mrr']:.3f}")
        print(f"Empty accuracy: {report['metrics']['empty_accuracy']:.3f}")
        print(f"Kalibrasiya edilmiş min_score: {min_score:.6f}")
        print(f"Acceptance gate: {'KEÇDİ' if report['metrics']['acceptance_gate'] else 'UĞURSUZ'}")
        if not report["metrics"]["acceptance_gate"]:
            return 1
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        baseline_path = settings.document_baseline_path
        if update_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(serialized, encoding="utf-8")
            print(f"Baseline yeniləndi: {baseline_path}")
            return 0
        try:
            expected = baseline_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print("Baseline yoxdur; --update-baseline istifadə edin", file=sys.stderr)
            return 1
        if expected != serialized:
            print("Nəticə saxlanmış baseline ilə uyğun deyil", file=sys.stderr)
            return 1
        print("Baseline ilə uyğundur.")
        return 0
    finally:
        embeddings.close()
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document retrieval keyfiyyətini ölçür")
    parser.add_argument("--update-baseline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(Settings(), update_baseline=args.update_baseline)  # type: ignore[call-arg]
    except (DocumentCorpusError, DocumentEvalError, EmbeddingError, VectorStoreError, ValidationError) as exc:
        print(f"Xəta: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Xəta: uzaq xidmət əməliyyatı uğursuz oldu", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
