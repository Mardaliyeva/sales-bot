from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DocumentBaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class DocumentBaseline:
    source_checksum: str
    collection_name: str
    embedding_deployment: str
    embedding_dimensions: int
    min_score: float


def load_document_baseline(
    path: Path,
    *,
    expected_source_checksum: str,
    expected_collection_name: str,
    expected_embedding_deployment: str,
    expected_embedding_dimensions: int,
) -> DocumentBaseline:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DocumentBaselineError(f"Document baseline tapılmadı: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DocumentBaselineError("Document baseline JSON etibarsızdır") from exc
    if not isinstance(payload, dict):
        raise DocumentBaselineError("Document baseline obyekt olmalıdır")
    dataset = payload.get("dataset")
    search = payload.get("search")
    embedding = payload.get("embedding")
    if not all(isinstance(item, dict) for item in (dataset, search, embedding)):
        raise DocumentBaselineError("Document baseline metadata-sı natamamdır")
    source_checksum = dataset.get("source_checksum")
    collection_name = search.get("collection_name")
    min_score = search.get("min_score")
    deployment = embedding.get("deployment")
    dimensions = embedding.get("dimensions")
    if source_checksum != expected_source_checksum:
        raise DocumentBaselineError("Document baseline source checksum ilə uyğun deyil")
    if collection_name != expected_collection_name:
        raise DocumentBaselineError("Document baseline collection adı ilə uyğun deyil")
    if deployment != expected_embedding_deployment:
        raise DocumentBaselineError("Document baseline embedding deployment ilə uyğun deyil")
    if dimensions != expected_embedding_dimensions:
        raise DocumentBaselineError("Document baseline embedding dimensions ilə uyğun deyil")
    if not isinstance(min_score, (int, float)) or not 0 <= float(min_score) <= 1:
        raise DocumentBaselineError("Document baseline min_score etibarsızdır")
    return DocumentBaseline(
        source_checksum=str(source_checksum),
        collection_name=str(collection_name),
        embedding_deployment=str(deployment),
        embedding_dimensions=int(dimensions),
        min_score=float(min_score),
    )

