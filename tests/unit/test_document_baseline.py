from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.documents.baseline import DocumentBaselineError, load_document_baseline


def write_baseline(path: Path, *, checksum: str = "abc") -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": {"source_checksum": checksum},
                "search": {
                    "collection_name": "sales_bot_documents_v1",
                    "min_score": 0.61,
                },
                "embedding": {"deployment": "embedding-test", "dimensions": 3072},
            }
        ),
        encoding="utf-8",
    )


def test_document_baseline_is_bound_to_dataset_and_embedding(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(path)
    baseline = load_document_baseline(
        path,
        expected_source_checksum="abc",
        expected_collection_name="sales_bot_documents_v1",
        expected_embedding_deployment="embedding-test",
        expected_embedding_dimensions=3072,
    )
    assert baseline.min_score == 0.61


def test_document_baseline_rejects_checksum_drift(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(path, checksum="old")
    with pytest.raises(DocumentBaselineError):
        load_document_baseline(
            path,
            expected_source_checksum="new",
            expected_collection_name="sales_bot_documents_v1",
            expected_embedding_deployment="embedding-test",
            expected_embedding_dimensions=3072,
        )

