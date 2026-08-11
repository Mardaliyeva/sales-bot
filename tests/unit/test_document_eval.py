from __future__ import annotations

from app.evals.document_retrieval import DocumentEvalCase, calibrate_min_score
from app.tools.schemas import DocumentSearchArguments
from app.vectorstores.documents import DocumentSearchHit


def make_hit(chunk_id: str, score: float) -> DocumentSearchHit:
    return DocumentSearchHit(chunk_id=chunk_id, score=score, payload={})


def test_document_threshold_is_calibrated_above_negative_score() -> None:
    cases = [
        DocumentEvalCase(
            query_id="positive_001",
            type="semantic",
            query="çatdırılma",
            expected_chunk_ids=["delivery:0001"],
        ),
        DocumentEvalCase(
            query_id="empty_001",
            type="empty",
            query="olmayan məlumat",
            expected_chunk_ids=[],
            expect_empty=True,
        ),
    ]
    candidates = {
        "positive_001": [make_hit("delivery:0001", 0.8)],
        "empty_001": [make_hit("delivery:0002", 0.45)],
    }
    assert calibrate_min_score(cases, candidates) == 0.4501


def test_document_search_arguments_reject_extra_fields() -> None:
    try:
        DocumentSearchArguments.model_validate({"query": "zəmanət", "document_id": "x"})
    except Exception:
        return
    raise AssertionError("extra field qəbul edilməməli idi")
