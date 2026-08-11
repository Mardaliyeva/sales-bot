from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.documents.corpus import MAX_CHUNK_CHARS, DocumentCorpus, DocumentCorpusError


def write_document(path: Path, name: str, text: str) -> None:
    (path / name).write_text(text, encoding="utf-8")


def test_markdown_corpus_is_deterministic_and_heading_aware(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_document(
        source,
        "delivery_policy.md",
        "# Çatdırılma qaydaları\n\n## Bakı\n\nÇatdırılma bir iş günü ərzində edilir.\n\n"
        "## Regionlar\n\nRegionlara çatdırılma üç iş günü çəkir.",
    )
    first = DocumentCorpus(source)
    first.load()
    second = DocumentCorpus(source)
    second.load()

    assert first.manifest == second.manifest
    assert first.chunks == second.chunks
    assert first.documents[0].document_id == "delivery_policy"
    assert first.documents[0].title == "Çatdırılma qaydaları"
    assert any("Bakı" in chunk.heading for chunk in first.chunks)
    assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in first.chunks)
    manifest_path = first.write_manifest()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["document_count"] == 1


def test_long_markdown_is_split_without_exceeding_limit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paragraphs = "\n\n".join(f"Qayda {index}: " + "məlumat " * 35 for index in range(12))
    write_document(source, "credit_terms.md", f"# Kredit şərtləri\n\n{paragraphs}")
    corpus = DocumentCorpus(source)
    corpus.load()

    assert len(corpus.chunks) > 1
    assert all(0 < len(chunk.text) <= MAX_CHUNK_CHARS for chunk in corpus.chunks)
    assert [chunk.chunk_id for chunk in corpus.chunks] == [
        f"credit_terms:{index:04d}" for index in range(1, len(corpus.chunks) + 1)
    ]


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("Invalid Name.md", "# Başlıq\n\nMətn"),
        ("valid.md", "Başlıqsız mətn"),
        ("empty.md", ""),
    ],
)
def test_invalid_markdown_documents_are_rejected(tmp_path: Path, name: str, text: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_document(source, name, text)
    with pytest.raises(DocumentCorpusError):
        DocumentCorpus(source).load()

