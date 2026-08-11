from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DOCUMENT_DATASET_VERSION = "1.0.0"
DOCUMENT_TEXT_VERSION = "document_markdown_v1"
TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200

DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,119}$")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class DocumentCorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarkdownDocument:
    document_id: str
    filename: str
    title: str
    text: str
    checksum: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    title: str
    heading: str
    chunk_index: int
    text: str
    document_checksum: str

    @property
    def embedding_text(self) -> str:
        parts = [self.title]
        if self.heading and self.heading != self.title:
            parts.append(self.heading)
        parts.append(self.text)
        return "\n".join(parts)


class DocumentCorpus:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.documents: tuple[MarkdownDocument, ...] = ()
        self.chunks: tuple[DocumentChunk, ...] = ()
        self.manifest: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return bool(self.documents and self.chunks and self.manifest)

    def load(self) -> None:
        if not self.source_path.exists() or not self.source_path.is_dir():
            raise DocumentCorpusError(f"Sənəd qovluğu tapılmadı: {self.source_path}")
        paths = sorted(self.source_path.glob("*.md"), key=lambda path: path.name.casefold())
        if not paths:
            raise DocumentCorpusError("data/documents/source qovluğunda Markdown sənədi yoxdur")

        documents: list[MarkdownDocument] = []
        seen_ids: set[str] = set()
        for path in paths:
            document_id = path.stem
            normalized_id = document_id.casefold()
            if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
                raise DocumentCorpusError(
                    f"Etibarsız sənəd filename-i: {path.name}; yalnız kiçik hərf, rəqəm, _ və - istifadə edin"
                )
            if normalized_id in seen_ids:
                raise DocumentCorpusError(f"Təkrarlanan document_id: {document_id}")
            seen_ids.add(normalized_id)
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
            except UnicodeDecodeError as exc:
                raise DocumentCorpusError(f"{path.name} UTF-8 formatında deyil") from exc
            if not text:
                raise DocumentCorpusError(f"Boş Markdown sənədi qəbul edilmir: {path.name}")
            title = _first_h1(text)
            if title is None:
                raise DocumentCorpusError(f"{path.name} daxilində '# Başlıq' yoxdur")
            documents.append(
                MarkdownDocument(
                    document_id=document_id,
                    filename=path.name,
                    title=title,
                    text=text,
                    checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )

        chunks = tuple(chunk for document in documents for chunk in _chunk_document(document))
        if not chunks:
            raise DocumentCorpusError("Markdown sənədlərindən istifadə edilə bilən mətn çıxmadı")
        self.documents = tuple(documents)
        self.chunks = chunks
        self.manifest = _build_manifest(self.documents, self.chunks)

    def write_manifest(self, path: Path | None = None) -> Path:
        if not self.ready:
            raise DocumentCorpusError("Manifest yazılmazdan əvvəl corpus yüklənməlidir")
        target = path or self.source_path.parent / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _first_h1(text: str) -> str | None:
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip().strip("#").strip()
            return title or None
    return None


def _chunk_document(document: MarkdownDocument) -> tuple[DocumentChunk, ...]:
    sections = _markdown_sections(document.text, document.title)
    raw_chunks: list[tuple[str, str]] = []
    for heading, blocks in sections:
        current = ""
        for block in blocks:
            for piece in _split_long_block(block):
                candidate = piece if not current else f"{current}\n\n{piece}"
                should_flush = bool(
                    current
                    and (
                        len(candidate) > MAX_CHUNK_CHARS
                        or (len(current) >= TARGET_CHUNK_CHARS and len(candidate) > TARGET_CHUNK_CHARS)
                    )
                )
                if should_flush:
                    raw_chunks.append((heading, current.strip()))
                    overlap = _overlap_tail(current)
                    current = f"{overlap}\n\n{piece}".strip() if overlap else piece
                else:
                    current = candidate
        if current.strip():
            raw_chunks.append((heading, current.strip()))

    chunks = []
    for index, (heading, text) in enumerate(raw_chunks, start=1):
        if len(text) > MAX_CHUNK_CHARS:
            raise DocumentCorpusError("Daxili chunk ölçüsü maksimum həddi keçdi")
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:{index:04d}",
                document_id=document.document_id,
                filename=document.filename,
                title=document.title,
                heading=heading,
                chunk_index=index,
                text=text,
                document_checksum=document.checksum,
            )
        )
    return tuple(chunks)


def _markdown_sections(text: str, title: str) -> list[tuple[str, list[str]]]:
    headings: list[str] = [title]
    sections: list[tuple[str, list[str]]] = []
    blocks: list[str] = []
    paragraph: list[str] = []
    seen_title = False

    def flush_paragraph() -> None:
        if paragraph:
            block = "\n".join(paragraph).strip()
            if block:
                blocks.append(block)
            paragraph.clear()

    def flush_section() -> None:
        flush_paragraph()
        if blocks:
            sections.append((" > ".join(headings), list(blocks)))
            blocks.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip().strip("#").strip()
            if level == 1 and not seen_title:
                seen_title = True
                continue
            flush_section()
            while len(headings) >= level:
                headings.pop()
            while len(headings) < level - 1:
                headings.append(headings[-1] if headings else title)
            headings.append(heading_text)
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
    flush_section()
    return sections


def _split_long_block(block: str) -> list[str]:
    remaining = block.strip()
    pieces: list[str] = []
    while len(remaining) > MAX_CHUNK_CHARS:
        split_at = remaining.rfind(" ", 0, MAX_CHUNK_CHARS + 1)
        if split_at < TARGET_CHUNK_CHARS // 2:
            split_at = MAX_CHUNK_CHARS
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _overlap_tail(text: str) -> str:
    if len(text) <= CHUNK_OVERLAP_CHARS:
        return text
    tail = text[-CHUNK_OVERLAP_CHARS:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail.strip()


def _build_manifest(
    documents: tuple[MarkdownDocument, ...],
    chunks: tuple[DocumentChunk, ...],
) -> dict[str, Any]:
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
    document_entries = [
        {
            "document_id": document.document_id,
            "filename": document.filename,
            "title": document.title,
            "sha256": document.checksum,
            "chunk_count": chunk_counts.get(document.document_id, 0),
        }
        for document in documents
    ]
    chunk_entries = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "heading": chunk.heading,
            "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        }
        for chunk in chunks
    ]
    checksum_payload = {
        "dataset_version": DOCUMENT_DATASET_VERSION,
        "text_version": DOCUMENT_TEXT_VERSION,
        "chunking": {
            "target_chars": TARGET_CHUNK_CHARS,
            "max_chars": MAX_CHUNK_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "documents": document_entries,
        "chunks": chunk_entries,
    }
    encoded = json.dumps(checksum_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return {
        **checksum_payload,
        "source_checksum": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }


def chunk_as_dict(chunk: DocumentChunk) -> dict[str, Any]:
    return asdict(chunk)
