"""End-to-end ingestion: load → clean → chunk → embed → index."""
from __future__ import annotations
import uuid
from pathlib import Path
from app.ingestion.loader import load_file
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_text
from app.models.schemas import DocumentChunk, IngestedDocument
from app.core.logging import logger


def ingest_file(path: Path) -> tuple[list[DocumentChunk], IngestedDocument]:
    """Ingest a single file and return its chunks + metadata."""
    doc_id = str(uuid.uuid4())
    source = path.name
    pages = load_file(path)

    all_chunks: list[DocumentChunk] = []
    for page_num, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        chunks = chunk_text(
            text=cleaned,
            source=source,
            doc_id=doc_id,
            page=page_num,
        )
        all_chunks.extend(chunks)

    meta = IngestedDocument(
        doc_id=doc_id,
        source=source,
        num_chunks=len(all_chunks),
    )
    logger.info(f"Ingested '{source}': {len(all_chunks)} chunks")
    return all_chunks, meta


def ingest_directory(directory: Path) -> tuple[list[DocumentChunk], list[IngestedDocument]]:
    """Ingest all supported files in *directory*."""
    supported = {".pdf", ".docx", ".txt", ".md", ".markdown"}
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in supported]
    all_chunks: list[DocumentChunk] = []
    all_meta: list[IngestedDocument] = []
    for f in files:
        try:
            chunks, meta = ingest_file(f)
            all_chunks.extend(chunks)
            all_meta.append(meta)
        except Exception as exc:
            logger.error(f"Failed to ingest '{f.name}': {exc}")
    return all_chunks, all_meta
