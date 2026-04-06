"""Split cleaned text into overlapping chunks."""
from __future__ import annotations
import uuid
from app.models.schemas import DocumentChunk
from app.core.config import settings


def chunk_text(
    text: str,
    source: str,
    doc_id: str,
    page: int = 1,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Split *text* into overlapping word-boundary chunks."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    words = text.split()
    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                source=source,
                page=page,
                chunk_index=idx,
                text=chunk_text,
            )
        )
        if end == len(words):
            break
        start += size - overlap
        idx += 1

    return chunks
