"""High-level retriever: embed query → FAISS search → return ranked chunks."""
from __future__ import annotations
from app.retrieval.embedder import embed_query
from app.retrieval.index import get_index
from app.models.schemas import DocumentChunk
from app.core.config import settings
from app.core.logging import logger


def retrieve(question: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
    """Return top-k (chunk, score) pairs for *question*."""
    k = top_k or settings.top_k
    q_emb = embed_query(question)
    index = get_index()
    results = index.search(q_emb, k)
    logger.debug(f"Retrieved {len(results)} chunks for query: {question[:60]}")
    return results
