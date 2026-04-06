"""RAG pipeline: retrieve chunks → generate answer → attach citations."""
from __future__ import annotations
from app.retrieval.retriever import retrieve
from app.generation.factory import get_generator
from app.models.schemas import QueryRequest, QueryResponse, SourceReference
from app.core.config import settings
from app.core.logging import logger


def answer_question(request: QueryRequest) -> QueryResponse:
    # 1. Retrieve
    results = retrieve(request.question, top_k=request.top_k)

    chunks = [chunk for chunk, _score in results]

    # 2. Generate
    generator = get_generator()
    answer = generator.generate(request.question, chunks)

    # 3. Build source references
    sources: list[SourceReference] = []
    seen_chunks: set[str] = set()
    for chunk, _score in results:
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)
        sources.append(
            SourceReference(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                snippet=chunk.text[:200],
            )
        )

    logger.info(f"Answered: '{request.question[:60]}' | sources={len(sources)}")

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        mode=settings.generation_mode,
    )
