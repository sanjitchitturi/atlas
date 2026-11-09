"""FastAPI router: health check."""

from fastapi import APIRouter
from app.retrieval.index import get_index
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    index = get_index()
    return HealthResponse(
        status="ok",
        index_size=index.size,
        corpus_generation=index.generation,
        embedding_model=(
            settings.embedding_model
            if settings.embedding_backend == "sentence-transformers"
            else "hashed-lexical-2048"
        ),
        generation_mode=settings.generation_mode,
    )
