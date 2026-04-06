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
        embedding_model=settings.embedding_model,
        generation_mode=settings.generation_mode,
    )
