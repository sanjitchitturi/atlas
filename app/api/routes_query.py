"""FastAPI router: question answering."""
from fastapi import APIRouter, HTTPException
from app.generation.pipeline import answer_question
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.index import get_index
from app.core.logging import logger

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Ask a question; get a grounded answer with source citations."""
    index = get_index()
    if index.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please ingest documents first.",
        )
    try:
        return answer_question(request)
    except Exception as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
