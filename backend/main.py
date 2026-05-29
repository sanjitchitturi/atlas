import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

Path("logs").mkdir(exist_ok=True)

from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_health import router as health_router
from app.core.config import settings
from app.core.logging import logger

app = FastAPI(
    title="Atlas",
    description="RAG-powered Q&A over your engineering documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)


@app.on_event("startup")
async def startup_event():
    logger.info("Atlas RAG API starting up…")
    settings.ensure_dirs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
