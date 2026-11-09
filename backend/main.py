from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from app.core.request_security import RequestSecurityMiddleware


from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_health import router as health_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Atlas API starting up")
    settings.ensure_dirs()
    yield


app = FastAPI(
    title="Atlas",
    description="RAG-powered Q&A over your engineering documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    RequestSecurityMiddleware,
    allowed_origins=settings.cors_origins,
    max_bytes=(settings.max_upload_mb + 1) * 1024 * 1024,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
