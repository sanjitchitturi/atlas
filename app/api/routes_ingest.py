"""FastAPI router: document upload and ingestion."""

from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ingestion.pipeline import ingest_file, ingest_directory
from app.retrieval.embedder import embed_texts
from app.retrieval.index import get_index
from app.models.schemas import IngestResponse
from app.utils.file_utils import is_allowed, save_upload
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(files: list[UploadFile] = File(...)):
    """Upload one or more documents and add them to the vector index."""
    all_chunks = []
    all_meta = []

    if len(files) > 10:
        raise HTTPException(413, "Upload at most 10 files at a time.")
    for upload in files:
        if not is_allowed(upload.filename or ""):
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported: {upload.filename}",
            )
        file_bytes = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(file_bytes) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "File exceeds upload limit.")
        if not file_bytes:
            raise HTTPException(422, "File is empty.")
        saved_path = save_upload(file_bytes, upload.filename)
        try:
            chunks, meta = ingest_file(saved_path)
            source = Path((upload.filename or "document").replace("\\", "/")).name
            meta.source = source
            for chunk in chunks:
                chunk.source = source
            all_chunks.extend(chunks)
            all_meta.append(meta)
        except Exception as exc:
            logger.error(f"Ingest error for {upload.filename}: {exc}")
            raise HTTPException(status_code=422, detail="Could not parse this document.") from exc
        finally:
            saved_path.unlink(missing_ok=True)

    if not all_chunks:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the uploaded files.",
        )

    # Embed and index
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)
    index = get_index()
    index.add(embeddings, all_chunks)
    index.save()

    return IngestResponse(
        documents=all_meta,
        total_chunks=len(all_chunks),
        index_size=index.size,
    )


@router.post("/directory", response_model=IngestResponse)
async def ingest_from_directory(directory: str | None = None):
    """Ingest all documents in the configured (or given) data directory."""
    d = Path(__file__).resolve().parents[2] / "sample_data"
    d = d.resolve()
    if directory is not None:
        raise HTTPException(400, "Only bundled sample ingestion is allowed through this endpoint.")
    if not d.exists():
        raise HTTPException(status_code=404, detail="Bundled sample documents are unavailable.")

    chunks, meta_list = ingest_directory(d)
    if not chunks:
        raise HTTPException(status_code=422, detail="No content extracted from directory.")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    index = get_index()
    index.add(embeddings, chunks)
    index.save()

    return IngestResponse(
        documents=meta_list,
        total_chunks=len(chunks),
        index_size=index.size,
    )
