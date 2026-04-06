"""Sentence-transformer embedding wrapper (singleton)."""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return float32 embedding matrix of shape (N, dim)."""
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Return float32 embedding of shape (1, dim)."""
    return embed_texts([query])
