"""FAISS vector index with persistence and metadata store."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from app.models.schemas import DocumentChunk
from app.core.config import settings
from app.core.logging import logger

_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.pkl"


class VectorIndex:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []

    def build(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = chunks
        logger.info(f"Index built with {self.index.ntotal} vectors (dim={dim})")

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if self.index is None:
            self.build(embeddings, chunks)
            return
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors; total={self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[DocumentChunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_embedding.copy().astype("float32")
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Path | None = None) -> None:
        d = directory or settings.index_dir
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / _INDEX_FILE))
        with open(d / _META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Index saved to {d}")

    def load(self, directory: Path | None = None) -> bool:
        d = directory or settings.index_dir
        idx_path = d / _INDEX_FILE
        meta_path = d / _META_FILE
        if not idx_path.exists() or not meta_path.exists():
            logger.warning("No saved index found")
            return False
        self.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return True

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0


# Singleton

_vector_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex()
        _vector_index.load()
    return _vector_index
