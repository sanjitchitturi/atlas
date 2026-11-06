"""FAISS cosine search with a single atomic, pickle-free persistence snapshot."""

import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import faiss
import numpy as np
from app.core.config import settings
from app.models.schemas import DocumentChunk


class VectorIndex:
    def __init__(self):
        self.index = None
        self.chunks = []
        self._lock = RLock()

    def build(self, embeddings, chunks):
        with self._lock:
            values = np.array(embeddings, dtype="float32", copy=True)
            if values.ndim != 2 or len(values) != len(chunks) or not np.isfinite(values).all():
                raise ValueError("Invalid embedding matrix")
            index = faiss.IndexFlatIP(values.shape[1])
            faiss.normalize_L2(values)
            index.add(values)
            self.index, self.chunks = index, list(chunks)

    def add(self, embeddings, chunks):
        with self._lock:
            keys = {(c.source, c.page, c.chunk_index, c.text) for c in self.chunks}
            keep = []
            for i, c in enumerate(chunks):
                key = (c.source, c.page, c.chunk_index, c.text)
                if key not in keys:
                    keep.append(i)
                    keys.add(key)
            if not keep:
                return
            if self.size + len(keep) > settings.max_index_chunks:
                raise ValueError("Index capacity reached; use a separate workspace")
            matrix = np.array(embeddings, dtype="float32", copy=True)
            if matrix.ndim != 2 or len(matrix) != len(chunks):
                raise ValueError("Embedding rows must match chunks")
            values = matrix[keep]
            selected = [chunks[i] for i in keep]
            if self.index is None:
                self.build(values, selected)
            else:
                if values.shape[1] != self.index.d:
                    raise ValueError("Embedding dimension changed; use a fresh INDEX_DIR")
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite embeddings")
                faiss.normalize_L2(values)
                self.index.add(values)
                self.chunks.extend(selected)

    def search(self, query_embedding, top_k):
        with self._lock:
            if self.index is None or not self.chunks:
                return []
            if top_k < 1:
                raise ValueError("top_k must be positive")
            q = np.array(query_embedding, dtype="float32", copy=True)
            if q.shape != (1, self.index.d):
                raise ValueError("Query embedding dimension differs from index")
            if not np.isfinite(q).all():
                raise ValueError("Query embedding must be finite")
            faiss.normalize_L2(q)
            scores, indices = self.index.search(q, min(top_k, self.size))
            return [
                (self.chunks[int(i)], float(s))
                for s, i in zip(scores[0], indices[0])
                if i >= 0 and s > 0
            ]

    def save(self, directory: Path | None = None):
        with self._lock:
            if self.index is None:
                return
            d = directory or settings.index_dir
            d.mkdir(parents=True, exist_ok=True)
            temporary = d / (uuid4().hex + ".tmp")
            try:
                with temporary.open("wb") as f:
                    np.savez_compressed(
                        f,
                        vectors=self.index.reconstruct_n(0, self.size),
                        metadata=np.array(json.dumps([c.model_dump() for c in self.chunks])),
                        model=np.array(settings.embedding_backend + ":" + settings.embedding_model),
                    )
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, d / "snapshot.npz")
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, directory: Path | None = None):
        with self._lock:
            path = (directory or settings.index_dir) / "snapshot.npz"
            if not path.exists():
                return False
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["model"])
                    != settings.embedding_backend + ":" + settings.embedding_model
                ):
                    raise ValueError("Embedding configuration changed; use a fresh INDEX_DIR")
                chunks = [
                    DocumentChunk.model_validate(c) for c in json.loads(str(saved["metadata"]))
                ]
                self.build(saved["vectors"], chunks)
            return True

    @property
    def size(self):
        return len(self.chunks)


_vector_index = None
_singleton_lock = RLock()


def get_index():
    global _vector_index
    with _singleton_lock:
        if _vector_index is None:
            index = VectorIndex()
            index.load()
            _vector_index = index
        return _vector_index
