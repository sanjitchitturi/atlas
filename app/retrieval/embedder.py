"""Local hashed lexical baseline; optional pretrained semantic embeddings."""

import hashlib
import re
from threading import Lock
import numpy as np
from app.core.config import settings

_model = None
_lock = Lock()


def get_embedder():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    if settings.embedding_backend == "sentence-transformers":
        return (
            get_embedder()
            .encode(texts, show_progress_bar=False, convert_to_numpy=True)
            .astype("float32")
        )
    vectors = np.zeros((len(texts), 2048), dtype="float32")
    stop = set("a an the of to at is does what how and in for".split())
    for i, text in enumerate(texts):
        for token in re.findall(r"\w+", text.casefold()):
            if token in stop:
                continue
            digest = hashlib.sha256(token.encode()).digest()
            vectors[i, int.from_bytes(digest[:4], "little") % 2048] += 1 if digest[4] & 1 else -1
    return vectors


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])
