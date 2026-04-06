"""Tests for the vector index and retrieval pipeline."""
import pytest
import numpy as np
from app.retrieval.index import VectorIndex
from app.models.schemas import DocumentChunk


def _make_chunk(text: str, idx: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{idx}",
        doc_id="test-doc",
        source="test.txt",
        page=1,
        chunk_index=idx,
        text=text,
    )


def test_index_build_and_search():
    from app.retrieval.embedder import embed_texts, embed_query

    texts = [
        "The heat exchanger operates at 350 degrees Celsius.",
        "The pressure relief valve opens at 22 bar.",
        "The steel frame can bear 500 kN per column.",
    ]
    chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
    embeddings = embed_texts(texts)

    idx = VectorIndex()
    idx.build(embeddings, chunks)

    assert idx.size == 3

    q_emb = embed_query("What temperature does the heat exchanger operate at?")
    results = idx.search(q_emb, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert "heat exchanger" in top_chunk.text.lower()
    assert top_score > 0.0


def test_index_add():
    from app.retrieval.embedder import embed_texts

    idx = VectorIndex()
    texts1 = ["Document one content here."]
    chunks1 = [_make_chunk(texts1[0], 0)]
    emb1 = embed_texts(texts1)
    idx.build(emb1, chunks1)
    assert idx.size == 1

    texts2 = ["Document two content here."]
    chunks2 = [_make_chunk(texts2[0], 1)]
    emb2 = embed_texts(texts2)
    idx.add(emb2, chunks2)
    assert idx.size == 2


def test_empty_index_returns_no_results():
    from app.retrieval.embedder import embed_query
    idx = VectorIndex()
    q = embed_query("anything")
    results = idx.search(q, top_k=5)
    assert results == []


def test_index_save_and_load(tmp_path):
    from app.retrieval.embedder import embed_texts, embed_query
    texts = ["Save and load test sentence for FAISS."]
    chunks = [_make_chunk(texts[0], 0)]
    emb = embed_texts(texts)

    idx = VectorIndex()
    idx.build(emb, chunks)
    idx.save(tmp_path)

    idx2 = VectorIndex()
    loaded = idx2.load(tmp_path)
    assert loaded is True
    assert idx2.size == 1

    q = embed_query("save and load test")
    results = idx2.search(q, top_k=1)
    assert len(results) == 1
