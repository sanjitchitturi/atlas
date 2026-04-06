"""Tests for the chunking module."""
import pytest
from app.ingestion.chunker import chunk_text


def test_basic_chunking():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, source="test.txt", doc_id="doc1", chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 100


def test_overlap_creates_shared_words():
    text = " ".join([str(i) for i in range(200)])
    chunks = chunk_text(text, source="test.txt", doc_id="doc1", chunk_size=50, chunk_overlap=10)
    if len(chunks) >= 2:
        end_of_first = set(chunks[0].text.split()[-10:])
        start_of_second = set(chunks[1].text.split()[:10])
        assert len(end_of_first & start_of_second) > 0


def test_small_text_single_chunk():
    text = "This is a short document with very few words."
    chunks = chunk_text(text, source="short.txt", doc_id="doc2", chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_metadata():
    text = " ".join(["hello"] * 200)
    chunks = chunk_text(text, source="meta_test.txt", doc_id="doc3", page=3)
    for c in chunks:
        assert c.source == "meta_test.txt"
        assert c.doc_id == "doc3"
        assert c.page == 3
        assert c.chunk_id != ""


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", source="empty.txt", doc_id="doc4")
    assert chunks == []
