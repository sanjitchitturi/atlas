"""Tests for the local answer generator."""
import pytest
from app.generation.local_generator import LocalGenerator
from app.models.schemas import DocumentChunk


def _chunk(text: str, idx: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"c{idx}",
        doc_id="doc1",
        source="test.txt",
        page=1,
        chunk_index=idx,
        text=text,
    )


def test_local_generator_returns_string():
    gen = LocalGenerator()
    chunks = [_chunk("The maximum operating temperature is 350 degrees Celsius.", 0)]
    answer = gen.generate("What is the maximum temperature?", chunks)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_local_generator_grounded_in_context():
    gen = LocalGenerator()
    chunks = [
        _chunk("The pressure relief valve opens at 22 bar set pressure.", 0),
        _chunk("The heat exchanger flow rate is 200 litres per minute.", 1),
    ]
    answer = gen.generate("What is the set pressure of the relief valve?", chunks)
    assert "22" in answer or "pressure" in answer.lower()


def test_local_generator_empty_chunks():
    gen = LocalGenerator()
    answer = gen.generate("What is the temperature?", [])
    assert "cannot find" in answer.lower() or "no" in answer.lower() or "not" in answer.lower()


def test_local_generator_no_relevant_sentences():
    gen = LocalGenerator()
    # Chunks with very short sentences that won't pass length threshold
    chunks = [_chunk("Ok. Yes. No. Fine.", 0)]
    answer = gen.generate("Describe the heat exchanger failover process in detail", chunks)
    assert isinstance(answer, str)


def test_factory_returns_local_by_default(monkeypatch):
    monkeypatch.setenv("GENERATION_MODE", "local")
    from importlib import reload
    import app.core.config as cfg_mod
    reload(cfg_mod)
    from app.generation.factory import get_generator
    from app.generation.local_generator import LocalGenerator
    gen = get_generator()
    assert isinstance(gen, LocalGenerator)


def test_factory_falls_back_to_local_when_key_missing(monkeypatch):
    monkeypatch.setenv("GENERATION_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app.generation.factory import get_generator
    from app.generation.local_generator import LocalGenerator
    gen = get_generator()
    assert isinstance(gen, LocalGenerator)
