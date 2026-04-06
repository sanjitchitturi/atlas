"""Tests for configuration loading."""
import pytest
from app.core.config import Settings


def test_default_settings():
    s = Settings()
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.top_k == 5
    assert s.generation_mode == "local"


def test_settings_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("GENERATION_MODE", "openai")
    s = Settings()
    assert s.chunk_size == 256
    assert s.top_k == 3
    assert s.generation_mode == "openai"


def test_data_dirs_are_paths():
    from pathlib import Path
    s = Settings()
    assert isinstance(s.data_dir, Path)
    assert isinstance(s.index_dir, Path)
