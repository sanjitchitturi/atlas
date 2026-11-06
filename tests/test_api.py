from fastapi.testclient import TestClient
import pytest
from backend.main import app
from app.core.config import settings
from app.retrieval import index
from app.ingestion.chunker import chunk_text
from app.retrieval.index import VectorIndex


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "documents")
    monkeypatch.setattr(settings, "embedding_backend", "hash")
    monkeypatch.setattr(settings, "generation_mode", "local")
    monkeypatch.setattr(index, "_vector_index", None)
    with TestClient(app) as client:
        yield client


def test_upload_query_persistence_and_dedup(client):
    data = (
        b"The heat exchanger maximum temperature is 350 Celsius. Pressure relief opens at 22 bar."
    )

    def upload():
        return client.post(
            "/ingest/upload", files=[("files", ("../../guide.txt", data, "text/plain"))]
        )

    assert upload().status_code == 200
    assert upload().json()["index_size"] == 1
    answer = client.post(
        "/query/ask", json={"question": "What is the heat exchanger temperature?"}
    ).json()
    assert answer["mode"] == "local"
    assert answer["sources"][0]["source"] == "guide.txt"
    assert "350" in answer["answer"]
    restored = VectorIndex()
    assert restored.load() and restored.size == 1
    assert not list(settings.data_dir.glob("*"))


def test_validation_and_samples(client):
    assert client.post("/ingest/upload", files=[("files", ("bad.exe", b"x"))]).status_code == 400
    assert client.post("/ingest/upload", files=[("files", ("empty.txt", b""))]).status_code == 422
    assert client.post("/ingest/directory?directory=/etc").status_code == 400
    first = client.post("/ingest/directory")
    assert first.status_code == 200
    assert first.json()["total_chunks"] > 0
    assert client.post("/ingest/directory").json()["index_size"] == first.json()["index_size"]


def test_chunk_settings_guard():
    with pytest.raises(ValueError):
        chunk_text("test words", "a", "b", chunk_size=2, chunk_overlap=2)
    assert len(chunk_text("one two three four", "a", "b", chunk_size=2, chunk_overlap=0)) == 2
