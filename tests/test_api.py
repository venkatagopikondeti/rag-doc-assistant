import io

from fastapi.testclient import TestClient

from ragdoc import api


def client() -> TestClient:
    api.pipeline.store.chunks.clear()
    api.pipeline.store._matrix = None
    api.pipeline.store._index = None
    return TestClient(api.app)


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_before_ingest_is_conflict():
    response = client().post("/query", json={"question": "hi"})
    assert response.status_code == 409


def test_unsupported_upload_rejected():
    files = {"file": ("model.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")}
    assert client().post("/ingest", files=files).status_code == 400


def test_ingest_then_query(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX_DIR", tmp_path / "index")
    c = client()
    payload = (
        b"Airflow orchestrates data pipelines as directed acyclic graphs.\n\n"
        b"FAISS is a vector similarity search library."
    )
    upload = c.post("/ingest", files={"file": ("notes.md", io.BytesIO(payload), "text/markdown")})
    assert upload.status_code == 200
    assert upload.json()["chunks_added"] >= 1

    answer = c.post("/query", json={"question": "what is faiss", "k": 2})
    assert answer.status_code == 200
    body = answer.json()
    assert body["sources"]
    assert "faiss" in body["answer"].lower()
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "index" / "vectors.npy").exists()


def test_upload_filename_is_reduced_to_a_basename(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX_DIR", tmp_path / "index")
    response = client().post(
        "/ingest",
        files={"file": ("../../notes.md", io.BytesIO(b"safe content"), "text/markdown")},
    )
    assert response.status_code == 200
    assert api.pipeline.store.chunks[0].source == "notes.md"


def test_oversized_upload_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = client().post(
        "/ingest", files={"file": ("notes.md", io.BytesIO(b"too large"), "text/markdown")}
    )
    assert response.status_code == 413
