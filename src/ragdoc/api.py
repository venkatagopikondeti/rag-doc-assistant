"""FastAPI service exposing ingest / query endpoints."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .pipeline import RagPipeline
from .store import VectorStore

INDEX_DIR = Path(os.getenv("INDEX_DIR", "artifacts/index"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

app = FastAPI(title="RAG Document Assistant", version="0.1.0")
pipeline = RagPipeline()

if (INDEX_DIR / "chunks.json").exists():
    pipeline.store = VectorStore.load(INDEX_DIR)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(pipeline.store)}


@app.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    # Browsers normally send a basename, but API clients can supply path-like
    # names. Normalising both slash styles prevents writes outside the temp dir.
    filename = Path((file.filename or "upload.txt").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix}")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / filename
        written = 0
        with target.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="uploaded file is too large")
                fh.write(chunk)
        added = pipeline.ingest_path(target)
        pipeline.store.save(INDEX_DIR)
    return {"chunks_added": added, "chunks_total": len(pipeline.store)}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    if len(pipeline.store) == 0:
        raise HTTPException(status_code=409, detail="index is empty - ingest documents first")
    return pipeline.answer(request.question, k=request.k)
