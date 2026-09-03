"""FastAPI service exposing ingest / query endpoints."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .pipeline import RagPipeline
from .store import VectorStore

INDEX_DIR = Path(os.getenv("INDEX_DIR", "artifacts/index"))

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
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix}")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / (file.filename or "upload.txt")
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        added = pipeline.ingest_path(target)
    return {"chunks_added": added, "chunks_total": len(pipeline.store)}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    if len(pipeline.store) == 0:
        raise HTTPException(status_code=409, detail="index is empty - ingest documents first")
    return pipeline.answer(request.question, k=request.k)
