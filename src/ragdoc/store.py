"""Vector store with a FAISS backend and a NumPy fallback.

FAISS is the production path; the NumPy implementation keeps the package
importable (and testable) anywhere, including slim CI images.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, get_embedder

try:  # pragma: no cover - depends on environment
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._index = None

    # ---------------------------------------------------------------- build
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.encode([c.text for c in chunks])
        self.chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._index = None  # invalidate

    def _ensure_index(self) -> None:
        if not _HAS_FAISS or self._matrix is None or self._index is not None:
            return
        index = faiss.IndexFlatIP(self._matrix.shape[1])
        index.add(self._matrix)
        self._index = index

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        qvec = self.embedder.encode([query])
        self._ensure_index()
        if self._index is not None:
            scores, ids = self._index.search(qvec, k)
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ qvec[0]).tolist()
            order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
            pairs = ((i, sims[i]) for i in order)
        return [(self.chunks[i], float(s)) for i, s in pairs if i >= 0]

    # ----------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8"
        )
        with (path / "vectors.pkl").open("wb") as fh:
            pickle.dump(self._matrix, fh)

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "VectorStore":
        path = Path(directory)
        store = cls(embedder)
        raw = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk(**item) for item in raw]
        with (path / "vectors.pkl").open("rb") as fh:
            store._matrix = pickle.load(fh)
        return store

    def __len__(self) -> int:
        return len(self.chunks)
