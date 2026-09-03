"""Pluggable embedding backends.

The default backend is dependency-light and deterministic so the retrieval
layer can be unit-tested offline / in CI. Set EMBEDDING_BACKEND=sentence
to use a real sentence-transformers model in production.
"""
from __future__ import annotations

import os
import re
import zlib
from abc import ABC, abstractmethod

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an L2-normalised (n, dim) float32 matrix."""

    @staticmethod
    def _normalise(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder.

    Not semantically strong, but it needs no model download, which keeps the
    test suite fast and hermetic. Good enough for keyword-ish retrieval.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # crc32 rather than hash(): stable across processes, so a
                # persisted index stays valid after a restart.
                mat[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        return self._normalise(mat)


class SentenceTransformerEmbedder(Embedder):
    """Real semantic embeddings. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._normalise(np.asarray(vecs, dtype=np.float32))


def get_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "hashing").lower()
    if backend == "sentence":
        return SentenceTransformerEmbedder(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return HashingEmbedder(int(os.getenv("EMBEDDING_DIM", "512")))
