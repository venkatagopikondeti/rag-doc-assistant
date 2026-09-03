"""Ingestion + query orchestration."""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .generator import Generator, get_generator
from .store import VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # lazy: only needed for PDFs

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


class RagPipeline:
    def __init__(
        self, store: VectorStore | None = None, generator: Generator | None = None
    ) -> None:
        self.store = store or VectorStore()
        self.generator = generator or get_generator()

    def ingest_path(self, path: str | Path, **chunk_kwargs) -> int:
        path = Path(path)
        files = (
            [path]
            if path.is_file()
            else [
                p
                for p in sorted(path.rglob("*"))
                if p.suffix.lower() in SUPPORTED_SUFFIXES
            ]
        )
        added = 0
        for file in files:
            text = read_document(file)
            if not text.strip():
                continue
            chunks = chunk_document(text, source=file.name, **chunk_kwargs)
            self.store.add(chunks)
            added += len(chunks)
        return added

    def answer(self, question: str, k: int = 4) -> dict:
        hits = self.store.search(question, k=k)
        return {
            "question": question,
            "answer": self.generator.generate(question, hits),
            "sources": [
                {"source": c.source, "index": c.index, "score": round(s, 4), "text": c.text[:300]}
                for c, s in hits
            ],
        }
