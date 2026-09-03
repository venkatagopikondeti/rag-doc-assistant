"""Split documents into overlapping chunks suitable for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    metadata: dict = field(default_factory=dict)


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Greedy paragraph-aware splitter with character overlap.

    Paragraphs are packed together until `chunk_size` is reached; a paragraph
    longer than `chunk_size` on its own is hard-wrapped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        while len(para) > chunk_size:
            flush()
            chunks.append(para[:chunk_size].strip())
            para = para[chunk_size - overlap :]
        if len(buffer) + len(para) + 2 > chunk_size:
            flush()
            if chunks and overlap:
                buffer = chunks[-1][-overlap:] + "\n\n"
        buffer += para + "\n\n"

    flush()
    return chunks


def chunk_document(text: str, source: str, **kwargs) -> list[Chunk]:
    return [
        Chunk(text=piece, source=source, index=i)
        for i, piece in enumerate(split_text(text, **kwargs))
    ]
