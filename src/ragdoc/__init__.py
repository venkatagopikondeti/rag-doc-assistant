"""Retrieval-augmented generation over your own documents."""
from .chunking import Chunk, chunk_document, split_text
from .pipeline import RagPipeline
from .store import VectorStore

__version__ = "0.1.0"
__all__ = ["Chunk", "RagPipeline", "VectorStore", "chunk_document", "split_text"]
