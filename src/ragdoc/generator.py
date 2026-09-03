"""Answer generation. Azure OpenAI in production, extractive fallback offline."""
from __future__ import annotations

import os
import textwrap
from abc import ABC, abstractmethod

from .chunking import Chunk

PROMPT = textwrap.dedent(
    """\
    You are a document assistant. Answer the question using ONLY the context
    below. If the context does not contain the answer, say you don't know.
    Cite sources as [source:index].

    Context:
    {context}

    Question: {question}
    Answer:"""
)


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.source}:{c.index}] {c.text}" for c, _ in hits)


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str: ...


class ExtractiveGenerator(Generator):
    """No-LLM fallback: returns the best-matching passages with citations.

    Keeps the API honest when no model credentials are configured, and makes
    retrieval quality measurable in isolation.
    """

    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        if not hits:
            return "I don't know - no relevant passages were retrieved."
        top = hits[: min(2, len(hits))]
        body = "\n\n".join(f"[{c.source}:{c.index}] {c.text.strip()}" for c, _ in top)
        return f"Most relevant passages for {question!r}:\n\n{body}"


class AzureOpenAIGenerator(Generator):  # pragma: no cover - needs credentials
    def __init__(self) -> None:
        from openai import AzureOpenAI

        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )

    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(context=build_context(hits), question=question),
                }
            ],
        )
        return response.choices[0].message.content or ""


def get_generator() -> Generator:
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        return AzureOpenAIGenerator()
    return ExtractiveGenerator()
