"""Embedder Protocol interface definition."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for generating dense vector embeddings from text."""

    @property
    def dimension(self) -> int:
        """Vector dimensionality produced by this embedder."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text passages."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        ...
