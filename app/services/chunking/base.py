"""Chunker protocol definition."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Chunker(Protocol):
    """Protocol for text chunking strategies."""

    async def chunk(self, text: str) -> list[str]:
        """Split raw text into structured chunks."""
        ...
