"""Semantic chunking strategy based on embedding cosine similarity transitions."""

import math
import re

from app.services.chunking.base import Chunker
from app.services.embeddings.base import Embedder


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two dense float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticChunker(Chunker):
    """Splits text on meaning shifts by embedding adjacent sentences and detecting drops.

    Guarded by configurable min and max chunk character lengths.
    """

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split raw text into clean, non-empty sentences."""
        # Split on sentence terminals followed by space, or double newlines
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        sentences: list[str] = []
        for s in raw_sentences:
            cleaned = s.strip()
            if cleaned:
                sentences.append(cleaned)
        return sentences

    async def chunk(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        sentences = self._split_into_sentences(cleaned)
        if not sentences:
            return [cleaned]

        if len(sentences) == 1:
            return [sentences[0]]

        # Generate embeddings for all individual sentences in one batch
        embeddings = await self.embedder.embed_texts(sentences)

        # Calculate cosine similarities between adjacent sentences
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]
        current_len: int = len(sentences[0])

        for i in range(len(similarities)):
            next_sentence = sentences[i + 1]
            sim = similarities[i]
            proposed_len = current_len + 1 + len(next_sentence)

            # Decision criteria:
            # 1. Force split if proposed length exceeds max_chunk_size
            force_split = proposed_len > self.max_chunk_size
            # 2. Semantic shift split: similarity below threshold and chunk >= min_chunk_size
            semantic_split = sim < self.similarity_threshold and current_len >= self.min_chunk_size

            if force_split or semantic_split:
                chunks.append(" ".join(current_sentences))
                current_sentences = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_sentences.append(next_sentence)
                current_len = proposed_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks
