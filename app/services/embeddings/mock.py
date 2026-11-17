"""Deterministic mock embedder for offline testing."""

import hashlib
import math

from app.services.embeddings.base import Embedder


class MockEmbedder(Embedder):
    """Generates deterministic pseudo-embeddings for tests without network calls."""

    def __init__(self, dimension: int = 1536, model_name: str = "mock-embedding") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_vector(self, text: str) -> list[float]:
        # Hash text to generate deterministic floating point numbers
        raw_vals: list[float] = []
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(self._dimension):
            byte_idx = i % len(seed)
            raw_vals.append(float(seed[byte_idx]) / 255.0 - 0.5)

        # Normalize to unit length for valid cosine distance
        norm = math.sqrt(sum(x * x for x in raw_vals))
        if norm == 0:
            return [1.0] + [0.0] * (self._dimension - 1)
        return [x / norm for x in raw_vals]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)
