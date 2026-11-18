"""Google Gemini embedding provider implementation."""

import asyncio

from google import genai
from google.genai import types

from app.services.embeddings.base import Embedder


class GeminiEmbedder(Embedder):
    """Generates embeddings using the Google Gemini Embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = genai.Client(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dimension,
            ),
        )
        if response.embeddings and response.embeddings[0].values is not None:
            return response.embeddings[0].values
        return [0.0] * self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text passages concurrently."""
        if not texts:
            return []

        # Concurrently generate embeddings for all chunks guaranteeing 1-to-1 ordering
        tasks = [self._embed_single(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        return await self._embed_single(text)
