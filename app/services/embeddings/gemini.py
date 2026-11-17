"""Google Gemini embedding provider implementation."""

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

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text passages."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._dimension,
                ),
            )
            if response.embeddings:
                for emb in response.embeddings:
                    if emb.values is not None:
                        all_embeddings.append(emb.values)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        results = await self.embed_texts([text])
        if not results:
            return [0.0] * self._dimension
        return results[0]
