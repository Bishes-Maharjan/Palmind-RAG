"""OpenAI embedding provider implementation."""

import openai

from app.services.embeddings.base import Embedder


class OpenAIEmbedder(Embedder):
    """Generates embeddings using OpenAI API or an OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches to respect rate limits and request payload sizes
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # OpenAI preserves item ordering by index
            batch_sorted = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in batch_sorted])

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        if not results:
            return [0.0] * self._dimension
        return results[0]
