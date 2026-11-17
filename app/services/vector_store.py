"""Qdrant vector store service with async client, idempotent upserts, and payload management."""

import logging
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger("app.services.vector_store")


class ScoredChunk(BaseModel):
    """Result chunk returned by vector similarity search."""

    document_id: str
    chunk_index: int
    text: str
    filename: str
    chunking_strategy: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Protocol defining vector storage and search operations."""

    async def init_collection(self) -> None:
        """Initialize collection if missing with matching vector parameters."""
        ...

    async def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunking_strategy: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """Idempotently persist chunk embeddings and metadata."""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        """Perform cosine similarity vector search."""
        ...

    async def ping(self) -> bool:
        """Verify vector DB connection health."""
        ...


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore using the official AsyncQdrantClient."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_dimension: int,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        if url == ":memory:":
            self.client = AsyncQdrantClient(location=":memory:")
        else:
            self.client = AsyncQdrantClient(url=url, api_key=api_key)

    async def init_collection(self) -> None:
        """Ensure the collection exists with the appropriate vector dimension and cosine metric."""
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' (dim=%d, distance=Cosine)",
                    self.collection_name,
                    self.vector_dimension,
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.info("Qdrant collection '%s' already exists.", self.collection_name)
        except Exception as e:
            logger.error("Failed to initialize Qdrant collection: %s", e)
            raise

    async def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunking_strategy: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        if not chunks:
            return

        points: list[PointStruct] = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            # Idempotent deterministic UUID for point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{i}"))
            payload: dict[str, Any] = {
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk_text,
                "filename": filename,
                "chunking_strategy": chunking_strategy,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(
            "Upserted %d chunk vectors for document_id '%s'",
            len(points),
            document_id,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results: list[ScoredChunk] = []
            for pt in response.points:
                payload = pt.payload or {}
                results.append(
                    ScoredChunk(
                        document_id=str(payload.get("document_id", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=str(payload.get("text", "")),
                        filename=str(payload.get("filename", "")),
                        chunking_strategy=str(payload.get("chunking_strategy", "")),
                        score=float(pt.score or 0.0),
                    )
                )
            return results
        except Exception as e:
            logger.error("Error executing vector search in Qdrant: %s", e)
            raise

    async def ping(self) -> bool:
        try:
            # Query collections to check live connectivity
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close client connections."""
        await self.client.close()
