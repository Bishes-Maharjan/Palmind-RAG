"""Dependency injection providers for FastAPI endpoints."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.db.session import get_db_session
from app.services.booking import BookingService
from app.services.embeddings.base import Embedder
from app.services.embeddings.gemini import GeminiEmbedder
from app.services.llm.base import LLMClient
from app.services.llm.gemini import GeminiLLMClient
from app.services.memory import MemoryService, RedisMemoryService
from app.services.rag import RAGService
from app.services.vector_store import QdrantVectorStore, VectorStore

# Module-level singletons for client connection pooling
_vector_store: VectorStore | None = None
_memory_service: MemoryService | None = None
_embedder: Embedder | None = None
_llm_client: LLMClient | None = None


def get_current_settings() -> Settings:
    """Retrieve global settings."""
    return get_settings()


def get_embedder(settings: Annotated[Settings, Depends(get_current_settings)]) -> Embedder:
    """Provide configured Gemini Embedder instance."""
    global _embedder
    if _embedder is None:
        if not settings.gemini_api_key:
            raise ServiceUnavailableError(
                "Gemini", "GEMINI_API_KEY environment variable is not configured."
            )
        _embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
            dimension=settings.gemini_embedding_dimensions,
        )
    return _embedder


def get_llm_client(settings: Annotated[Settings, Depends(get_current_settings)]) -> LLMClient:
    """Provide configured Gemini LLMClient instance."""
    global _llm_client
    if _llm_client is None:
        if not settings.gemini_api_key:
            raise ServiceUnavailableError(
                "Gemini", "GEMINI_API_KEY environment variable is not configured."
            )
        _llm_client = GeminiLLMClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_chat_model,
        )
    return _llm_client


def get_vector_store(settings: Annotated[Settings, Depends(get_current_settings)]) -> VectorStore:
    """Provide configured VectorStore (Qdrant) instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            vector_dimension=settings.gemini_embedding_dimensions,
            api_key=settings.qdrant_api_key,
        )
    return _vector_store


def get_memory_service(
    settings: Annotated[Settings, Depends(get_current_settings)],
) -> MemoryService:
    """Provide Redis-backed MemoryService instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = RedisMemoryService(
            redis_url=settings.redis_url,
            ttl_seconds=settings.redis_chat_ttl_seconds,
            max_turns=settings.redis_max_turns,
        )
    return _memory_service


def get_booking_service(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    memory: Annotated[MemoryService, Depends(get_memory_service)],
) -> BookingService:
    """Provide BookingService instance."""
    return BookingService(llm_client=llm_client, memory=memory)


def get_rag_service(
    settings: Annotated[Settings, Depends(get_current_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> RAGService:
    """Provide RAGService instance."""
    return RAGService(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )


# Type aliases for clean router signatures
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
SettingsDep = Annotated[Settings, Depends(get_current_settings)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
MemoryDep = Annotated[MemoryService, Depends(get_memory_service)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
BookingServiceDep = Annotated[BookingService, Depends(get_booking_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
