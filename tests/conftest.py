"""Shared test fixtures, mocks, and dependency overrides for offline testing."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import (
    get_embedder,
    get_llm_client,
    get_memory_service,
    get_vector_store,
)
from app.db.base import Base
from app.db.session import get_db_session
from app.main import create_app
from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage
from app.services.chunking.semantic import cosine_similarity
from app.services.embeddings.base import Embedder
from app.services.embeddings.mock import MockEmbedder
from app.services.llm.base import LLMClient
from app.services.llm.mock import MockLLMClient
from app.services.memory import MemoryService
from app.services.vector_store import ScoredChunk, VectorStore


class InMemoryVectorStore(VectorStore):
    """In-memory mock vector store calculating real cosine similarities."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self.points: list[dict[str, Any]] = []
        self.is_connected = True

    async def init_collection(self) -> None:
        pass

    async def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunking_strategy: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        for i, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            self.points.append(
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": text,
                    "filename": filename,
                    "chunking_strategy": chunking_strategy,
                    "vector": vector,
                }
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for p in self.points:
            score = cosine_similarity(query_vector, p["vector"])
            if score_threshold is None or score >= score_threshold:
                scored.append((score, p))

        # Sort descending by cosine similarity score
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[ScoredChunk] = []
        for score, p in scored[:top_k]:
            results.append(
                ScoredChunk(
                    document_id=p["document_id"],
                    chunk_index=p["chunk_index"],
                    text=p["text"],
                    filename=p["filename"],
                    chunking_strategy=p["chunking_strategy"],
                    score=score,
                )
            )
        return results

    async def ping(self) -> bool:
        return self.is_connected


class InMemoryMemoryService(MemoryService):
    """In-memory mock Redis memory service for chat history and booking state."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.histories: dict[str, list[ChatMessage]] = {}
        self.booking_states: dict[str, BookingState] = {}
        self.is_connected = True

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self.histories:
            self.histories[session_id] = []
        msg = ChatMessage(role=role, content=content, timestamp=datetime.now(UTC))
        self.histories[session_id].append(msg)
        max_messages = self.max_turns * 2
        if len(self.histories[session_id]) > max_messages:
            self.histories[session_id] = self.histories[session_id][-max_messages:]

    async def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        items = self.histories.get(session_id, [])
        if limit is not None:
            return items[-limit:]
        return list(items)

    async def get_booking_state(self, session_id: str) -> BookingState | None:
        return self.booking_states.get(session_id)

    async def save_booking_state(self, session_id: str, state: BookingState) -> None:
        self.booking_states[session_id] = state

    async def clear_booking_state(self, session_id: str) -> None:
        self.booking_states.pop(session_id, None)

    async def ping(self) -> bool:
        return self.is_connected


@pytest_asyncio.fixture
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated in-memory SQLite async database session."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def mock_embedder() -> Embedder:
    return MockEmbedder(dimension=128, model_name="mock-embedder-128")


@pytest.fixture
def mock_llm_client() -> LLMClient:
    return MockLLMClient()


@pytest.fixture
def in_memory_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=128)


@pytest.fixture
def in_memory_memory_service() -> InMemoryMemoryService:
    return InMemoryMemoryService(max_turns=5)


@pytest_asyncio.fixture
async def async_client(
    test_db_session: AsyncSession,
    mock_embedder: Embedder,
    mock_llm_client: LLMClient,
    in_memory_vector_store: InMemoryVectorStore,
    in_memory_memory_service: InMemoryMemoryService,
) -> AsyncGenerator[AsyncClient, None]:
    """Configured AsyncClient for integration testing with mocked external dependencies."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_db_session] = lambda: test_db_session
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_vector_store] = lambda: in_memory_vector_store
    app.dependency_overrides[get_memory_service] = lambda: in_memory_memory_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
