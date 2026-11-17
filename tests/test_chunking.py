"""Unit tests for fixed and semantic chunking strategies."""

import pytest

from app.core.config import Settings
from app.schemas.document import ChunkingStrategy
from app.services.chunking.factory import ChunkerFactory
from app.services.chunking.fixed import FixedChunker
from app.services.chunking.semantic import SemanticChunker, cosine_similarity
from app.services.embeddings.base import Embedder
from app.services.embeddings.mock import MockEmbedder


def test_cosine_similarity() -> None:
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0

    v_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(v1, v_zero) == 0.0


@pytest.mark.asyncio
async def test_fixed_chunker_basic() -> None:
    chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
    text = "The quick brown fox jumps over the lazy dog repeatedly and silently."
    chunks = await chunker.chunk(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 50


@pytest.mark.asyncio
async def test_fixed_chunker_short_text() -> None:
    chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
    text = "Short text."
    chunks = await chunker.chunk(text)
    assert chunks == ["Short text."]


def test_fixed_chunker_invalid_params() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=0)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, chunk_overlap=150)


class MockControllableEmbedder(Embedder):
    """Embedder whose vectors can be controlled to trigger semantic boundaries."""

    def __init__(self) -> None:
        pass

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "mock-controllable"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            # Topic A: [1, 0, 0], Topic B: [0, 1, 0]
            if "database" in text.lower() or "sql" in text.lower():
                embeddings.append([1.0, 0.0, 0.0])
            else:
                embeddings.append([0.0, 1.0, 0.0])
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_semantic_chunker_boundary_split() -> None:
    embedder = MockControllableEmbedder()
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.7,
        min_chunk_size=10,
        max_chunk_size=500,
    )

    text = (
        "PostgreSQL is a powerful relational database system. "
        "SQL queries allow structured data access. "
        "Cooking pasta requires boiling water and salt. "
        "Tomatoes and garlic make a classic sauce."
    )

    chunks = await chunker.chunk(text)
    assert len(chunks) == 2
    assert "PostgreSQL" in chunks[0] and "SQL" in chunks[0]
    assert "Cooking pasta" in chunks[1] and "sauce" in chunks[1]


@pytest.mark.asyncio
async def test_semantic_chunker_min_chunk_size_guard() -> None:
    embedder = MockControllableEmbedder()
    # High min_chunk_size prevents split even when similarity drops
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.7,
        min_chunk_size=300,
        max_chunk_size=1000,
    )

    text = "PostgreSQL is a database. Cooking pasta is fun."
    chunks = await chunker.chunk(text)
    # Since total length is < 300, it should not split
    assert len(chunks) == 1


def test_chunker_factory() -> None:
    settings = Settings()
    mock_embed = MockEmbedder()

    fixed = ChunkerFactory.create(ChunkingStrategy.FIXED, settings, mock_embed)
    assert isinstance(fixed, FixedChunker)

    semantic = ChunkerFactory.create(ChunkingStrategy.SEMANTIC, settings, mock_embed)
    assert isinstance(semantic, SemanticChunker)

    with pytest.raises(ValueError):
        ChunkerFactory.create("unknown_strategy", settings, mock_embed)
