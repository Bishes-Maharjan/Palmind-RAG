"""Factory and registry for chunker instantiation."""

from app.core.config import Settings
from app.schemas.document import ChunkingStrategy
from app.services.chunking.base import Chunker
from app.services.chunking.fixed import FixedChunker
from app.services.chunking.semantic import SemanticChunker
from app.services.embeddings.base import Embedder


class ChunkerFactory:
    """Factory resolving chunking strategy enum values to Chunker instances."""

    @staticmethod
    def create(
        strategy: ChunkingStrategy | str,
        settings: Settings,
        embedder: Embedder,
    ) -> Chunker:
        strategy_str = strategy.value if isinstance(strategy, ChunkingStrategy) else str(strategy)

        if strategy_str == ChunkingStrategy.FIXED.value:
            return FixedChunker(
                chunk_size=settings.fixed_chunk_size,
                chunk_overlap=settings.fixed_chunk_overlap,
            )
        elif strategy_str == ChunkingStrategy.SEMANTIC.value:
            return SemanticChunker(
                embedder=embedder,
                similarity_threshold=settings.semantic_similarity_threshold,
                min_chunk_size=settings.semantic_min_chunk_size,
                max_chunk_size=settings.semantic_max_chunk_size,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: '{strategy}'")
