"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "production", "testing"] = Field(
        default="development", alias="APP_ENV"
    )
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # PostgreSQL (Async SQLAlchemy via asyncpg)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb",
        alias="DATABASE_URL",
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_chat_ttl_seconds: int = Field(default=86400, alias="REDIS_CHAT_TTL_SECONDS")
    redis_max_turns: int = Field(default=10, alias="REDIS_MAX_TURNS")

    # Qdrant Vector DB
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="document_chunks", alias="QDRANT_COLLECTION_NAME")

    # LLM & Embeddings (Google Gemini)
    gemini_api_key: str = Field(default="mock-key-for-local", alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_CHAT_MODEL")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )
    gemini_embedding_dimensions: int = Field(default=768, alias="GEMINI_EMBEDDING_DIMENSIONS")

    # Ingestion & RAG
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES"
    )  # 10 MB
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_score_threshold: float = Field(default=0.3, alias="RAG_SCORE_THRESHOLD")

    # Chunking
    fixed_chunk_size: int = Field(default=500, alias="FIXED_CHUNK_SIZE")
    fixed_chunk_overlap: int = Field(default=100, alias="FIXED_CHUNK_OVERLAP")
    semantic_similarity_threshold: float = Field(
        default=0.75, alias="SEMANTIC_SIMILARITY_THRESHOLD"
    )
    semantic_min_chunk_size: int = Field(default=100, alias="SEMANTIC_MIN_CHUNK_SIZE")
    semantic_max_chunk_size: int = Field(default=1000, alias="SEMANTIC_MAX_CHUNK_SIZE")


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
