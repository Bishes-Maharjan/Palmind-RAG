"""Pydantic schemas for Document ingestion and queries."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    """Supported document chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"


class DocumentMetadata(BaseModel):
    """Metadata representation for an ingested document."""

    id: str = Field(description="Unique document identifier (UUID)")
    filename: str = Field(description="Original uploaded file name")
    content_type: str = Field(description="MIME type of the uploaded file")
    size_bytes: int = Field(description="Size of the file in bytes")
    chunking_strategy: ChunkingStrategy = Field(
        description="Chunking strategy applied during ingestion"
    )
    chunk_count: int = Field(description="Number of chunks generated")
    embedding_model: str = Field(description="Embedding model used for vectorization")
    status: str = Field(description="Processing status (e.g. processed, failed)")
    created_at: datetime = Field(description="Timestamp of ingestion")

    model_config = {"from_attributes": True}


class DocumentUploadResponse(DocumentMetadata):
    """Response returned upon successful document ingestion."""

    pass


class DocumentListResponse(BaseModel):
    """Paginated or listed documents response."""

    total: int = Field(description="Total number of documents")
    documents: list[DocumentMetadata] = Field(description="List of document metadata items")
