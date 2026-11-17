"""Pydantic schemas for multi-turn chat and RAG Q&A."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.booking import BookingInfoResponse


class ChatRequest(BaseModel):
    """Input payload for conversational chat endpoint."""

    session_id: str | None = Field(
        default=None,
        description="Optional session identifier. If omitted, a new UUID session is generated.",
    )
    message: str = Field(..., min_length=1, description="User's input query or response")


class SourceReference(BaseModel):
    """Retrieved document chunk citation used for RAG grounding."""

    document_id: str = Field(description="Source document UUID")
    filename: str = Field(description="Name of the source document")
    chunk_index: int = Field(description="Zero-based index of the chunk in the document")
    score: float = Field(description="Vector cosine similarity score")
    snippet: str = Field(description="Excerpt of text from the chunk")


class ChatResponse(BaseModel):
    """Structured response from the conversational RAG endpoint."""

    session_id: str = Field(description="Unique chat session identifier")
    answer: str = Field(description="Assistant response text")
    sources: list[SourceReference] = Field(
        default_factory=list, description="Grounding source citations"
    )
    booking: BookingInfoResponse | None = Field(
        default=None,
        description="Active or completed booking status and slot progress, if triggered.",
    )


class ChatMessage(BaseModel):
    """A single conversational turn message stored in history."""

    role: str = Field(description="Message author role: user | assistant")
    content: str = Field(description="Content of the message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp message was recorded",
    )


class ChatHistoryResponse(BaseModel):
    """Full message history for a given session."""

    session_id: str = Field(description="Chat session identifier")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="Ordered conversational turns"
    )
