"""LLM client protocol and extraction structures."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BookingExtractionResult(BaseModel):
    """Structured extraction of interview booking intent and slot candidates."""

    is_booking_intent: bool = Field(
        description=(
            "True if user wants to book, reschedule, continue booking an interview, "
            "or is providing contact/date/time details."
        )
    )
    name: str | None = Field(default=None, description="Candidate full name if mentioned")
    email: str | None = Field(default=None, description="Candidate email address if mentioned")
    date: str | None = Field(
        default=None,
        description="Desired date in YYYY-MM-DD format if mentioned (e.g. 2026-12-01)",
    )
    time: str | None = Field(
        default=None,
        description="Desired time in HH:MM (24-hour) format if mentioned (e.g. 14:30)",
    )


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM interactions."""

    async def generate_answer(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate conversational reply for RAG or general conversation."""
        ...

    async def extract_booking_intent_and_slots(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        current_slots: dict[str, Any],
    ) -> BookingExtractionResult:
        """Analyze message in context to detect booking intent and extract slot values."""
        ...
