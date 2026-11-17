"""Mock LLM client implementation for testing and offline development."""

import re
from typing import Any

from app.services.llm.base import BookingExtractionResult, LLMClient


class MockLLMClient(LLMClient):
    """Rule-based mock LLM for testing RAG Q&A and multi-turn interview booking offline."""

    def __init__(self, default_answer: str = "This is a mock answer based on context.") -> None:
        self.default_answer = default_answer

    async def generate_answer(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        _ = system_prompt
        # If user asks something specific in tests, we can provide context-aware responses
        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = m.get("content", "")
                break

        if "insufficient" in user_message.lower():
            return (
                "I don't have enough information in the provided documents to answer that question."
            )

        return f"{self.default_answer} User asked: '{user_message}'"

    async def extract_booking_intent_and_slots(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        current_slots: dict[str, Any],
    ) -> BookingExtractionResult:
        _ = conversation_history
        lower_msg = message.lower()

        is_question = any(
            q in lower_msg
            for q in ["what", "how", "why", "where", "who", "tell me", "can you explain"]
        ) or lower_msg.endswith("?")
        has_booking_action = any(
            k in lower_msg for k in ["book", "interview", "schedule", "slot", "appointment"]
        )

        has_intent = has_booking_action and not is_question

        name: str | None = None
        email: str | None = None
        date: str | None = None
        time_slot: str | None = None

        # Extract email
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", message)
        if email_match:
            email = email_match.group(0)
            has_intent = True

        # Extract date (YYYY-MM-DD)
        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
        if date_match:
            date = date_match.group(1)
            has_intent = True

        # Extract time (HH:MM or HH:MM AM/PM)
        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", message)
        if time_match:
            time_slot = time_match.group(1)
            has_intent = True

        # Extract name (e.g. "my name is John Doe" or "I am Alice")
        name_match = re.search(
            r"(?i:name is|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", message
        )
        if name_match:
            name = name_match.group(1).strip()
            has_intent = True
        elif not name and not email_match and not date_match and not time_match:
            # If message is short and doesn't match other things, candidate name
            if current_slots and not current_slots.get("name") and not is_question:
                parts = message.strip().split()
                if 1 <= len(parts) <= 3 and all(p.isalpha() for p in parts):
                    name = message.strip()
                    has_intent = True

        return BookingExtractionResult(
            is_booking_intent=has_intent,
            name=name,
            email=email,
            date=date,
            time=time_slot,
        )
