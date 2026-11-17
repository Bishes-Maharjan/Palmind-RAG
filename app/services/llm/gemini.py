"""Google Gemini LLM provider implementation with structured output parsing."""

import json
from typing import Any

from google import genai
from google.genai import types

from app.services.llm.base import BookingExtractionResult, LLMClient


class GeminiLLMClient(LLMClient):
    """Async Gemini LLM client supporting conversational answers and structured slot extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def generate_answer(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate conversational reply using Gemini."""
        contents: list[types.Content] = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )

        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system_prompt,
        )

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return (response.text or "").strip()

    async def extract_booking_intent_and_slots(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        current_slots: dict[str, Any],
    ) -> BookingExtractionResult:
        """Analyze message in context to detect booking intent and extract slot values."""
        system_instruction = (
            "You are an intelligent booking assistant parser. Determine whether the user "
            "wishes to book an interview, continue scheduling, or is answering booking slot "
            "requests (e.g., providing their name, email, date, or time).\n"
            f"Already collected slots: {current_slots}\n"
            "Extract any slot values provided in the message. Always convert dates to "
            "YYYY-MM-DD and times to 24-hour HH:MM format if mentioned. If a slot was "
            "not mentioned, set it to null."
        )

        contents: list[types.Content] = []
        for turn in conversation_history[-4:]:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn["content"])],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=BookingExtractionResult,
        )

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = response.text or ""
        if not text.strip():
            return BookingExtractionResult(is_booking_intent=False)

        parsed_data: dict[str, Any] = json.loads(text)
        return BookingExtractionResult(**parsed_data)
