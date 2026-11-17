"""OpenAI LLM provider implementation with structured output parsing."""

from typing import Any

import openai

from app.services.llm.base import BookingExtractionResult, LLMClient


class OpenAILLMClient(LLMClient):
    """Async OpenAI LLM client supporting conversational answers and structured slot extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def generate_answer(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,  # type: ignore[arg-type]
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    async def extract_booking_intent_and_slots(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        current_slots: dict[str, Any],
    ) -> BookingExtractionResult:
        system_instruction = (
            "You are an intelligent booking assistant parser. Determine whether the user "
            "wishes to book an interview, continue scheduling, or is answering booking slot "
            "requests (e.g., providing their name, email, date, or time).\n"
            f"Already collected slots: {current_slots}\n"
            "Extract any slot values provided in the message. Always convert dates to YYYY-MM-DD "
            "and times to 24-hour HH:MM format if mentioned. If a slot was not mentioned, "
            "set it to null."
        )

        formatted_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        # Include recent history turns for context
        for turn in conversation_history[-4:]:
            formatted_messages.append({"role": turn["role"], "content": turn["content"]})

        formatted_messages.append({"role": "user", "content": message})

        # Structured output parsing via OpenAI SDK
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=formatted_messages,  # type: ignore[arg-type]
            response_format=BookingExtractionResult,
            temperature=0.0,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            return BookingExtractionResult(is_booking_intent=False)
        return parsed
