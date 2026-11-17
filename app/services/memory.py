"""Redis-backed conversational memory and booking slot state management."""

import json
import logging
from typing import Protocol, runtime_checkable

import redis.asyncio as aioredis

from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage

logger = logging.getLogger("app.services.memory")


@runtime_checkable
class MemoryService(Protocol):
    """Protocol for session-scoped memory persistence."""

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a conversational turn and enforce maximum turn limits and TTL."""
        ...

    async def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Fetch chat history for a session in chronological order."""
        ...

    async def get_booking_state(self, session_id: str) -> BookingState | None:
        """Fetch in-progress interview booking state."""
        ...

    async def save_booking_state(self, session_id: str, state: BookingState) -> None:
        """Persist or update interview booking state."""
        ...

    async def clear_booking_state(self, session_id: str) -> None:
        """Remove booking state once completed or cancelled."""
        ...

    async def ping(self) -> bool:
        """Verify Redis connectivity."""
        ...


class RedisMemoryService(MemoryService):
    """Redis implementation for session-keyed chat memory and slot filling state."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 86400,
        max_turns: int = 10,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.client = aioredis.from_url(redis_url, decode_responses=True)

    def _history_key(self, session_id: str) -> str:
        return f"chat:{session_id}:history"

    def _booking_key(self, session_id: str) -> str:
        return f"booking:{session_id}:state"

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        key = self._history_key(session_id)
        msg = ChatMessage(role=role, content=content)
        payload = msg.model_dump_json()

        # Push to Redis list
        await self.client.rpush(key, payload)
        # Cap stored turns (each turn has user + assistant, so max_turns * 2 messages)
        max_messages = self.max_turns * 2
        await self.client.ltrim(key, -max_messages, -1)
        # Refresh TTL
        await self.client.expire(key, self.ttl_seconds)

    async def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        key = self._history_key(session_id)
        start_idx = -limit if limit is not None else 0
        raw_items = await self.client.lrange(key, start_idx, -1)

        messages: list[ChatMessage] = []
        for item in raw_items:
            try:
                data = json.loads(item)
                messages.append(ChatMessage.model_validate(data))
            except Exception as e:
                logger.warning("Failed to deserialize chat message: %s", e)
        return messages

    async def get_booking_state(self, session_id: str) -> BookingState | None:
        key = self._booking_key(session_id)
        raw_data = await self.client.get(key)
        if not raw_data:
            return None
        try:
            return BookingState.model_validate_json(raw_data)
        except Exception as e:
            logger.warning("Failed to deserialize booking state: %s", e)
            return None

    async def save_booking_state(self, session_id: str, state: BookingState) -> None:
        key = self._booking_key(session_id)
        payload = state.model_dump_json()
        await self.client.set(key, payload, ex=self.ttl_seconds)

    async def clear_booking_state(self, session_id: str) -> None:
        key = self._booking_key(session_id)
        await self.client.delete(key)

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        """Close connection pool."""
        await self.client.aclose()
