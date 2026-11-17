"""Unit tests for memory service and conversation turn capping."""

import pytest

from app.schemas.booking import BookingState, BookingStatus
from tests.conftest import InMemoryMemoryService


@pytest.mark.asyncio
async def test_memory_turn_capping() -> None:
    # 2 turns = max 4 messages (user + assistant)
    memory = InMemoryMemoryService(max_turns=2)
    session_id = "test-session-cap"

    # Add 3 turns (6 messages)
    for i in range(1, 4):
        await memory.add_turn(session_id, "user", f"User question {i}")
        await memory.add_turn(session_id, "assistant", f"Assistant answer {i}")

    history = await memory.get_history(session_id)
    assert len(history) == 4  # Capped at 2 turns * 2 = 4 messages
    assert history[0].content == "User question 2"
    assert history[-1].content == "Assistant answer 3"


@pytest.mark.asyncio
async def test_memory_booking_state_lifecycle() -> None:
    memory = InMemoryMemoryService()
    session_id = "test-session-state"

    # Initially empty
    state = await memory.get_booking_state(session_id)
    assert state is None

    # Save in-progress state
    new_state = BookingState(
        status=BookingStatus.IN_PROGRESS,
        collected_fields={"name": "Bob"},
        missing_fields=["email", "date", "time"],
    )
    await memory.save_booking_state(session_id, new_state)

    fetched = await memory.get_booking_state(session_id)
    assert fetched is not None
    assert fetched.status == BookingStatus.IN_PROGRESS
    assert fetched.collected_fields == {"name": "Bob"}

    # Clear state
    await memory.clear_booking_state(session_id)
    assert await memory.get_booking_state(session_id) is None


@pytest.mark.asyncio
async def test_memory_ping() -> None:
    memory = InMemoryMemoryService()
    assert await memory.ping() is True
    memory.is_connected = False
    assert await memory.ping() is False
