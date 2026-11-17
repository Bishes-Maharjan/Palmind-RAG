"""Unit and integration tests for interview booking slot filling and validation."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Booking
from app.schemas.booking import BookingSlots, BookingStatus
from app.services.booking import BookingService
from app.services.llm.mock import MockLLMClient
from tests.conftest import InMemoryMemoryService


def test_booking_slots_past_date_rejection() -> None:
    past_date = date.today() - timedelta(days=1)
    with pytest.raises(ValidationError) as exc:
        BookingSlots(name="Alice", email="alice@example.com", date=past_date)
    assert "cannot be in the past" in str(exc.value)


def test_booking_slots_valid() -> None:
    future_date = date.today() + timedelta(days=7)
    slots = BookingSlots(name="Alice", email="alice@example.com", date=future_date)
    assert slots.date == future_date
    assert slots.email == "alice@example.com"


@pytest.mark.asyncio
async def test_booking_multi_turn_flow(test_db_session: AsyncSession) -> None:
    memory = InMemoryMemoryService()
    llm = MockLLMClient()
    service = BookingService(llm_client=llm, memory=memory)
    session_id = "test-booking-session-1"

    # Turn 1: User indicates intent to book
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message="I'd like to book an interview please.",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert reply is not None
    assert info is not None
    assert info.status == BookingStatus.IN_PROGRESS
    assert "name" in info.missing_fields

    # Turn 2: User provides name and email
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message="My name is Alice Smith and my email is alice@example.com",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert info is not None
    assert info.status == BookingStatus.IN_PROGRESS
    assert info.collected_fields.get("name") == "Alice Smith"
    assert info.collected_fields.get("email") == "alice@example.com"
    assert "date" in info.missing_fields
    assert "time" in info.missing_fields

    # Turn 3: User provides invalid past date
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message="Let's do 2020-01-01 at 10:00",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert reply is not None
    assert "in the past" in reply.lower()
    assert info is not None
    assert info.status == BookingStatus.IN_PROGRESS

    # Turn 4: User provides valid future date and time
    future_date_str = (date.today() + timedelta(days=10)).isoformat()
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message=f"How about {future_date_str} at 14:00?",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert reply is not None
    assert info is not None
    assert info.status == BookingStatus.CONFIRMED
    assert info.booking_id is not None
    assert "scheduled" in reply.lower()

    # Verify DB persistence
    res = await test_db_session.execute(select(Booking).where(Booking.id == info.booking_id))
    persisted = res.scalar_one_or_none()
    assert persisted is not None
    assert persisted.name == "Alice Smith"
    assert persisted.email == "alice@example.com"
    assert persisted.time.strftime("%H:%M") == "14:00"


@pytest.mark.asyncio
async def test_booking_interruption_and_resumption(test_db_session: AsyncSession) -> None:
    memory = InMemoryMemoryService()
    llm = MockLLMClient()
    service = BookingService(llm_client=llm, memory=memory)
    session_id = "test-interrupt-session"

    # Turn 1: Provide Name and Email
    handled, _, info = await service.handle_turn(
        session_id=session_id,
        message="I want to book an interview, my name is Bob Ross, email is bob@ross.com",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert info is not None
    assert info.status == BookingStatus.IN_PROGRESS
    assert info.collected_fields["name"] == "Bob Ross"

    # Turn 2: Interruption question
    handled, _, info = await service.handle_turn(
        session_id=session_id,
        message="What technology stack do you use?",
        db_session=test_db_session,
        history=[],
    )
    # Not handled by booking flow -> falls through to RAG!
    assert handled is False

    # Verify slot state is STILL safely preserved in Redis!
    state = await memory.get_booking_state(session_id)
    assert state is not None
    assert state.collected_fields["name"] == "Bob Ross"
    assert state.collected_fields["email"] == "bob@ross.com"

    # Turn 3: Resume scheduling by providing date and time
    future_date = (date.today() + timedelta(days=3)).isoformat()
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message=f"Great, let's schedule for {future_date} at 11:00",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert info is not None
    assert info.status == BookingStatus.CONFIRMED
    assert info.booking_id is not None


@pytest.mark.asyncio
async def test_booking_cancellation(test_db_session: AsyncSession) -> None:
    memory = InMemoryMemoryService()
    llm = MockLLMClient()
    service = BookingService(llm_client=llm, memory=memory)
    session_id = "test-cancel-session"

    # Start booking
    await service.handle_turn(
        session_id=session_id,
        message="I want to schedule an interview",
        db_session=test_db_session,
        history=[],
    )

    # Cancel booking
    handled, reply, info = await service.handle_turn(
        session_id=session_id,
        message="cancel booking",
        db_session=test_db_session,
        history=[],
    )
    assert handled is True
    assert reply is not None
    assert "cancelled" in reply.lower()
    assert info is not None
    assert info.status == BookingStatus.CANCELLED

    # Booking state in memory should be cleared
    state = await memory.get_booking_state(session_id)
    assert state is None
