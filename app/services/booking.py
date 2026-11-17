"""Interview booking service handling slot extraction, validation, and persistence."""

import logging
from datetime import date as dt_date
from datetime import datetime
from typing import Any

from pydantic import EmailStr, TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Booking
from app.schemas.booking import BookingInfoResponse, BookingState, BookingStatus
from app.services.llm.base import LLMClient
from app.services.memory import MemoryService

logger = logging.getLogger("app.services.booking")
email_adapter = TypeAdapter(EmailStr)


class BookingService:
    """Manages conversational interview scheduling, slot filling, and database persistence."""

    def __init__(self, llm_client: LLMClient, memory: MemoryService) -> None:
        self.llm_client = llm_client
        self.memory = memory

    async def handle_turn(
        self,
        session_id: str,
        message: str,
        db_session: AsyncSession,
        history: list[dict[str, str]],
    ) -> tuple[bool, str | None, BookingInfoResponse | None]:
        """Process a conversational turn for interview booking.

        Returns:
            tuple of (is_handled, response_text, booking_info)
            - is_handled=True indicates the booking flow consumed this turn.
            - is_handled=False indicates the message should fall through to RAG.
        """
        existing_state = await self.memory.get_booking_state(session_id)
        current_slots = existing_state.collected_fields if existing_state else {}

        # Check for explicit cancellation
        lower_msg = message.lower().strip()
        if existing_state and existing_state.status == BookingStatus.IN_PROGRESS:
            if lower_msg in {"cancel", "cancel booking", "stop", "nevermind", "abort"}:
                await self.memory.clear_booking_state(session_id)
                cancelled_info = BookingInfoResponse(
                    status=BookingStatus.CANCELLED,
                    collected_fields={},
                    missing_fields=[],
                    booking_id=None,
                )
                return (
                    True,
                    "Interview booking has been cancelled. How else can I help you?",
                    cancelled_info,
                )

        # Use LLM to extract slots and detect booking intent
        extraction = await self.llm_client.extract_booking_intent_and_slots(
            message=message,
            conversation_history=history,
            current_slots=current_slots,
        )

        # If user was not in booking flow and did not express booking intent, return False
        if not existing_state and not extraction.is_booking_intent:
            return False, None, None

        # If user had an existing booking flow, but this message has NO booking slots
        has_new_slots = bool(
            extraction.name or extraction.email or extraction.date or extraction.time
        )
        if (
            existing_state
            and existing_state.status == BookingStatus.IN_PROGRESS
            and not extraction.is_booking_intent
            and not has_new_slots
        ):
            # User interrupted booking flow with a general query. Let RAG handle it!
            # Keep existing booking state preserved in Redis.
            return False, None, None

        # Merge candidate slots into collected slots
        collected: dict[str, Any] = dict(current_slots)
        validation_errors: list[str] = []

        if extraction.name:
            name_val = extraction.name.strip()
            if len(name_val) >= 2:
                collected["name"] = name_val
            else:
                validation_errors.append("Please provide a valid full name.")

        if extraction.email:
            try:
                valid_email = email_adapter.validate_python(extraction.email.strip())
                collected["email"] = str(valid_email)
            except Exception:
                validation_errors.append(
                    f"'{extraction.email}' is not a valid email address. "
                    "Please provide a valid email."
                )

        if extraction.date:
            try:
                parsed_date = datetime.strptime(extraction.date.strip(), "%Y-%m-%d").date()
                if parsed_date < dt_date.today():
                    validation_errors.append(
                        f"Date '{extraction.date}' is in the past. "
                        "Please choose today or a future date."
                    )
                else:
                    collected["date"] = parsed_date.isoformat()
            except ValueError:
                validation_errors.append(
                    f"Could not parse date '{extraction.date}'. "
                    "Please use YYYY-MM-DD format (e.g. 2026-12-01)."
                )

        if extraction.time:
            try:
                # Support HH:MM or HH:MM:SS
                time_str = extraction.time.strip()
                if len(time_str) == 5:
                    parsed_time = datetime.strptime(time_str, "%H:%M").time()
                else:
                    parsed_time = datetime.strptime(time_str, "%H:%M:%S").time()
                collected["time"] = parsed_time.strftime("%H:%M")
            except ValueError:
                validation_errors.append(
                    f"Could not parse time '{extraction.time}'. "
                    "Please use 24-hour HH:MM format (e.g. 14:00)."
                )

        required_fields = ["name", "email", "date", "time"]
        missing_fields = [f for f in required_fields if f not in collected]

        # Case 1: All required slots are present and validated!
        if not missing_fields and not validation_errors:
            # Persist to PostgreSQL
            booking_record = Booking(
                name=str(collected["name"]),
                email=str(collected["email"]),
                date=datetime.strptime(str(collected["date"]), "%Y-%m-%d").date(),
                time=datetime.strptime(str(collected["time"]), "%H:%M").time(),
                session_id=session_id,
            )
            db_session.add(booking_record)
            await db_session.flush()

            # Persist confirmed state in Redis
            confirmed_state = BookingState(
                status=BookingStatus.CONFIRMED,
                collected_fields=collected,
                missing_fields=[],
                booking_id=booking_record.id,
            )
            await self.memory.save_booking_state(session_id, confirmed_state)

            answer = (
                f"Your interview has been successfully scheduled!\n\n"
                f"Confirmation ID: {booking_record.id}\n"
                f"- Candidate: {booking_record.name}\n"
                f"- Email: {booking_record.email}\n"
                f"- Date: {booking_record.date.isoformat()}\n"
                f"- Time: {booking_record.time.strftime('%H:%M')}\n\n"
                f"We look forward to speaking with you!"
            )

            booking_info = BookingInfoResponse(
                status=BookingStatus.CONFIRMED,
                collected_fields=collected,
                missing_fields=[],
                booking_id=booking_record.id,
            )
            return True, answer, booking_info

        # Case 2: Slots are still missing or have validation errors
        in_progress_state = BookingState(
            status=BookingStatus.IN_PROGRESS,
            collected_fields=collected,
            missing_fields=missing_fields,
            booking_id=None,
        )
        await self.memory.save_booking_state(session_id, in_progress_state)

        # Generate polite guidance for missing or invalid slots
        reply_parts: list[str] = []
        if validation_errors:
            reply_parts.append(" ".join(validation_errors))

        # Summarize already collected details
        collected_summary: list[str] = []
        for k, v in collected.items():
            collected_summary.append(f"{k.capitalize()}: {v}")

        if collected_summary:
            reply_parts.append(f"Recorded so far: {', '.join(collected_summary)}.")

        field_prompts = {
            "name": "your full name",
            "email": "your email address",
            "date": "your preferred date (YYYY-MM-DD)",
            "time": "your preferred time (e.g. 14:00)",
        }
        remaining_prompts = [field_prompts[f] for f in missing_fields if f in field_prompts]
        if remaining_prompts:
            reply_parts.append(
                f"Please provide {', '.join(remaining_prompts)} to complete your interview booking."
            )

        answer = "\n\n".join(reply_parts)
        booking_info = BookingInfoResponse(
            status=BookingStatus.IN_PROGRESS,
            collected_fields=collected,
            missing_fields=missing_fields,
            booking_id=None,
        )
        return True, answer, booking_info
