"""Pydantic schemas for interview booking validation and state management."""

from datetime import date as dt_date
from datetime import datetime
from datetime import time as dt_time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class BookingStatus(StrEnum):
    """Lifecycle status of a conversational booking session."""

    IN_PROGRESS = "in_progress"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class BookingSlots(BaseModel):
    """Slot values collected for interview scheduling."""

    name: str | None = Field(default=None, description="Candidate full name")
    email: EmailStr | None = Field(default=None, description="Candidate contact email")
    date: dt_date | None = Field(default=None, description="Interview scheduled date")
    time: dt_time | None = Field(default=None, description="Interview scheduled time")

    @field_validator("date")
    @classmethod
    def validate_date_not_in_past(cls, v: dt_date | None) -> dt_date | None:
        """Reject dates in the past."""
        if v is not None and v < dt_date.today():
            raise ValueError(f"Booking date {v} cannot be in the past.")
        return v


class BookingState(BaseModel):
    """Redis-persisted state tracking in-progress interview booking."""

    status: BookingStatus = Field(
        default=BookingStatus.IN_PROGRESS, description="Current booking status"
    )
    collected_fields: dict[str, Any] = Field(
        default_factory=dict, description="Dictionary of filled valid slots"
    )
    missing_fields: list[str] = Field(
        default_factory=lambda: ["name", "email", "date", "time"],
        description="List of fields still required",
    )
    booking_id: str | None = Field(default=None, description="Persisted booking ID once confirmed")


class BookingInfoResponse(BaseModel):
    """Booking segment returned in the chat response."""

    status: BookingStatus = Field(description="Booking status")
    collected_fields: dict[str, Any] = Field(
        default_factory=dict, description="Fields collected so far"
    )
    missing_fields: list[str] = Field(default_factory=list, description="Fields still missing")
    booking_id: str | None = Field(
        default=None, description="Created booking database ID if completed"
    )


class BookingRecord(BaseModel):
    """Complete booking record persisted in the database."""

    id: str
    name: str
    email: str
    date: dt_date
    time: dt_time
    session_id: str
    created_at: datetime

    model_config = {"from_attributes": True}
