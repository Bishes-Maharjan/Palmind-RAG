"""Common Pydantic schemas for health and error responses."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured error payload details."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional context or validation details"
    )


class ErrorResponse(BaseModel):
    """Standardized top-level API error envelope."""

    error: ErrorDetail


class HealthStatus(BaseModel):
    """Overall system and dependency health status."""

    status: str = Field(description="Overall health status: ok | degraded | unhealthy")
    postgres: bool = Field(description="PostgreSQL connectivity status")
    redis: bool = Field(description="Redis connectivity status")
    qdrant: bool = Field(description="Qdrant connectivity status")
