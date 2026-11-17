"""Custom application exceptions with uniform error schema support."""

from typing import Any


class AppException(Exception):
    """Base application exception for standardized error handling."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class FileValidationError(AppException):
    """Raised when an uploaded file fails validation (size, MIME type, extension)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="FILE_VALIDATION_ERROR",
            details=details,
        )


class DocumentExtractionError(AppException):
    """Raised when text cannot be extracted (e.g. scanned PDF without OCR or empty file)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="DOCUMENT_EXTRACTION_ERROR",
            details=details,
        )


class DocumentNotFoundError(AppException):
    """Raised when a document is not found in the database."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with id '{document_id}' was not found.",
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
            details={"document_id": document_id},
        )


class BookingValidationError(AppException):
    """Raised when booking slot details fail validation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="BOOKING_VALIDATION_ERROR",
            details=details,
        )


class ServiceUnavailableError(AppException):
    """Raised when an external dependency (Postgres, Redis, Qdrant) is unreachable."""

    def __init__(self, service_name: str, message: str) -> None:
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable: {message}",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service_name},
        )
