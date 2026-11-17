"""Text extraction service supporting PDF and plain text documents."""

from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from app.core.exceptions import DocumentExtractionError, FileValidationError

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "application/octet-stream",  # Sometimes sent for plain text or generic files
}


def validate_file(
    filename: str | None,
    content_type: str | None,
    size_bytes: int,
    max_size_bytes: int,
) -> tuple[str, str]:
    """Validate file extension, mime type, and maximum upload size.

    Returns the sanitized filename and validated content type.
    """
    if not filename:
        raise FileValidationError("Uploaded file must have a filename.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file extension '{ext}'. Only {sorted(ALLOWED_EXTENSIONS)} are allowed."
        )

    if size_bytes <= 0:
        raise FileValidationError("Uploaded file is empty (0 bytes).")

    if size_bytes > max_size_bytes:
        raise FileValidationError(
            f"File size {size_bytes} exceeds the maximum allowed limit of {max_size_bytes} bytes."
        )

    resolved_mime = content_type or ("application/pdf" if ext == ".pdf" else "text/plain")
    return filename, resolved_mime


def extract_text(file_stream: BinaryIO, filename: str) -> str:
    """Extract plain text from an uploaded PDF or TXT file.

    Raises:
        DocumentExtractionError: If text is unreadable or empty (e.g. scanned PDF).
    """
    ext = Path(filename).suffix.lower()
    extracted_text = ""

    if ext == ".txt":
        try:
            content = file_stream.read()
            # Try utf-8 first, fallback to latin-1
            try:
                extracted_text = content.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = content.decode("latin-1")
        except Exception as e:
            raise DocumentExtractionError(f"Failed to read text file '{filename}': {e}") from e

    elif ext == ".pdf":
        try:
            reader = PdfReader(file_stream)
            if len(reader.pages) == 0:
                raise DocumentExtractionError(f"PDF file '{filename}' has no pages.")

            pages_text: list[str] = []
            for _i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())

            extracted_text = "\n\n".join(pages_text)
        except DocumentExtractionError:
            raise
        except Exception as e:
            raise DocumentExtractionError(
                f"Corrupt or unreadable PDF file '{filename}': {e}"
            ) from e
    else:
        raise DocumentExtractionError(f"Unsupported file format '{ext}' for extraction.")

    # Clean text
    cleaned_text = extracted_text.strip()
    if not cleaned_text:
        raise DocumentExtractionError(
            f"File '{filename}' contains no extractable text. "
            "Scanned PDFs without OCR or blank documents are not supported."
        )

    return cleaned_text
