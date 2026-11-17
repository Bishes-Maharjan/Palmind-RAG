"""Unit tests for text extraction service."""

import io

import pytest
from pypdf import PdfWriter

from app.core.exceptions import DocumentExtractionError, FileValidationError
from app.services.extraction import extract_text, validate_file


def create_sample_pdf(_text: str = "") -> io.BytesIO:
    """Generate in-memory PDF with specified text content."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf doesn't easily render text directly without ReportLab,
    # but we can add text annotations or test blank page handling
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf


def test_validate_file_valid() -> None:
    filename, mime = validate_file("test.pdf", "application/pdf", 1024, 10000)
    assert filename == "test.pdf"
    assert mime == "application/pdf"

    filename, mime = validate_file("notes.txt", None, 500, 10000)
    assert filename == "notes.txt"
    assert mime == "text/plain"


def test_validate_file_invalid_extension() -> None:
    with pytest.raises(FileValidationError) as exc:
        validate_file("malicious.exe", "application/octet-stream", 1024, 10000)
    assert "Unsupported file extension" in str(exc.value)


def test_validate_file_empty_size() -> None:
    with pytest.raises(FileValidationError) as exc:
        validate_file("empty.txt", "text/plain", 0, 10000)
    assert "empty (0 bytes)" in str(exc.value)


def test_validate_file_exceeds_max_size() -> None:
    with pytest.raises(FileValidationError) as exc:
        validate_file("large.txt", "text/plain", 20000, 10000)
    assert "exceeds the maximum allowed limit" in str(exc.value)


def test_extract_text_txt_utf8() -> None:
    content = b"Hello, world! This is a test file for RAG ingestion."
    stream = io.BytesIO(content)
    text = extract_text(stream, "sample.txt")
    assert "Hello, world!" in text


def test_extract_text_txt_empty() -> None:
    stream = io.BytesIO(b"   \n\n  \t  ")
    with pytest.raises(DocumentExtractionError) as exc:
        extract_text(stream, "empty.txt")
    assert "contains no extractable text" in str(exc.value)


def test_extract_text_pdf_blank_or_scanned() -> None:
    # A blank page PDF simulates a scanned PDF with no extractable text layer
    pdf_stream = create_sample_pdf("")
    with pytest.raises(DocumentExtractionError) as exc:
        extract_text(pdf_stream, "scanned.pdf")
    assert "contains no extractable text" in str(exc.value)
