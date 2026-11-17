"""API tests for document ingestion and metadata management endpoints."""

import io

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_txt_fixed_chunking(async_client: AsyncClient) -> None:
    content = b"This is a sample document for RAG ingestion. It has some text content."
    files = {"file": ("sample.txt", io.BytesIO(content), "text/plain")}
    data = {"chunking_strategy": "fixed"}

    response = await async_client.post("/api/v1/documents", files=files, data=data)
    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "sample.txt"
    assert payload["chunking_strategy"] == "fixed"
    assert payload["chunk_count"] >= 1
    assert "id" in payload


@pytest.mark.asyncio
async def test_upload_txt_semantic_chunking(async_client: AsyncClient) -> None:
    content = (
        b"Machine learning algorithms extract patterns. Data science is fascinating.\n\n"
        b"Baking bread requires flour and yeast. The dough must rise properly."
    )
    files = {"file": ("data.txt", io.BytesIO(content), "text/plain")}
    data = {"chunking_strategy": "semantic"}

    response = await async_client.post("/api/v1/documents", files=files, data=data)
    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "data.txt"
    assert payload["chunking_strategy"] == "semantic"
    assert payload["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_upload_empty_document_error(async_client: AsyncClient) -> None:
    files = {"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")}
    data = {"chunking_strategy": "fixed"}

    response = await async_client.post("/api/v1/documents", files=files, data=data)
    assert response.status_code == 422
    error_data = response.json()
    assert error_data["error"]["code"] == "DOCUMENT_EXTRACTION_ERROR"


@pytest.mark.asyncio
async def test_upload_invalid_file_extension(async_client: AsyncClient) -> None:
    files = {"file": ("test.sh", io.BytesIO(b"echo 1"), "application/x-sh")}
    data = {"chunking_strategy": "fixed"}

    response = await async_client.post("/api/v1/documents", files=files, data=data)
    assert response.status_code == 422
    error_data = response.json()
    assert error_data["error"]["code"] == "FILE_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_list_and_get_documents(async_client: AsyncClient) -> None:
    # 1. Ingest a document
    content = b"Content to test document listing and fetching."
    files = {"file": ("list_test.txt", io.BytesIO(content), "text/plain")}
    upload_res = await async_client.post(
        "/api/v1/documents", files=files, data={"chunking_strategy": "fixed"}
    )
    doc_id = upload_res.json()["id"]

    # 2. List documents
    list_res = await async_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(d["id"] == doc_id for d in list_data["documents"])

    # 3. Get document by ID
    get_res = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["id"] == doc_id
    assert doc_data["filename"] == "list_test.txt"

    # 4. Non-existent document
    not_found = await async_client.get("/api/v1/documents/non-existent-id")
    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
