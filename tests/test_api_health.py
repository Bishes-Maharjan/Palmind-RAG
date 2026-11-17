"""API tests for the /health endpoint."""

import pytest
from httpx import AsyncClient

from tests.conftest import InMemoryMemoryService, InMemoryVectorStore


@pytest.mark.asyncio
async def test_health_all_ok(async_client: AsyncClient) -> None:
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["postgres"] is True
    assert data["redis"] is True
    assert data["qdrant"] is True


@pytest.mark.asyncio
async def test_health_degraded(
    async_client: AsyncClient,
    in_memory_memory_service: InMemoryMemoryService,
) -> None:
    # Simulate redis failure
    in_memory_memory_service.is_connected = False
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["redis"] is False
    assert data["postgres"] is True
    assert data["qdrant"] is True
    # Restore connection
    in_memory_memory_service.is_connected = True


@pytest.mark.asyncio
async def test_health_unhealthy(
    async_client: AsyncClient,
    in_memory_memory_service: InMemoryMemoryService,
    in_memory_vector_store: InMemoryVectorStore,
) -> None:
    # Simulate both redis and vector store failing
    in_memory_memory_service.is_connected = False
    in_memory_vector_store.is_connected = False
    response = await async_client.get("/health")
    # postgres is still ok so status is degraded
    data = response.json()
    assert data["status"] == "degraded"
    assert data["redis"] is False
    assert data["qdrant"] is False
    # Restore
    in_memory_memory_service.is_connected = True
    in_memory_vector_store.is_connected = True
