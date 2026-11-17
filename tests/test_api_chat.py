"""API tests for the conversational RAG and multi-turn interview booking endpoint."""

import io
from datetime import date, timedelta

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_rag_query(async_client: AsyncClient) -> None:
    # Ingest document first
    content = b"PalmMind RAG uses Qdrant for high-performance vector retrieval."
    files = {"file": ("rag_doc.txt", io.BytesIO(content), "text/plain")}
    await async_client.post("/api/v1/documents", files=files, data={"chunking_strategy": "fixed"})

    # Chat query
    payload = {"message": "What vector database does PalmMind use?"}
    response = await async_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["answer"] is not None
    assert isinstance(data["sources"], list)
    assert data["booking"] is None


@pytest.mark.asyncio
async def test_chat_multi_turn_booking_and_history(async_client: AsyncClient) -> None:
    # Turn 1: Initiate booking
    t1_res = await async_client.post(
        "/api/v1/chat",
        json={"message": "I would like to book an interview slot."},
    )
    assert t1_res.status_code == 200
    t1_data = t1_res.json()
    session_id = t1_data["session_id"]
    assert t1_data["booking"] is not None
    assert t1_data["booking"]["status"] == "in_progress"

    # Turn 2: Provide Name and Email
    t2_res = await async_client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "message": "My name is Carlos Santana and my email is carlos@example.com",
        },
    )
    assert t2_res.status_code == 200
    t2_data = t2_res.json()
    assert t2_data["booking"]["status"] == "in_progress"
    assert t2_data["booking"]["collected_fields"]["name"] == "Carlos Santana"
    assert t2_data["booking"]["collected_fields"]["email"] == "carlos@example.com"

    # Turn 3: Complete with future Date and Time
    future_date = (date.today() + timedelta(days=5)).isoformat()
    t3_res = await async_client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "message": f"I prefer {future_date} at 11:30 please.",
        },
    )
    assert t3_res.status_code == 200
    t3_data = t3_res.json()
    assert t3_data["booking"]["status"] == "confirmed"
    assert t3_data["booking"]["booking_id"] is not None

    # Retrieve history
    hist_res = await async_client.get(f"/api/v1/chat/{session_id}/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["session_id"] == session_id
    # 3 turns = 6 messages (3 user, 3 assistant)
    assert len(hist_data["messages"]) == 6
    roles = [m["role"] for m in hist_data["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
