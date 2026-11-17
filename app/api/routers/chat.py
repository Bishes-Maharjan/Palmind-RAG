"""Conversational RAG and multi-turn interview booking endpoint."""

import uuid

from fastapi import APIRouter, status

from app.api.deps import (
    BookingServiceDep,
    DbSessionDep,
    MemoryDep,
    RAGServiceDep,
)
from app.schemas.booking import BookingInfoResponse
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-turn Conversational RAG and Interview Booking",
    description=(
        "Process user queries with defensive RAG retrieval or multi-turn "
        "interview booking slot filling."
    ),
)
async def chat_endpoint(
    payload: ChatRequest,
    db: DbSessionDep,
    memory: MemoryDep,
    booking_service: BookingServiceDep,
    rag_service: RAGServiceDep,
) -> ChatResponse:
    # 1. Assign or preserve session ID
    session_id = payload.session_id or str(uuid.uuid4())
    user_message = payload.message.strip()

    # 2. Retrieve conversation history from Redis
    history = await memory.get_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]

    # 3. Check interview booking intent and slot progress
    is_booking, booking_answer, booking_info = await booking_service.handle_turn(
        session_id=session_id,
        message=user_message,
        db_session=db,
        history=history_dicts,
    )

    if is_booking and booking_answer is not None:
        # Save user and assistant turns to Redis
        await memory.add_turn(session_id, "user", user_message)
        await memory.add_turn(session_id, "assistant", booking_answer)

        return ChatResponse(
            session_id=session_id,
            answer=booking_answer,
            sources=[],
            booking=booking_info,
        )

    # 4. Standard RAG Q&A
    rag_answer, sources = await rag_service.answer_query(
        query=user_message,
        conversation_history=history,
    )

    # If an in-progress booking exists, attach its info so client knows booking state is retained
    existing_booking = await memory.get_booking_state(session_id)
    current_booking_info: BookingInfoResponse | None = None
    if existing_booking:
        current_booking_info = BookingInfoResponse(
            status=existing_booking.status,
            collected_fields=existing_booking.collected_fields,
            missing_fields=existing_booking.missing_fields,
            booking_id=existing_booking.booking_id,
        )

    # Save user and assistant turns to Redis
    await memory.add_turn(session_id, "user", user_message)
    await memory.add_turn(session_id, "assistant", rag_answer)

    return ChatResponse(
        session_id=session_id,
        answer=rag_answer,
        sources=sources,
        booking=current_booking_info,
    )


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history",
    description="Retrieve all conversational turns preserved in Redis for the given session.",
)
async def get_chat_history(session_id: str, memory: MemoryDep) -> ChatHistoryResponse:
    messages = await memory.get_history(session_id)
    return ChatHistoryResponse(session_id=session_id, messages=messages)
