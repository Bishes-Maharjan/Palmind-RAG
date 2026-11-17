# Sample cURL Transcript

This document provides a realistic, end-to-end command-line transcript illustrating interactions with all API endpoints, including multi-turn conversational booking and RAG knowledge retrieval.

---

## 1. System Health Check

```bash
curl -X GET "http://localhost:8000/health" -H "Accept: application/json"
```

### Response (200 OK)
```json
{
  "status": "ok",
  "postgres": true,
  "redis": true,
  "qdrant": true
}
```

---

## 2. Ingest Document (Fixed Chunking)

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@docs/sample.txt;type=text/plain" \
  -F "chunking_strategy=fixed"
```

### Response (201 Created)
```json
{
  "id": "7b88939c-c081-430b-93f5-e41da7d1b329",
  "filename": "sample.txt",
  "content_type": "text/plain",
  "size_bytes": 1693,
  "chunking_strategy": "fixed",
  "chunk_count": 4,
  "embedding_model": "text-embedding-3-small",
  "status": "processed",
  "created_at": "2026-11-17T04:30:00.000000Z"
}
```

---

## 3. Ingest Document (Semantic Chunking)

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@docs/sample.txt;type=text/plain" \
  -F "chunking_strategy=semantic"
```

### Response (201 Created)
```json
{
  "id": "e93bc145-21d7-4ee4-b6d8-c923d8c11aa2",
  "filename": "sample.txt",
  "content_type": "text/plain",
  "size_bytes": 1693,
  "chunking_strategy": "semantic",
  "chunk_count": 3,
  "embedding_model": "text-embedding-3-small",
  "status": "processed",
  "created_at": "2026-11-17T04:31:00.000000Z"
}
```

---

## 4. List Ingested Documents

```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

### Response (200 OK)
```json
{
  "total": 2,
  "documents": [
    {
      "id": "e93bc145-21d7-4ee4-b6d8-c923d8c11aa2",
      "filename": "sample.txt",
      "content_type": "text/plain",
      "size_bytes": 1693,
      "chunking_strategy": "semantic",
      "chunk_count": 3,
      "embedding_model": "text-embedding-3-small",
      "status": "processed",
      "created_at": "2026-11-17T04:31:00.000000Z"
    },
    {
      "id": "7b88939c-c081-430b-93f5-e41da7d1b329",
      "filename": "sample.txt",
      "content_type": "text/plain",
      "size_bytes": 1693,
      "chunking_strategy": "fixed",
      "chunk_count": 4,
      "embedding_model": "text-embedding-3-small",
      "status": "processed",
      "created_at": "2026-11-17T04:30:00.000000Z"
    }
  ]
}
```

---

## 5. Conversational RAG Query

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What backend technologies does PalmMind use?"
  }'
```

### Response (200 OK)
```json
{
  "session_id": "9bf2e3da-8e1d-4074-b52b-23e59dcbfcb6",
  "answer": "Based on the reference documents, PalmMind Technologies builds backend services primarily with Python 3.11+ using FastAPI, SQLAlchemy 2.0 with asyncpg, and Alembic for relational migrations. They also use Qdrant for vector search and Redis for caching and session state persistence.",
  "sources": [
    {
      "document_id": "e93bc145-21d7-4ee4-b6d8-c923d8c11aa2",
      "filename": "sample.txt",
      "chunk_index": 1,
      "score": 0.8841,
      "snippet": "Our backend services are built primarily with Python 3.11+ using FastAPI, SQLAlchemy 2.0 with asyncpg, and Alembic for relational migrations. For vector search, we run Qdrant clusters..."
    }
  ],
  "booking": null
}
```

---

## 6. Multi-Turn Interview Booking (with RAG Interruption)

### Turn 1: Candidate triggers booking intent
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "candidate-session-42",
    "message": "Hi! I would like to schedule my technical deep dive interview."
  }'
```

#### Response:
```json
{
  "session_id": "candidate-session-42",
  "answer": "Please provide your full name, your email address, your preferred date (YYYY-MM-DD), your preferred time (e.g. 14:00) to complete your interview booking.",
  "sources": [],
  "booking": {
    "status": "in_progress",
    "collected_fields": {},
    "missing_fields": ["name", "email", "date", "time"],
    "booking_id": null
  }
}
```

---

### Turn 2: Candidate provides Name and Email
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "candidate-session-42",
    "message": "My name is Samantha Vance and my email is samantha.vance@example.com"
  }'
```

#### Response:
```json
{
  "session_id": "candidate-session-42",
  "answer": "Recorded so far: Name: Samantha Vance, Email: samantha.vance@example.com.\n\nPlease provide your preferred date (YYYY-MM-DD), your preferred time (e.g. 14:00) to complete your interview booking.",
  "sources": [],
  "booking": {
    "status": "in_progress",
    "collected_fields": {
      "name": "Samantha Vance",
      "email": "samantha.vance@example.com"
    },
    "missing_fields": ["date", "time"],
    "booking_id": null
  }
}
```

---

### Turn 3: Interruption — Candidate asks a RAG question
The user interrupts slot collection to ask about the company process. The system answers from context while preserving Redis slot state.

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "candidate-session-42",
    "message": "Before picking a date, how long is the technical deep dive round?"
  }'
```

#### Response:
```json
{
  "session_id": "candidate-session-42",
  "answer": "According to the handbook, the Architecture & Technical Deep Dive round is 60 minutes long and focuses on discussing design choices, trade-offs, chunking strategies, and performance characteristics.",
  "sources": [
    {
      "document_id": "e93bc145-21d7-4ee4-b6d8-c923d8c11aa2",
      "filename": "sample.txt",
      "chunk_index": 2,
      "score": 0.8124,
      "snippet": "3. Architecture & Technical Deep Dive (60 minutes): Discussion of your design choices, trade-offs, chunking strategies, and performance characteristics..."
    }
  ],
  "booking": {
    "status": "in_progress",
    "collected_fields": {
      "name": "Samantha Vance",
      "email": "samantha.vance@example.com"
    },
    "missing_fields": ["date", "time"],
    "booking_id": null
  }
}
```

---

### Turn 4: Resumption — Candidate provides Date and Time
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "candidate-session-42",
    "message": "Got it! Let's do 2026-12-15 at 14:30 please."
  }'
```

#### Response:
```json
{
  "session_id": "candidate-session-42",
  "answer": "Your interview has been successfully scheduled!\n\nConfirmation ID: 4f1a92e1-87ab-49ce-b5a9-4676527ba8de\n- Candidate: Samantha Vance\n- Email: samantha.vance@example.com\n- Date: 2026-12-15\n- Time: 14:30\n\nWe look forward to speaking with you!",
  "sources": [],
  "booking": {
    "status": "confirmed",
    "collected_fields": {
      "name": "Samantha Vance",
      "email": "samantha.vance@example.com",
      "date": "2026-12-15",
      "time": "14:30"
    },
    "missing_fields": [],
    "booking_id": "4f1a92e1-87ab-49ce-b5a9-4676527ba8de"
  }
}
```

---

## 7. Get Conversation History

```bash
curl -X GET "http://localhost:8000/api/v1/chat/candidate-session-42/history"
```

### Response (200 OK)
```json
{
  "session_id": "candidate-session-42",
  "messages": [
    {
      "role": "user",
      "content": "Hi! I would like to schedule my technical deep dive interview.",
      "timestamp": "2026-11-17T04:35:10.000000Z"
    },
    {
      "role": "assistant",
      "content": "Please provide your full name, your email address, your preferred date (YYYY-MM-DD), your preferred time (e.g. 14:00) to complete your interview booking.",
      "timestamp": "2026-11-17T04:35:11.000000Z"
    },
    {
      "role": "user",
      "content": "My name is Samantha Vance and my email is samantha.vance@example.com",
      "timestamp": "2026-11-17T04:35:25.000000Z"
    },
    {
      "role": "assistant",
      "content": "Recorded so far: Name: Samantha Vance, Email: samantha.vance@example.com.\n\nPlease provide your preferred date (YYYY-MM-DD), your preferred time (e.g. 14:00) to complete your interview booking.",
      "timestamp": "2026-11-17T04:35:26.000000Z"
    },
    {
      "role": "user",
      "content": "Before picking a date, how long is the technical deep dive round?",
      "timestamp": "2026-11-17T04:35:40.000000Z"
    },
    {
      "role": "assistant",
      "content": "According to the handbook, the Architecture & Technical Deep Dive round is 60 minutes long...",
      "timestamp": "2026-11-17T04:35:42.000000Z"
    },
    {
      "role": "user",
      "content": "Got it! Let's do 2026-12-15 at 14:30 please.",
      "timestamp": "2026-11-17T04:36:00.000000Z"
    },
    {
      "role": "assistant",
      "content": "Your interview has been successfully scheduled!\n\nConfirmation ID: 4f1a92e1-87ab-49ce-b5a9-4676527ba8de...",
      "timestamp": "2026-11-17T04:36:02.000000Z"
    }
  ]
}
```
