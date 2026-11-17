"""Document ingestion and metadata management endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status
from sqlalchemy import select

from app.api.deps import DbSessionDep, EmbedderDep, SettingsDep, VectorStoreDep
from app.core.exceptions import DocumentNotFoundError
from app.db.models import Document
from app.schemas.document import (
    ChunkingStrategy,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.chunking.factory import ChunkerFactory
from app.services.extraction import extract_text, validate_file

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new document",
    description=(
        "Upload a .pdf or .txt file, apply the chosen chunking strategy, "
        "vectorize chunks, and store metadata."
    ),
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or TXT document file")],
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(description="Chunking strategy: 'fixed' or 'semantic'"),
    ],
    db: DbSessionDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
    vector_store: VectorStoreDep,
) -> DocumentUploadResponse:
    # 1. Read file bytes and validate size & mime
    content = await file.read()
    size_bytes = len(content)

    sanitized_filename, resolved_mime = validate_file(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=size_bytes,
        max_size_bytes=settings.max_upload_size_bytes,
    )

    # 2. Extract plain text (raises DocumentExtractionError / 422 on empty or scanned PDF)
    import io

    extracted_text = extract_text(io.BytesIO(content), sanitized_filename)

    # 3. Create chunker via registry/factory and chunk document
    chunker = ChunkerFactory.create(
        strategy=chunking_strategy,
        settings=settings,
        embedder=embedder,
    )
    chunks = await chunker.chunk(extracted_text)
    if not chunks:
        chunks = [extracted_text]

    # 4. Create document database record
    doc_record = Document(
        filename=sanitized_filename,
        content_type=resolved_mime,
        size_bytes=size_bytes,
        chunking_strategy=chunking_strategy.value,
        chunk_count=len(chunks),
        embedding_model=embedder.model_name,
        status="processed",
    )
    db.add(doc_record)
    await db.flush()  # Populates doc_record.id

    # 5. Generate embeddings in batch and upsert to Qdrant
    vectors = await embedder.embed_texts(chunks)
    await vector_store.upsert_chunks(
        document_id=doc_record.id,
        filename=sanitized_filename,
        chunking_strategy=chunking_strategy.value,
        chunks=chunks,
        vectors=vectors,
    )

    return DocumentUploadResponse(
        id=doc_record.id,
        filename=doc_record.filename,
        content_type=doc_record.content_type,
        size_bytes=doc_record.size_bytes,
        chunking_strategy=chunking_strategy,
        chunk_count=doc_record.chunk_count,
        embedding_model=doc_record.embedding_model,
        status=doc_record.status,
        created_at=doc_record.created_at,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents",
    description="Retrieve metadata for all ingested documents.",
)
async def list_documents(db: DbSessionDep) -> DocumentListResponse:
    query = select(Document).order_by(Document.created_at.desc())
    result = await db.execute(query)
    records = result.scalars().all()

    documents = [DocumentMetadata.model_validate(rec) for rec in records]
    return DocumentListResponse(total=len(documents), documents=documents)


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document metadata",
    description="Retrieve metadata for a specific document by UUID.",
)
async def get_document(document_id: str, db: DbSessionDep) -> DocumentMetadata:
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    record = result.scalar_one_or_none()

    if not record:
        raise DocumentNotFoundError(document_id=document_id)

    return DocumentMetadata.model_validate(record)
