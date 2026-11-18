"""Main FastAPI application entry point with lifespan management and global exception handlers."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import get_current_settings, get_vector_store
from app.api.routers import chat, documents, health
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.db.base import Base
from app.db.session import engine

# Initialize structured logging
settings: Settings = get_current_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info("Starting up palmmind-rag application...")

    # Ensure database tables exist (for dev / local test setups without manual alembic execution)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema validated successfully.")
    except Exception as e:
        logger.warning("Could not automatically initialize DB tables at startup: %s", e)

    # Initialize Qdrant collection
    try:
        vector_store = get_vector_store(settings)
        await vector_store.init_collection()
        logger.info("Qdrant collection validated.")
    except Exception as e:
        logger.warning("Could not initialize Qdrant collection at startup: %s", e)

    yield

    logger.info("Shutting down palmmind-rag application...")
    await engine.dispose()
    logger.info("Disposed database connection engine.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="PalmMind RAG & Conversational Booking API",
        description=(
            "Production-grade backend featuring document ingestion with multiple chunking "
            "strategies (fixed & semantic), Qdrant vector retrieval, PostgreSQL metadata, "
            "Redis multi-turn chat memory, and conversational interview booking slot filling."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "Application exception on %s %s: [%s] %s",
            request.method,
            request.url.path,
            exc.error_code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "Validation error on %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed.",
                    "details": {"validation_errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled server error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "details": {},
                }
            },
        )

    # Register Routers
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    return app


app = create_app()
