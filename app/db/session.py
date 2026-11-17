"""Database session and async engine management."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def create_engine(database_url: str | None = None) -> AsyncEngine:
    """Create a configured SQLAlchemy async engine."""
    url = database_url or get_settings().database_url
    if "sqlite" in url:
        return create_async_engine(url, echo=False, future=True)
    return create_async_engine(
        url,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )


# Global engine and sessionmaker instances initialized per process
engine: AsyncEngine = create_engine()
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session per request."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping_database(session: AsyncSession) -> bool:
    """Check database health by running a simple SELECT 1 query."""
    try:
        result = await session.execute(text("SELECT 1"))
        return result.scalar() == 1
    except Exception:
        return False
