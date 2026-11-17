"""Health check endpoint evaluating external service connectivity."""

import asyncio

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.deps import DbSessionDep, MemoryDep, VectorStoreDep
from app.db.session import ping_database
from app.schemas.common import HealthStatus

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Service Health Check",
    description="Inspect connectivity to PostgreSQL, Redis, and Qdrant.",
)
async def check_health(
    db: DbSessionDep,
    memory: MemoryDep,
    vector_store: VectorStoreDep,
) -> JSONResponse:
    # Run pings concurrently for low latency
    pg_task = asyncio.create_task(ping_database(db))
    redis_task = asyncio.create_task(memory.ping())
    qdrant_task = asyncio.create_task(vector_store.ping())

    pg_ok, redis_ok, qdrant_ok = await asyncio.gather(pg_task, redis_task, qdrant_task)

    all_ok = pg_ok and redis_ok and qdrant_ok
    any_ok = pg_ok or redis_ok or qdrant_ok

    if all_ok:
        overall_status = "ok"
        http_code = status.HTTP_200_OK
    elif any_ok:
        overall_status = "degraded"
        http_code = status.HTTP_200_OK
    else:
        overall_status = "unhealthy"
        http_code = status.HTTP_503_SERVICE_UNAVAILABLE

    payload = HealthStatus(
        status=overall_status,
        postgres=pg_ok,
        redis=redis_ok,
        qdrant=qdrant_ok,
    )
    return JSONResponse(status_code=http_code, content=payload.model_dump())
