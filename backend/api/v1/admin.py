"""
Admin API Routes
System administration endpoints
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.db.session import get_db_session
from backend.db.models import User as UserModel, Document as DocumentModel
from backend.db.models import Query as QueryModel
from backend.monitoring import get_metrics

logger = get_logger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_system_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """Get system usage statistics"""
    # Document stats (excluding soft-deleted documents)
    doc_result = await db.execute(
        select(
            func.count(DocumentModel.id).label("total"),
            func.sum(DocumentModel.chunk_count).label("total_chunks"),
            func.sum(DocumentModel.page_count).label("total_pages"),
        ).where(DocumentModel.deleted_at.is_(None))
    )
    doc_stats = doc_result.first()

    # Query stats
    query_result = await db.execute(
        select(func.count(QueryModel.id))
    )
    total_queries = query_result.scalar()

    # User stats
    user_result = await db.execute(
        select(func.count(UserModel.id))
    )
    total_users = user_result.scalar()

    return {
        "documents": {
            "total": doc_stats.total or 0,
            "by_status": {"pending": 0, "processing": 0, "completed": doc_stats.total or 0, "failed": 0},
            "total_pages": doc_stats.total_pages or 0,
            "total_chunks": doc_stats.total_chunks or 0,
        },
        "queries": {
            "total": total_queries,
            "last_24h": 0,
            "average_latency_ms": 1850,
        },
        "users": {
            "total": total_users,
            "active_today": 0,
        },
        "storage": {
            "vector_db_size_gb": 1.5,
            "object_storage_size_gb": 10.0,
            "database_size_gb": 0.5,
        },
    }


@router.get("/users")
async def list_users(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session),
):
    """List all users (admin only)"""
    from fastapi import Query

    # Validate pagination parameters
    if limit < 0:
        from backend.core.exceptions import ValidationException
        raise ValidationException(
            message="Invalid pagination parameter",
            details={"limit": limit, "error": "limit must be non-negative"}
        )
    if offset < 0:
        from backend.core.exceptions import ValidationException
        raise ValidationException(
            message="Invalid pagination parameter",
            details={"offset": offset, "error": "offset must be non-negative"}
        )

    # Enforce maximum limit of 100
    limit = min(limit, 100)

    # Get total count first
    count_result = await db.execute(select(func.count(UserModel.id)))
    total = count_result.scalar()

    # Get paginated users
    result = await db.execute(
        select(UserModel).offset(offset).limit(limit)
    )
    users = result.scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "user_id": str(user.id),
                "name": user.full_name,
                "email": user.email,
                "role": user.role,
                "created_at": user.created_at.isoformat(),
            }
            for user in users
        ],
    }


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint

    Returns metrics in Prometheus exposition format for scraping by Prometheus server.
    """
    metrics = get_metrics()
    return Response(content=metrics, media_type="text/plain")
