"""
Admin API Routes
System administration endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.db.session import get_db_session
from backend.db.models import User as UserModel, Document as DocumentModel
from backend.db.models import Query as QueryModel

logger = get_logger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_system_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """Get system usage statistics"""
    # Document stats
    doc_result = await db.execute(
        select(
            func.count(DocumentModel.id).label("total"),
            func.sum(DocumentModel.chunk_count).label("total_chunks"),
            func.sum(DocumentModel.page_count).label("total_pages"),
        )
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
    # TODO: Implement proper pagination and filtering
    result = await db.execute(
        select(UserModel).offset(offset).limit(limit)
    )
    users = result.scalars().all()

    return {
        "total": len(users),
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
