"""
Database Session Management
PostgreSQL connection and session handling
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.db.base import Base

logger = get_logger(__name__)

# Engine
engine = None
async_session_maker = None


async def init_db() -> None:
    """Initialize database engine and create tables"""
    global engine, async_session_maker

    logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")

    engine = create_async_engine(
        settings.POSTGRES_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Import all SQLAlchemy models to ensure they're registered with Base
    from backend.db.models import User, Document, Query
    from backend.models.chunk import Chunk
    from backend.models.permission import Permission

    # Create tables (use Alembic for production migrations)
    if settings.ENVIRONMENT == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")


async def close_db() -> None:
    """Close database connections"""
    global engine

    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session (dependency injection)"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_schema() -> bool:
    """Check if database schema exists"""
    try:
        from sqlalchemy import text

        async with async_session_maker() as session:
            result = await session.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1")
            )
            return result.scalar() is not None
    except Exception as e:
        logger.warning(f"Schema check failed: {e}")
        return False
