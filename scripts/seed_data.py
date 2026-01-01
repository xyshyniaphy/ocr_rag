#!/usr/bin/env python3
"""
Seed Data Script
Create initial admin user and test data
"""

import asyncio
import sys
import uuid

from sqlalchemy import select

from backend.db.session import async_session_maker
from backend.models.user import User
from backend.core.security import get_password_hash
from backend.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def seed_admin_user():
    """Create admin user"""
    async with async_session_maker() as session:
        # Check if admin exists
        result = await session.execute(
            select(User).where(User.email == "admin@example.com")
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info("Admin user already exists")
            return

        # Create admin user
        admin = User(
            id=uuid.uuid4(),
            email="admin@example.com",
            password_hash=get_password_hash("admin123"),
            full_name="System Administrator",
            role="admin",
            is_active=True,
            is_verified=True,
        )

        session.add(admin)
        await session.commit()
        await session.refresh(admin)

        logger.info(f"Created admin user: {admin.email}")
        logger.info("  Password: admin123")
        logger.info("  PLEASE CHANGE THIS PASSWORD IN PRODUCTION!")


async def seed_test_data():
    """Create test data"""
    async with async_session_maker() as session:
        # Check if test users exist
        result = await session.execute(
            select(User).where(User.email == "user@example.com")
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info("Test data already exists")
            return

        # Create test user
        test_user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            password_hash=get_password_hash("user123"),
            full_name="Test User",
            role="user",
            is_active=True,
            is_verified=True,
        )

        session.add(test_user)
        await session.commit()
        await session.refresh(test_user)

        logger.info(f"Created test user: {test_user.email}")
        logger.info("  Password: user123")


async def main():
    """Main seeding function"""
    logger.info("Seeding database...")

    try:
        await seed_admin_user()
        await seed_test_data()
        logger.info("Database seeded successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to seed database: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
