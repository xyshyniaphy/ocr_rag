#!/usr/bin/env python3
"""
Seed Admin User Script
Creates the default admin user in the database
"""

import asyncio
import sys
import uuid

# Add the backend to the path
sys.path.insert(0, "/app")

from sqlalchemy import select

from backend.core.security import get_password_hash
from backend.db.session import init_db
from backend.db.models import User


async def seed_admin_user() -> None:
    """Seed the admin user"""

    admin_email = "admin@example.com"
    admin_password = "admin123"

    # Initialize database connection (creates async_session_maker)
    await init_db()

    # Import async_session_maker after init_db
    from backend.db.session import async_session_maker

    async with async_session_maker() as session:
        # Check if admin user already exists
        result = await session.execute(
            select(User).where(User.email == admin_email)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print(f"Admin user already exists: {admin_email}")
            # Update password to ensure it matches the expected one
            existing_user.hashed_password = get_password_hash(admin_password)
            existing_user.is_verified = True
            await session.commit()
            print(f"Updated admin user password")
        else:
            # Create new admin user
            admin_user = User(
                id=uuid.uuid4(),
                email=admin_email,
                full_name="Admin User",
                hashed_password=get_password_hash(admin_password),
                role="admin",
                is_active=True,
                is_verified=True,
            )
            session.add(admin_user)
            await session.commit()
            print(f"Created admin user: {admin_email}")

        print(f"Email: {admin_email}")
        print(f"Password: {admin_password}")


if __name__ == "__main__":
    asyncio.run(seed_admin_user())
