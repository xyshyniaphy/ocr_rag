"""
API Dependencies
Common dependencies for API routes
"""

from typing import Optional
from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import verify_access_token
from backend.core.exceptions import AuthenticationException
from backend.db.session import get_db_session
from backend.db.models import User as UserModel
from sqlalchemy import select


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session),
) -> UserModel:
    """
    Dependency to get current user from JWT token

    Args:
        authorization: Authorization header with Bearer token
        db: Database session

    Returns:
        Current authenticated user

    Raises:
        AuthenticationException: If token is invalid or user not found
    """
    # Extract token
    if not authorization:
        raise AuthenticationException(message="Missing authorization header")

    if not authorization.startswith("Bearer "):
        raise AuthenticationException(message="Invalid authorization header format")

    token = authorization.split(" ")[1]

    # Verify token
    payload = verify_access_token(token)

    # Get user
    result = await db.execute(
        select(UserModel).where(UserModel.id == payload["sub"])
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise AuthenticationException(message="Invalid token or user inactive")

    return user


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session),
) -> Optional[UserModel]:
    """
    Optional dependency to get current user from JWT token
    Returns None if not authenticated instead of raising exception

    Args:
        authorization: Authorization header with Bearer token
        db: Database session

    Returns:
        Current authenticated user or None
    """
    try:
        return await get_current_user(authorization, db)
    except AuthenticationException:
        return None
