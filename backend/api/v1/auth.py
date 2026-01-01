"""
Authentication API Routes
Login, logout, token refresh, and user management
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from backend.core.exceptions import AuthenticationException, ValidationException
from backend.db.session import get_db_session
from backend.db.models import User as UserModel
from backend.models.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
)

logger = get_logger(__name__)
router = APIRouter()


# ============================================
# Dependencies
# ============================================
async def get_current_user_dependency(
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
    from backend.core.security import verify_access_token
    from sqlalchemy import select

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


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Authenticate user and return JWT tokens

    - **email**: User email address
    - **password**: User password
    """
    from sqlalchemy import select

    # Find user
    result = await db.execute(
        select(UserModel).where(UserModel.email == request.email)
    )
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(request.password, user.hashed_password):
        logger.warning(f"Failed login attempt for email: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()

    # Create tokens
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_user_model(user),
    )


@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Register a new user account

    - **email**: User email address (must be unique)
    - **password**: User password (min 8 characters)
    - **full_name**: User's full name
    """
    from sqlalchemy import select
    import uuid

    # Check if user already exists
    result = await db.execute(
        select(UserModel).where(UserModel.email == request.email)
    )
    if result.scalar_one_or_none():
        raise ValidationException(
            message="Email already registered",
            details={"email": request.email},
        )

    # Create new user
    user = UserModel(
        id=uuid.uuid4(),
        email=request.email,
        hashed_password=get_password_hash(request.password),
        full_name=request.full_name,
        display_name=request.display_name,
        role=request.role,
        is_active=True,
        is_verified=False,
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info(f"New user registered: {user.email}")

    # Create tokens
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_user_model(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Refresh access token using refresh token

    - **refresh_token**: Valid refresh token
    """
    from sqlalchemy import select

    # Verify refresh token
    payload = verify_refresh_token(request.refresh_token)

    # Get user
    result = await db.execute(
        select(UserModel).where(UserModel.id == payload["sub"])
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise AuthenticationException(message="Invalid token")

    # Create new tokens
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="Bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_user_model(user),
    )


@router.post("/logout")
async def logout():
    """
    Logout user (invalidate tokens)

    Note: In a stateless JWT setup, tokens are invalidated on the client side.
    For true invalidation, implement a token blacklist in Redis.
    """
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: UserModel = Depends(get_current_user_dependency),
):
    """
    Get current authenticated user information
    """
    return UserResponse.from_user_model(current_user)
