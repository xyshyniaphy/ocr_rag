"""
Authentication Pydantic Models
Request/response schemas for authentication endpoints
"""

from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_serializer


class UserBase(BaseModel):
    """Base user fields"""
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=100)
    role: str = Field("user", pattern="^(admin|power_user|user|viewer)$")


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, pattern="^(admin|power_user|user|viewer)$")


class UserResponse(UserBase):
    """User response schema"""
    user_id: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login_at: Optional[str] = None
    permissions: dict = Field(default_factory=dict)

    # Allow SQLAlchemy model to populate this response
    model_config = {"from_attributes": True, "populate_by_name": True}

    @field_serializer("user_id", mode="plain")
    def serialize_user_id(self, value: Optional[str]) -> str:
        """Serialize user_id from id field"""
        if value is None:
            # When loading from SQLAlchemy model, id becomes user_id
            return str(getattr(self, "id", value))
        return value

    @classmethod
    def from_user_model(cls, user) -> "UserResponse":
        """Create UserResponse from SQLAlchemy User model"""
        return cls(
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            display_name=user.display_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at.isoformat() if user.created_at else "",
            last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
            permissions={},
        )


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserResponse


class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema"""
    refresh_token: str


class RegisterRequest(UserCreate):
    """Registration request schema"""
    pass


class ChangePasswordRequest(BaseModel):
    """Change password request schema"""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
