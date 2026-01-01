"""
Authentication Pydantic Models
Request/response schemas for authentication endpoints
"""

from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


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
    last_login_at: Optional[str]
    permissions: dict

    model_config = {"from_attributes": True}


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
