"""
Common Pydantic Models
Shared schemas used across the application
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error detail model"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: ErrorDetail


class SuccessResponse(BaseModel):
    """Standard success response"""
    message: str
    data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="Application version")
    timestamp: Optional[str] = Field(None, description="Response timestamp")
    services: Optional[Dict[str, str]] = Field(None, description="Service health status")


class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(20, ge=1, le=100, description="Number of items per page")
    offset: int = Field(0, ge=0, description="Number of items to skip")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Items skipped")
    results: list = Field(..., description="List of items")
