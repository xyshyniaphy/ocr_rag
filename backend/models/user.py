"""
User Pydantic Models
Request/response schemas for user management
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UserListResponse(BaseModel):
    """User list response schema"""
    total: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]


class UserStatsResponse(BaseModel):
    """User statistics response schema"""
    user_id: str
    name: str
    email: str
    role: str
    document_count: int
    query_count: int
    average_rating: Optional[float]
