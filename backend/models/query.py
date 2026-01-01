"""
Query Pydantic Models
Request/response schemas for query endpoints
"""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from .document import SourceReference


class QueryRequest(BaseModel):
    """Query request schema"""
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[str]] = Field(None, max_length=10)
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True
    language: str = Field("ja", max_length=10)
    stream: bool = False
    rerank: bool = True


class QueryResponse(BaseModel):
    """Query response schema"""
    query_id: str
    query: str
    answer: str
    sources: List[SourceReference]
    processing_time_ms: int
    stage_timings_ms: Optional[Dict[str, int]]
    confidence: Optional[float]
    timestamp: str


class QueryListResponse(BaseModel):
    """Query list response schema"""
    total: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]


class QueryFeedbackRequest(BaseModel):
    """Query feedback request schema"""
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    is_helpful: Optional[bool] = None
    user_feedback: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request schema"""
    q: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SearchResponse(BaseModel):
    """Search response schema"""
    total: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]
