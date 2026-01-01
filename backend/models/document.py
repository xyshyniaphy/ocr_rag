"""
Document Pydantic Models
Request/response schemas for document endpoints
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Import SQLAlchemy model
from backend.db.models import Document as DocumentSQLModel


class DocumentMetadata(BaseModel):
    """Document metadata schema"""
    title: Optional[str] = Field(None, max_length=512)
    author: Optional[str] = Field(None, max_length=255)
    subject: Optional[str] = Field(None, max_length=255)
    keywords: List[str] = Field(default_factory=list)
    language: str = Field("ja", max_length=10)
    category: Optional[str] = Field(None, max_length=100)


class DocumentUploadRequest(BaseModel):
    """Document upload request schema"""
    metadata: Optional[DocumentMetadata] = None


class DocumentResponse(BaseModel):
    """Document response schema"""
    document_id: str
    filename: str
    file_size_bytes: int
    file_hash: str
    content_type: str
    title: Optional[str]
    author: Optional[str]
    keywords: List[str]
    language: str
    category: Optional[str]
    metadata: Dict[str, Any]
    status: str
    page_count: Optional[int]
    chunk_count: int
    ocr_confidence: Optional[float]
    thumbnail_url: Optional[str]
    uploaded_at: datetime
    processing_completed_at: Optional[datetime]
    owner: Dict[str, str]

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """Document status response schema"""
    document_id: str
    status: str
    progress: int
    current_stage: Optional[str]
    stages: Optional[Dict[str, Dict[str, Any]]]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Document list response schema"""
    total: int
    limit: int
    offset: int
    results: List[DocumentResponse]


class ChunkResponse(BaseModel):
    """Chunk response schema"""
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text_content: str
    chunk_type: str
    confidence: Optional[float]
    section_header: Optional[str]

    model_config = {"from_attributes": True}


class SourceReference(BaseModel):
    """Source reference schema"""
    document_id: str
    document_title: str
    page_number: int
    chunk_index: Optional[int]
    chunk_text: Optional[str]
    relevance_score: float
    rerank_score: Optional[float]


# Export SQLAlchemy model for backward compatibility with API imports
Document = DocumentSQLModel
