"""
Document Pydantic Models
Request/response schemas for document endpoints
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import json

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

    @classmethod
    def from_db_model(cls, doc: DocumentSQLModel, owner_info: Optional[Dict] = None) -> "DocumentResponse":
        """Create DocumentResponse from database model"""
        # Parse keywords from JSON string
        keywords_list = []
        if doc.keywords:
            try:
                keywords_list = json.loads(doc.keywords) if isinstance(doc.keywords, str) else doc.keywords
            except (json.JSONDecodeError, TypeError):
                keywords_list = []

        # Parse metadata from JSON string
        metadata_dict = {}
        if doc.doc_metadata:
            try:
                metadata_dict = json.loads(doc.doc_metadata) if isinstance(doc.doc_metadata, str) else doc.doc_metadata
            except (json.JSONDecodeError, TypeError):
                metadata_dict = {}

        # Create owner info
        owner_dict = owner_info or {"id": str(doc.owner_id), "email": "", "full_name": ""}

        return cls(
            document_id=str(doc.id),
            filename=doc.filename,
            file_size_bytes=doc.file_size_bytes,
            file_hash=doc.file_hash,
            content_type=doc.content_type,
            title=doc.title,
            author=doc.author,
            keywords=keywords_list,
            language=doc.language,
            category=doc.category,
            metadata=metadata_dict,
            status=doc.status,
            page_count=doc.page_count,
            chunk_count=doc.chunk_count or 0,
            ocr_confidence=doc.ocr_confidence,
            thumbnail_url=doc.thumbnail_url,
            uploaded_at=doc.created_at,
            processing_completed_at=doc.processing_completed_at,
            owner=owner_dict,
        )


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
