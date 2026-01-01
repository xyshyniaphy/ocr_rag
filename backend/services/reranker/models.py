"""
Reranker Models
Pydantic models for reranking service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RerankerOptions(BaseModel):
    """Options for reranking configuration"""

    top_k_input: int = Field(
        default=20,
        description="Number of input documents to rerank",
        ge=1,
        le=100,
    )
    top_k_output: int = Field(
        default=5,
        description="Number of top documents to return after reranking",
        ge=1,
        le=50,
    )
    threshold: float = Field(
        default=0.65,
        description="Minimum relevance score threshold (0-1)",
        ge=0.0,
        le=1.0,
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for reranking inference",
        ge=1,
        le=128,
    )
    return_documents: bool = Field(
        default=True,
        description="Whether to return document text in results",
    )
    return_scores: bool = Field(
        default=True,
        description="Whether to return relevance scores",
    )


class RerankerDocument(BaseModel):
    """Document to be reranked"""

    text: str = Field(..., description="Document text content")
    doc_id: Optional[str] = Field(None, description="Document ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    original_score: Optional[float] = Field(None, description="Original retrieval score")


class RerankerResult(BaseModel):
    """Result from reranking"""

    doc_id: Optional[str] = Field(None, description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score from reranker", ge=0.0, le=1.0)
    rank: int = Field(..., description="Rank after reranking (1-indexed)", ge=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    original_score: Optional[float] = Field(None, description="Original retrieval score")


class RerankingOutput(BaseModel):
    """Output from reranking process"""

    results: List[RerankerResult] = Field(
        default_factory=list,
        description="Reranked and filtered results",
    )
    query: str = Field(..., description="Original query")
    total_input: int = Field(..., description="Number of input documents")
    total_output: int = Field(..., description="Number of output documents after filtering")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="Model used for reranking")
    threshold_applied: float = Field(..., description="Score threshold applied")


class RerankingError(Exception):
    """Base exception for reranking errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RerankingValidationError(RerankingError):
    """Raised when input validation fails"""

    pass


class RerankingProcessingError(RerankingError):
    """Raised when reranking processing fails"""

    pass
