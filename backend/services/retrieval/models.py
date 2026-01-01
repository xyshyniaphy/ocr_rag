"""
Retrieval Models
Pydantic models for retrieval operations and results
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with metadata"""

    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Document ID")
    text: str = Field(description="Chunk text content")
    score: float = Field(ge=0.0, le=1.0, description="Similarity/relevance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    source: str = Field(
        description="Retrieval source: vector, keyword, or hybrid"
    )
    page_number: Optional[int] = Field(default=None, description="Page number")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_123",
                "text": "This is a sample chunk of text...",
                "score": 0.92,
                "metadata": {"page": 1, "section": "Introduction"},
                "source": "vector",
                "page_number": 1,
                "chunk_index": 0,
            }
        }


class RetrievalResult(BaseModel):
    """Result from retrieval operation"""

    chunks: List[RetrievedChunk] = Field(description="List of retrieved chunks")
    total_results: int = Field(description="Total number of results found")
    query: str = Field(description="Original query text")
    retrieval_method: str = Field(description="Method used: vector, keyword, or hybrid")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional retrieval metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "chunk_id": "chunk_001",
                        "document_id": "doc_123",
                        "text": "Sample text...",
                        "score": 0.92,
                        "source": "hybrid",
                        "page_number": 1,
                    }
                ],
                "total_results": 10,
                "query": "What is the main topic?",
                "retrieval_method": "hybrid",
                "execution_time_ms": 145.5,
            }
        }


class HybridRetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval"""

    vector_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for vector search results"
    )
    keyword_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for keyword search results"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    min_score: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum score threshold"
    )
    rrf_k: int = Field(
        default=60, description="RRF constant for rank fusion (higher = less rank fusion)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "top_k": 10,
                "min_score": 0.3,
                "rrf_k": 60,
            }
        }


class RetrievalOptions(BaseModel):
    """Options for retrieval operations"""

    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(
        default=None, description="Filter by document IDs"
    )
    include_metadata: bool = Field(default=True)
    use_cache: bool = Field(default=True)

    class Config:
        json_schema_extra = {
            "example": {
                "top_k": 10,
                "min_score": 0.3,
                "document_ids": ["doc_123", "doc_456"],
                "include_metadata": True,
                "use_cache": True,
            }
        }
