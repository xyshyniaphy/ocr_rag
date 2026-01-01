"""
RAG Models
Pydantic models for RAG orchestration operations and results
"""

from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field


class RAGQueryOptions(BaseModel):
    """Options for RAG query processing"""

    top_k: int = Field(default=10, ge=1, le=100, description="Number of documents to retrieve")
    retrieval_top_k: int = Field(
        default=20, ge=1, le=100, description="Number of documents to retrieve before reranking"
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of documents to keep after reranking",
    )
    rerank: bool = Field(default=True, description="Whether to apply reranking")
    retrieval_method: str = Field(
        default="hybrid",
        description="Retrieval method: vector, keyword, or hybrid",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold",
    )
    document_ids: Optional[Sequence[str]] = Field(
        default=None, description="Filter by specific document IDs"
    )
    include_sources: bool = Field(default=True, description="Include source chunks in response")
    use_cache: bool = Field(default=True, description="Use query cache")
    language: str = Field(default="ja", description="Query language (ja, en, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "top_k": 10,
                "retrieval_top_k": 20,
                "rerank_top_k": 10,
                "rerank": True,
                "retrieval_method": "hybrid",
                "min_score": 0.3,
                "document_ids": None,
                "include_sources": True,
                "use_cache": True,
                "language": "ja",
            }
        }


class RAGSource(BaseModel):
    """A single source document used in RAG generation"""

    chunk_id: str = Field(description="Chunk identifier")
    document_id: str = Field(description="Document ID")
    document_title: Optional[str] = Field(default=None, description="Document title")
    text: str = Field(description="Chunk text content")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    rerank_score: Optional[float] = Field(
        default=None, description="Reranking score (if reranking was applied)"
    )
    page_number: Optional[int] = Field(default=None, description="Page number")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_123",
                "document_title": "Sample Document",
                "text": "This is a sample chunk of text...",
                "score": 0.92,
                "rerank_score": 0.95,
                "page_number": 1,
                "chunk_index": 0,
                "metadata": {"source": "vector"},
            }
        }


class RAGStageMetrics(BaseModel):
    """Metrics for a single RAG pipeline stage"""

    stage_name: str = Field(description="Stage name")
    duration_ms: float = Field(description="Stage duration in milliseconds")
    success: bool = Field(description="Whether stage completed successfully")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional stage metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "stage_name": "retrieval",
                "duration_ms": 145.5,
                "success": True,
                "error": None,
                "metadata": {"retrieved_count": 20, "method": "hybrid"},
            }
        }


class RAGResult(BaseModel):
    """Result from RAG query processing"""

    query: str = Field(description="Original query text")
    answer: str = Field(description="Generated answer")
    sources: List[RAGSource] = Field(description="Source documents used")
    query_id: Optional[str] = Field(default=None, description="Query identifier")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")
    stage_timings: List[RAGStageMetrics] = Field(description="Timing breakdown by stage")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Answer confidence score"
    )
    llm_model: Optional[str] = Field(default=None, description="LLM model used")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model used")
    reranker_model: Optional[str] = Field(default=None, description="Reranker model used")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is...",
                "sources": [
                    {
                        "chunk_id": "chunk_001",
                        "document_id": "doc_123",
                        "text": "Machine learning is...",
                        "score": 0.92,
                    }
                ],
                "query_id": "query_123",
                "processing_time_ms": 2540.5,
                "stage_timings": [
                    {
                        "stage_name": "query_understanding",
                        "duration_ms": 1.5,
                        "success": True,
                    },
                    {
                        "stage_name": "retrieval",
                        "duration_ms": 145.5,
                        "success": True,
                    },
                    {
                        "stage_name": "reranking",
                        "duration_ms": 523.2,
                        "success": True,
                    },
                    {
                        "stage_name": "context_assembly",
                        "duration_ms": 2.3,
                        "success": True,
                    },
                    {
                        "stage_name": "llm_generation",
                        "duration_ms": 1868.0,
                        "success": True,
                    },
                ],
                "confidence": 0.85,
                "llm_model": "qwen3:4b",
                "embedding_model": "sbintuitions/sarashina-embedding-v1-1b",
                "reranker_model": "llama-3.2-nv-rerankerqa",
            }
        }


class RAGPipelineConfig(BaseModel):
    """Configuration for RAG pipeline"""

    # Retrieval settings
    retrieval_method: str = Field(default="hybrid", description="Retrieval method")
    retrieval_top_k: int = Field(default=20, description="Initial retrieval count")
    min_score: float = Field(default=0.0, description="Minimum relevance score")

    # Reranking settings
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    rerank_top_k: int = Field(default=10, description="Documents to keep after reranking")

    # LLM settings
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=2048, description="Max tokens to generate")
    llm_top_p: float = Field(default=0.9, description="LLM top-p sampling")

    # System prompt
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt for LLM"
    )

    # Cache settings
    enable_cache: bool = Field(default=True, description="Enable query cache")

    # Language settings
    default_language: str = Field(default="ja", description="Default query language")

    class Config:
        json_schema_extra = {
            "example": {
                "retrieval_method": "hybrid",
                "retrieval_top_k": 20,
                "min_score": 0.0,
                "enable_reranking": True,
                "rerank_top_k": 10,
                "llm_temperature": 0.1,
                "llm_max_tokens": 2048,
                "llm_top_p": 0.9,
                "system_prompt": None,
                "enable_cache": True,
                "default_language": "ja",
            }
        }


# Custom exceptions
class RAGValidationError(Exception):
    """RAG query validation error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RAGProcessingError(Exception):
    """RAG pipeline processing error"""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)


class RAGServiceError(Exception):
    """RAG service initialization or connection error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
