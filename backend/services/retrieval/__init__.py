"""
Retrieval Services
Provides vector, keyword, and hybrid retrieval for RAG
"""

from backend.services.retrieval.base import BaseRetriever, RetrieverMixin
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.retrieval.keyword import KeywordRetriever
from backend.services.retrieval.models import (
    HybridRetrievalConfig,
    RetrievedChunk,
    RetrievalOptions,
    RetrievalResult,
)
from backend.services.retrieval.service import (
    RetrievalService,
    get_retrieval_service,
    health_check as retrieval_health_check,
    retrieve,
)
from backend.services.retrieval.vector import VectorRetriever

__all__ = [
    # Main service
    "RetrievalService",
    "get_retrieval_service",
    "retrieve",
    "retrieval_health_check",
    # Retrievers
    "BaseRetriever",
    "RetrieverMixin",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    # Models
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalOptions",
    "HybridRetrievalConfig",
]
