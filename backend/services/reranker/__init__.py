"""
Reranking Service
Cross-encoder reranking for improved search result relevance using NVIDIA Llama-3.2-NV

This service provides:
- Document reranking using cross-encoder models
- Query-document relevance scoring
- Batch reranking for efficiency
- GPU-accelerated inference
"""

from backend.services.reranker.service import RerankingService, get_reranking_service

__all__ = [
    "RerankingService",
    "get_reranking_service",
]
