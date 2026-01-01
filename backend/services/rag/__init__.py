"""
RAG Orchestration Service
End-to-end Retrieval-Augmented Generation pipeline

This service provides:
- Complete RAG query processing
- Orchestration of retrieval, reranking, and LLM services
- Stage-by-stage timing and metrics
- Query result caching

Example:
    ```python
    from backend.services.rag import get_rag_service, RAGQueryOptions

    service = await get_rag_service()

    result = await service.query(
        query="機械学習と深層学習の違いは何ですか？",
        options=RAGQueryOptions(top_k=10, rerank=True),
    )

    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)} documents")
    print(f"Time: {result.processing_time_ms}ms")
    ```
"""

from backend.services.rag.pipeline import (
    RAGService,
    get_rag_service,
    query_rag,
    DEFAULT_RAG_SYSTEM_PROMPT,
)
from backend.services.rag.models import (
    # Main models
    RAGQueryOptions,
    RAGResult,
    RAGSource,
    RAGStageMetrics,
    RAGPipelineConfig,
    # Exceptions
    RAGValidationError,
    RAGProcessingError,
    RAGServiceError,
)

__all__ = [
    # Main service
    "RAGService",
    "get_rag_service",
    "query_rag",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    # Models
    "RAGQueryOptions",
    "RAGResult",
    "RAGSource",
    "RAGStageMetrics",
    "RAGPipelineConfig",
    # Exceptions
    "RAGValidationError",
    "RAGProcessingError",
    "RAGServiceError",
]
