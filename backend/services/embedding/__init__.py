"""
Embedding Service
Main embedding service with model management and batch processing

This module provides:
- EmbeddingService: High-level service for text embedding
- SarashinaEmbeddingModel: Sarashina-1B Japanese embedding model
- Convenience functions for common operations
"""

# Import main service and models
from backend.services.embedding.service import (
    EmbeddingService,
    get_embedding_service,
)
from backend.services.embedding.base import BaseEmbeddingModel
from backend.services.embedding.sarashina import SarashinaEmbeddingModel

# Import all models
from backend.services.embedding.models import (
    Embedding,
    TextEmbedding,
    EmbeddingResult,
    EmbeddingOptions,
    ChunkEmbeddingResult,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmbeddingProcessingError,
    EmbeddingValidationError,
)

# Legacy convenience functions (using the new service)
from typing import List, Optional, Any


async def embed_text(
    text: str,
    model: str = None,
    options: EmbeddingOptions = None,
) -> TextEmbedding:
    """
    Convenience function to embed a single text

    Args:
        text: Input text
        model: Model name (currently only 'sarashina' supported)
        options: Embedding options

    Returns:
        TextEmbedding with vector and metadata
    """
    service = await get_embedding_service()
    return await service.embed_text(text, use_cache=False)


async def embed_texts(
    texts: List[str],
    model: str = None,
    options: EmbeddingOptions = None,
) -> EmbeddingResult:
    """
    Convenience function to embed multiple texts

    Args:
        texts: List of input texts
        model: Model name (currently only 'sarashina' supported)
        options: Embedding options

    Returns:
        EmbeddingResult with all embeddings
    """
    service = await get_embedding_service()
    return await service.embed_texts(texts, use_cache=False)


async def embed_chunks(
    chunks: List[Any],  # List[TextChunk]
    model: str = None,
    options: EmbeddingOptions = None,
) -> ChunkEmbeddingResult:
    """
    Convenience function to embed document chunks

    Args:
        chunks: List of TextChunk objects from chunking service
        model: Model name (currently only 'sarashina' supported)
        options: Embedding options

    Returns:
        ChunkEmbeddingResult with chunk embeddings
    """
    import time

    service = await get_embedding_service()

    # Extract texts and chunk IDs
    texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    document_id = chunks[0].metadata.document_id if chunks else "unknown"

    # Create chunks dict for service
    chunks_dict = dict(zip(chunk_ids, texts))

    start_time = time.time()
    result = await service.embed_chunks(
        chunks_dict,
        document_id=document_id,
        use_cache=False,
    )
    processing_time = int((time.time() - start_time) * 1000)

    # Update processing time to include overhead
    result.processing_time_ms = processing_time

    return result


__all__ = [
    # Service
    "EmbeddingService",
    "get_embedding_service",
    # Models
    "BaseEmbeddingModel",
    "SarashinaEmbeddingModel",
    # Pydantic models
    "Embedding",
    "TextEmbedding",
    "EmbeddingResult",
    "EmbeddingOptions",
    "ChunkEmbeddingResult",
    # Exceptions
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingProcessingError",
    "EmbeddingValidationError",
    # Convenience functions
    "embed_text",
    "embed_texts",
    "embed_chunks",
]
