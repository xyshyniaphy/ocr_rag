"""
Embedding Service
Main embedding service with model management and batch processing
"""

import asyncio
from typing import List, Optional, Dict, Any
import time

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.embedding.models import (
    Embedding,
    TextEmbedding,
    EmbeddingResult,
    EmbeddingOptions,
    ChunkEmbeddingResult,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmbeddingProcessingError,
)
from backend.services.embedding.base import BaseEmbeddingModel

logger = get_logger(__name__)


class EmbeddingService:
    """
    Main embedding service that manages embedding models

    Features:
    - Model selection (Sarashina, future models)
    - GPU memory management
    - Batch processing
    - Text and chunk embedding
    """

    # Model registry
    _models: Dict[str, type[BaseEmbeddingModel]] = {}
    _instances: Dict[str, BaseEmbeddingModel] = {}
    _lock = asyncio.Lock()

    @classmethod
    def register_model(cls, name: str, model_class: type[BaseEmbeddingModel]) -> None:
        """
        Register an embedding model

        Args:
            name: Model name
            model_class: Model class (must inherit from BaseEmbeddingModel)
        """
        cls._models[name] = model_class
        logger.info(f"Registered embedding model: {name}")

    @classmethod
    async def get_model(
        cls,
        name: str = "sarashina",
        options: Optional[EmbeddingOptions] = None,
    ) -> BaseEmbeddingModel:
        """
        Get or create an embedding model instance

        Args:
            name: Model name ('sarashina')
            options: Embedding options

        Returns:
            BaseEmbeddingModel instance

        Raises:
            EmbeddingModelNotFoundError: If model is not registered
        """
        async with cls._lock:
            # Return existing instance if available
            if name in cls._instances:
                return cls._instances[name]

            # Check if model is registered
            if name not in cls._models:
                raise EmbeddingModelNotFoundError(
                    f"Embedding model '{name}' not registered",
                    details={"available_models": list(cls._models.keys())},
                )

            # Create new instance
            model_class = cls._models[name]
            instance = model_class(options)

            # Load the model
            await instance.load_model()

            # Cache the instance
            cls._instances[name] = instance
            logger.info(f"Created embedding model instance: {name}")

            return instance

    @classmethod
    async def embed_text(
        cls,
        text: str,
        model: str = "sarashina",
        options: Optional[EmbeddingOptions] = None,
    ) -> TextEmbedding:
        """
        Embed a single text

        Args:
            text: Input text
            model: Model name
            options: Embedding options

        Returns:
            TextEmbedding with vector and metadata
        """
        embedding_model = await cls.get_model(model, options)
        return await embedding_model.embed_text(text)

    @classmethod
    async def embed_texts(
        cls,
        texts: List[str],
        model: str = "sarashina",
        options: Optional[EmbeddingOptions] = None,
    ) -> EmbeddingResult:
        """
        Embed multiple texts in batches

        Args:
            texts: List of input texts
            model: Model name
            options: Embedding options

        Returns:
            EmbeddingResult with all embeddings
        """
        embedding_model = await cls.get_model(model, options)
        return await embedding_model.embed_texts(texts)

    @classmethod
    async def embed_chunks(
        cls,
        chunks: List[Any],  # List[TextChunk]
        model: str = "sarashina",
        options: Optional[EmbeddingOptions] = None,
    ) -> ChunkEmbeddingResult:
        """
        Embed document chunks

        Args:
            chunks: List of TextChunk objects from chunking service
            model: Model name
            options: Embedding options

        Returns:
            ChunkEmbeddingResult with chunk_id -> embedding mapping
        """
        start_time = time.time()
        document_id = chunks[0].metadata.document_id if chunks else "unknown"

        # Extract texts and chunk IDs
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Generate embeddings
        result = await cls.embed_texts(texts, model, options)

        # Map chunk IDs to embeddings
        chunk_embeddings = {}
        for chunk_id, text_emb in zip(chunk_ids, result.embeddings):
            chunk_embeddings[chunk_id] = text_emb.embedding

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Embedded {len(chunks)} chunks for document {document_id}: "
            f"{processing_time}ms"
        )

        return ChunkEmbeddingResult(
            chunk_embeddings=chunk_embeddings,
            document_id=document_id,
            total_chunks=len(chunks),
            dimension=result.dimension,
            model=result.model,
            processing_time_ms=processing_time,
        )

    @classmethod
    async def unload_model(cls, name: str) -> None:
        """
        Unload an embedding model and free memory

        Args:
            name: Model name
        """
        async with cls._lock:
            if name in cls._instances:
                await cls._instances[name].unload()
                del cls._instances[name]
                logger.info(f"Unloaded embedding model: {name}")

    @classmethod
    async def unload_all(cls) -> None:
        """Unload all embedding models and free memory"""
        async with cls._lock:
            for name, instance in cls._instances.items():
                try:
                    await instance.unload()
                except Exception as e:
                    logger.error(f"Error unloading {name}: {e}")
            cls._instances.clear()
            logger.info("Unloaded all embedding models")

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of registered model names"""
        return list(cls._models.keys())

    @classmethod
    def get_loaded_models(cls) -> List[str]:
        """Get list of loaded model names"""
        return list(cls._instances.keys())


# Register models
from backend.services.embedding.sarashina import SarashinaEmbeddingModel

EmbeddingService.register_model("sarashina", SarashinaEmbeddingModel)


# Convenience functions for simple usage

async def embed_text(
    text: str,
    model: str = None,
    options: EmbeddingOptions = None,
) -> TextEmbedding:
    """
    Convenience function to embed a single text

    Args:
        text: Input text
        model: Model name (uses default from config if None)
        options: Embedding options

    Returns:
        TextEmbedding with vector and metadata
    """
    if model is None:
        model = "sarashina"

    return await EmbeddingService.embed_text(text, model, options)


async def embed_texts(
    texts: List[str],
    model: str = None,
    options: EmbeddingOptions = None,
) -> EmbeddingResult:
    """
    Convenience function to embed multiple texts

    Args:
        texts: List of input texts
        model: Model name (uses default from config if None)
        options: Embedding options

    Returns:
        EmbeddingResult with all embeddings
    """
    if model is None:
        model = "sarashina"

    return await EmbeddingService.embed_texts(texts, model, options)


async def embed_chunks(
    chunks: List[Any],  # List[TextChunk]
    model: str = None,
    options: EmbeddingOptions = None,
) -> ChunkEmbeddingResult:
    """
    Convenience function to embed document chunks

    Args:
        chunks: List of TextChunk objects
        model: Model name (uses default from config if None)
        options: Embedding options

    Returns:
        ChunkEmbeddingResult with chunk embeddings
    """
    if model is None:
        model = "sarashina"

    return await EmbeddingService.embed_chunks(chunks, model, options)


__all__ = [
    "EmbeddingService",
    "embed_text",
    "embed_texts",
    "embed_chunks",
    "SarashinaEmbeddingModel",
    # Models
    "Embedding",
    "TextEmbedding",
    "EmbeddingResult",
    "EmbeddingOptions",
    "ChunkEmbeddingResult",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingProcessingError",
]
