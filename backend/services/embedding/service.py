"""
Embedding Service
Main service for text embedding generation with Sarashina model

This service provides:
- Single text embedding
- Batch text embedding
- Document chunk embedding
- Caching and performance monitoring
"""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.core.cache import cache_manager
from backend.services.embedding.models import (
    Embedding,
    TextEmbedding,
    EmbeddingResult,
    ChunkEmbeddingResult,
    EmbeddingOptions,
    EmbeddingProcessingError,
    EmbeddingValidationError,
)
from backend.services.embedding.sarashina import SarashinaEmbeddingModel

logger = get_logger(__name__)


class EmbeddingService:
    """
    Main embedding service for text embedding generation

    This service manages the Sarashina embedding model and provides
    a high-level API for generating embeddings for texts and document chunks.

    Example:
        ```python
        from backend.services.embedding.service import embedding_service

        # Embed a single text
        result = await embedding_service.embed_text("日本語のテキスト")

        # Embed multiple texts
        result = await embedding_service.embed_texts([
            "テキスト1",
            "テキスト2",
        ])

        # Embed document chunks
        result = await embedding_service.embed_chunks({
            "chunk1": "テキスト1",
            "chunk2": "テキスト2",
        })
        ```
    """

    def __init__(
        self,
        model: Optional[SarashinaEmbeddingModel] = None,
        options: Optional[EmbeddingOptions] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the embedding service

        Args:
            model: Sarashina model instance (created automatically if None)
            options: Embedding generation options
            enable_cache: Whether to enable embedding caching
        """
        self.model = model or SarashinaEmbeddingModel(options=options)
        self.options = self.model.options
        self.enable_cache = enable_cache and settings.EMBEDDING_CACHE_ENABLED
        self._is_initialized = False

        logger.info(
            f"EmbeddingService initialized (cache={self.enable_cache}, "
            f"model={self.model.model_name})"
        )

    async def initialize(self) -> None:
        """
        Initialize the embedding service

        Loads the embedding model into memory. This should be called
        before using the service, or it will be called automatically
        on the first request.
        """
        if self._is_initialized:
            logger.debug("EmbeddingService already initialized")
            return

        try:
            start_time = time.time()
            await self.model.load_model()
            self._is_initialized = True

            init_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"EmbeddingService initialized in {init_time}ms "
                f"(model={self.model.model_name}, dim={self.model.dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise EmbeddingProcessingError(
                f"Failed to initialize embedding service: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    async def embed_text(
        self,
        text: str,
        normalize: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> TextEmbedding:
        """
        Embed a single text

        Args:
            text: Input text to embed
            normalize: Whether to normalize embeddings (uses options default if None)
            use_cache: Whether to use cache (uses service default if None)

        Returns:
            TextEmbedding with vector and metadata

        Raises:
            EmbeddingValidationError: If text validation fails
            EmbeddingProcessingError: If embedding generation fails
        """
        # Validate input
        if not text or not isinstance(text, str):
            raise EmbeddingValidationError(
                "Text must be a non-empty string",
                details={"text_type": type(text).__name__},
            )

        # Trim whitespace
        text = text.strip()
        if not text:
            raise EmbeddingValidationError(
                "Text cannot be empty or whitespace only",
                details={"text_length": len(text)},
            )

        # Check cache if enabled
        should_cache = use_cache if use_cache is not None else self.enable_cache
        if should_cache:
            cached = await self._get_cached_embedding(text)
            if cached is not None:
                logger.debug(f"Cache hit for text (hash={cached.embedding.text_hash})")
                return cached

        # Ensure service is initialized
        if not self._is_initialized:
            await self.initialize()

        # Generate embedding
        try:
            result = await self.model.embed_text(text, normalize=normalize)

            # Cache result if enabled
            if should_cache:
                await self._cache_embedding(result)

            return result

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise EmbeddingProcessingError(
                f"Failed to embed text: {str(e)}",
                details={"text_length": len(text), "error_type": type(e).__name__},
            )

    async def embed_texts(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> EmbeddingResult:
        """
        Embed multiple texts in batches

        Args:
            texts: List of input texts to embed
            normalize: Whether to normalize embeddings (uses options default if None)
            use_cache: Whether to use cache (uses service default if None)

        Returns:
            EmbeddingResult with all embeddings and metadata

        Raises:
            EmbeddingValidationError: If text validation fails
            EmbeddingProcessingError: If embedding generation fails
        """
        # Validate input
        if not isinstance(texts, list):
            raise EmbeddingValidationError(
                "Texts must be a list of strings",
                details={"texts_type": type(texts).__name__},
            )

        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning(f"Skipping non-string text at index {i}")
                continue
            text = text.strip()
            if not text:
                logger.warning(f"Skipping empty text at index {i}")
                continue
            valid_texts.append(text)
            valid_indices.append(i)

        if not valid_texts:
            raise EmbeddingValidationError(
                "No valid texts provided",
                details={"total_texts": len(texts), "valid_texts": 0},
            )

        # Check cache for individual texts if enabled
        should_cache = use_cache if use_cache is not None else self.enable_cache
        cached_results: Dict[int, TextEmbedding] = {}
        texts_to_embed = []

        if should_cache:
            for i, text in enumerate(valid_texts):
                cached = await self._get_cached_embedding(text)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(valid_texts))

        # Embed non-cached texts
        if texts_to_embed:
            # Ensure service is initialized
            if not self._is_initialized:
                await self.initialize()

            try:
                texts_batch = [text for _, text in texts_to_embed]
                new_results = await self.model.embed_texts(texts_batch, normalize=normalize)

                # Cache new results
                for embedding in new_results.embeddings:
                    if should_cache:
                        await self._cache_embedding(embedding)

                # Merge with cached results
                for (i, _), embedding in zip(texts_to_embed, new_results.embeddings):
                    cached_results[i] = embedding

            except Exception as e:
                logger.error(f"Failed to embed texts: {e}")
                raise EmbeddingProcessingError(
                    f"Failed to embed texts: {str(e)}",
                    details={
                        "num_texts": len(texts_to_embed),
                        "error_type": type(e).__name__,
                    },
                )

        # Build final result in original order
        final_embeddings = [cached_results[i] for i in range(len(valid_texts))]

        # Calculate stats
        total_tokens = sum(emb.token_count for emb in final_embeddings)
        total_time = sum(emb.processing_time_ms for emb in final_embeddings)

        return EmbeddingResult(
            embeddings=final_embeddings,
            total_embeddings=len(final_embeddings),
            dimension=self.model.dimension,
            model=self.model.model_name,
            total_tokens=total_tokens,
            processing_time_ms=total_time,
            options=self.options,
            warnings=[],
        )

    async def embed_chunks(
        self,
        chunks: Dict[str, str],
        document_id: str,
        normalize: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> ChunkEmbeddingResult:
        """
        Embed document chunks

        This is a convenience method for embedding document chunks with
        proper document ID tracking.

        Args:
            chunks: Dictionary mapping chunk_id to text content
            document_id: Document UUID
            normalize: Whether to normalize embeddings
            use_cache: Whether to use cache

        Returns:
            ChunkEmbeddingResult with all chunk embeddings

        Raises:
            EmbeddingValidationError: If chunk validation fails
            EmbeddingProcessingError: If embedding generation fails
        """
        # Validate input
        if not chunks:
            raise EmbeddingValidationError(
                "Chunks dictionary cannot be empty",
                details={"document_id": document_id},
            )

        if not document_id:
            raise EmbeddingValidationError(
                "Document ID cannot be empty",
                details={"chunks_count": len(chunks)},
            )

        start_time = time.time()

        # Embed all chunks
        chunk_ids = list(chunks.keys())
        texts = list(chunks.values())

        result = await self.embed_texts(texts, normalize=normalize, use_cache=use_cache)

        # Map chunk IDs to embeddings
        chunk_embeddings: Dict[str, Embedding] = {}
        for chunk_id, text_embedding in zip(chunk_ids, result.embeddings):
            chunk_embeddings[chunk_id] = text_embedding.embedding

        processing_time = int((time.time() - start_time) * 1000)

        return ChunkEmbeddingResult(
            chunk_embeddings=chunk_embeddings,
            document_id=document_id,
            total_chunks=len(chunk_embeddings),
            dimension=self.model.dimension,
            model=self.model.model_name,
            processing_time_ms=processing_time,
        )

    async def _get_cached_embedding(self, text: str) -> Optional[TextEmbedding]:
        """
        Get embedding from cache

        Args:
            text: Input text

        Returns:
            Cached TextEmbedding or None if not found
        """
        try:
            import hashlib

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            cache_key = f"embedding:{self.model.model_name}:{text_hash}"

            cached_data = await cache_manager.get(cache_key)
            if cached_data is not None:
                # Reconstruct TextEmbedding from cached data
                return TextEmbedding(**cached_data)

        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")

        return None

    async def _cache_embedding(self, embedding: TextEmbedding) -> None:
        """
        Cache embedding result

        Args:
            embedding: TextEmbedding to cache
        """
        try:
            text_hash = embedding.embedding.text_hash
            cache_key = f"embedding:{self.model.model_name}:{text_hash}"

            # Cache for 24 hours
            await cache_manager.set(
                cache_key,
                embedding.model_dump(),
                ttl=settings.EMBEDDING_CACHE_TTL,
            )

        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension"""
        return self.model.dimension

    @property
    def max_length(self) -> int:
        """Return the maximum token length"""
        return self.model.max_length

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self._is_initialized

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the embedding service

        Returns:
            Dictionary with health status and metrics
        """
        health = {
            "status": "healthy" if self._is_initialized else "uninitialized",
            "model": self.model.model_name,
            "dimension": self.model.dimension,
            "max_length": self.model.max_length,
            "cache_enabled": self.enable_cache,
            "is_loaded": self.model.is_loaded(),
        }

        if self._is_initialized:
            # Test embedding generation
            try:
                start_time = time.time()
                test_result = await self.embed_text("テスト", use_cache=False)
                health["test_embedding_time_ms"] = test_result.processing_time_ms
                health["test_token_count"] = test_result.token_count
                health["test_dimension"] = len(test_result.embedding.vector)
            except Exception as e:
                health["status"] = "error"
                health["error"] = str(e)

        return health

    async def shutdown(self) -> None:
        """
        Shutdown the embedding service and free resources
        """
        if self._is_initialized:
            await self.model.unload()
            self._is_initialized = False
            logger.info("EmbeddingService shutdown complete")


# Global singleton instance
_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance

    Returns:
        EmbeddingService singleton
    """
    global _embedding_service

    if _embedding_service is None:
        options = EmbeddingOptions(
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize=settings.EMBEDDING_NORMALIZE,
            truncate=settings.EMBEDDING_TRUNCATE,
            show_progress=False,  # Never show progress in API context
        )
        _embedding_service = EmbeddingService(options=options)
        await _embedding_service.initialize()

    return _embedding_service


# Convenience export
embedding_service: EmbeddingService = None  # Will be initialized by get_embedding_service()
