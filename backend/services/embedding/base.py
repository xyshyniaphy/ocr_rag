"""
Embedding Base Classes
Abstract base class and common functionality for embedding models
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import hashlib
import time

from backend.core.logging import get_logger
from backend.services.embedding.models import (
    Embedding,
    TextEmbedding,
    EmbeddingResult,
    EmbeddingOptions,
    EmbeddingProcessingError,
)

logger = get_logger(__name__)


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models

    All embedding model implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, options: Optional[EmbeddingOptions] = None):
        """
        Initialize the embedding model

        Args:
            options: Embedding generation options
        """
        self.options = options or EmbeddingOptions()
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of this embedding model"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension"""
        pass

    @property
    @abstractmethod
    def max_length(self) -> int:
        """Return the maximum token length"""
        pass

    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the embedding model into memory

        This method should load the model and set self._is_loaded = True
        """
        pass

    @abstractmethod
    async def _encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Encode a batch of texts to embeddings

        Args:
            texts: List of input texts
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        pass

    async def embed_text(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> TextEmbedding:
        """
        Embed a single text

        Args:
            text: Input text
            normalize: Whether to normalize (uses options default if None)

        Returns:
            TextEmbedding with vector and metadata
        """
        if not self._is_loaded:
            await self.load_model()

        start_time = time.time()

        # Truncate if needed
        if self.options.truncate:
            text = self._truncate_text(text)

        # Estimate token count
        token_count = self._estimate_tokens(text)

        # Generate embedding
        embeddings = await self._encode_batch(
            [text],
            normalize=normalize if normalize is not None else self.options.normalize,
        )

        vector = embeddings[0]
        processing_time = int((time.time() - start_time) * 1000)

        # Create text hash for caching
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        return TextEmbedding(
            text=text,
            embedding=Embedding(
                vector=vector,
                dimension=self.dimension,
                model=self.model_name,
                text_hash=text_hash,
            ),
            token_count=token_count,
            processing_time_ms=processing_time,
        )

    async def embed_texts(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
    ) -> EmbeddingResult:
        """
        Embed multiple texts in batches

        Args:
            texts: List of input texts
            normalize: Whether to normalize (uses options default if None)

        Returns:
            EmbeddingResult with all embeddings
        """
        if not self._is_loaded:
            await self.load_model()

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                total_embeddings=0,
                dimension=self.dimension,
                model=self.model_name,
                total_tokens=0,
                processing_time_ms=0,
                options=self.options,
                warnings=[],
            )

        start_time = time.time()
        warnings = []

        # Process in batches
        batch_size = self.options.batch_size
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Truncate if needed
            if self.options.truncate:
                batch_texts = [self._truncate_text(t) for t in batch_texts]

            # Estimate tokens
            batch_tokens = sum(self._estimate_tokens(t) for t in batch_texts)
            total_tokens += batch_tokens

            # Generate embeddings
            batch_vectors = await self._encode_batch(
                batch_texts,
                normalize=normalize if normalize is not None else self.options.normalize,
            )

            # Create TextEmbedding objects
            for text, vector in zip(batch_texts, batch_vectors):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                all_embeddings.append(
                    TextEmbedding(
                        text=text,
                        embedding=Embedding(
                            vector=vector,
                            dimension=self.dimension,
                            model=self.model_name,
                            text_hash=text_hash,
                        ),
                        token_count=self._estimate_tokens(text),
                        processing_time_ms=0,  # Not tracked per-item in batch
                    )
                )

            # Warn if any text was truncated
            if self.options.truncate:
                for j, text in enumerate(batch_texts):
                    orig_len = len(texts[i + j])
                    truncated_len = len(text)
                    if orig_len > truncated_len:
                        warnings.append(
                            f"Text {i + j} truncated from {orig_len} to {truncated_len} chars"
                        )

        processing_time = int((time.time() - start_time) * 1000)

        return EmbeddingResult(
            embeddings=all_embeddings,
            total_embeddings=len(all_embeddings),
            dimension=self.dimension,
            model=self.model_name,
            total_tokens=total_tokens,
            processing_time_ms=processing_time,
            options=self.options,
            warnings=warnings,
        )

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to maximum length

        Args:
            text: Input text

        Returns:
            Truncated text
        """
        # Rough estimate: Japanese chars ≈ 2 tokens, ASCII ≈ 4 chars per token
        # This is conservative to ensure we stay under max_length
        max_chars = self.max_length * 2  # Conservative estimate

        if len(text) <= max_chars:
            return text

        return text[:max_chars]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate for Japanese text
        japanese_chars = sum(1 for c in text if ord(c) > 0x3000)
        ascii_chars = len(text) - japanese_chars

        # Japanese chars ≈ 0.5 tokens, ASCII ≈ 0.25 tokens
        return int(japanese_chars * 0.5 + ascii_chars * 0.25)

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._is_loaded

    async def unload(self) -> None:
        """
        Unload the model and free memory

        Override this method if the model needs specific cleanup
        """
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        logger.info(f"{self.model_name} model unloaded")

    async def is_available(self) -> bool:
        """
        Check if this embedding model is available

        Returns:
            True if the model can be used
        """
        try:
            await self.load_model()
            return self._is_loaded
        except Exception as e:
            logger.warning(f"{self.model_name} not available: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.unload()
