"""
Base Retrieval Interface
Abstract base class for retrieval implementations
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from backend.core.logging import get_logger
from backend.core.exceptions import RetrievalException
from backend.services.retrieval.models import (
    RetrievalOptions,
    RetrievalResult,
    RetrievedChunk,
)

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval implementations

    All retrievers must implement the retrieve() method.
    Subclasses can override health_check() for custom health checks.
    """

    def __init__(self, name: str):
        """
        Initialize the retriever

        Args:
            name: Retriever name (e.g., "vector", "keyword", "hybrid")
        """
        self.name = name
        self._is_initialized = False
        logger.debug(f"{self.name} retriever initialized")

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for the given query

        Args:
            query: Query text
            options: Retrieval options (top_k, min_score, etc.)

        Returns:
            RetrievalResult with list of RetrievedChunk objects

        Raises:
            RetrievalException: If retrieval fails
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the retriever is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Default implementation: check if initialized
            return self._is_initialized
        except Exception as e:
            logger.error(f"{self.name} health check failed: {e}")
            return False

    async def initialize(self) -> None:
        """
        Initialize the retriever (load models, connect to databases, etc.)

        Subclasses should override this method if initialization is needed.
        """
        self._is_initialized = True
        logger.info(f"{self.name} retriever initialized")

    async def shutdown(self) -> None:
        """
        Shutdown the retriever (cleanup resources)

        Subclasses should override this method if cleanup is needed.
        """
        self._is_initialized = False
        logger.info(f"{self.name} retriever shutdown")

    def _create_result(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        execution_time_ms: float,
        metadata: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Create a RetrievalResult object

        Args:
            chunks: List of retrieved chunks
            query: Original query text
            execution_time_ms: Execution time in milliseconds
            metadata: Optional metadata

        Returns:
            RetrievalResult object
        """
        return RetrievalResult(
            chunks=chunks,
            total_results=len(chunks),
            query=query,
            retrieval_method=self.name,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    def _log_retrieval(
        self,
        query: str,
        num_results: int,
        execution_time_ms: float,
    ) -> None:
        """
        Log retrieval results

        Args:
            query: Query text
            num_results: Number of results returned
            execution_time_ms: Execution time
        """
        logger.debug(
            f"{self.name} retrieval: query='{query[:50]}...', "
            f"results={num_results}, time={execution_time_ms:.2f}ms"
        )


class RetrieverMixin:
    """
    Mixin class with common retrieval utilities
    """

    @staticmethod
    def normalize_scores(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Normalize scores to [0, 1] range

        Args:
            chunks: List of chunks with scores

        Returns:
            List of chunks with normalized scores
        """
        if not chunks:
            return chunks

        # Find min and max scores
        scores = [chunk.score for chunk in chunks]
        min_score = min(scores)
        max_score = max(scores)

        # Normalize
        if max_score == min_score:
            # All scores are the same
            for chunk in chunks:
                chunk.score = 1.0
        else:
            for chunk in chunks:
                chunk.score = (chunk.score - min_score) / (max_score - min_score)

        return chunks

    @staticmethod
    def filter_by_score(
        chunks: List[RetrievedChunk],
        min_score: float,
    ) -> List[RetrievedChunk]:
        """
        Filter chunks by minimum score

        Args:
            chunks: List of chunks
            min_score: Minimum score threshold

        Returns:
            Filtered list of chunks
        """
        return [chunk for chunk in chunks if chunk.score >= min_score]

    @staticmethod
    def truncate_to_top_k(
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Truncate chunks to top K results

        Args:
            chunks: List of chunks (should already be sorted)
            top_k: Maximum number of chunks to return

        Returns:
            Truncated list of chunks
        """
        return chunks[:top_k]
