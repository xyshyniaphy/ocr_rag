"""
Retrieval Service
Main orchestration service for retrieval operations
"""

from typing import List, Optional

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import RetrievalException
from backend.services.retrieval.base import BaseRetriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.retrieval.keyword import KeywordRetriever
from backend.services.retrieval.models import (
    HybridRetrievalConfig,
    RetrievalOptions,
    RetrievalResult,
    RetrievedChunk,
)
from backend.services.retrieval.vector import VectorRetriever

logger = get_logger(__name__)

# Global retrieval service instance
_retrieval_service: Optional["RetrievalService"] = None


def get_retrieval_service() -> "RetrievalService":
    """
    Get the global retrieval service instance

    Returns:
        RetrievalService instance
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


class RetrievalService:
    """
    Main retrieval service for managing all retrieval operations

    Provides a unified interface for:
    - Vector search (semantic)
    - Keyword search (lexical)
    - Hybrid search (combined)
    """

    def __init__(self):
        """Initialize the retrieval service"""
        self.vector_retriever: Optional[VectorRetriever] = None
        self.keyword_retriever: Optional[KeywordRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._is_initialized = False
        logger.debug("Retrieval service created")

    async def initialize(self) -> None:
        """Initialize the retrieval service and all retrievers"""
        if self._is_initialized:
            logger.debug("Retrieval service already initialized")
            return

        try:
            logger.info("Initializing retrieval service...")

            # Initialize retrievers
            self.vector_retriever = VectorRetriever()
            self.keyword_retriever = KeywordRetriever()
            self.hybrid_retriever = HybridRetriever()

            await self.vector_retriever.initialize()
            await self.keyword_retriever.initialize()
            await self.hybrid_retriever.initialize()

            self._is_initialized = True
            logger.info("Retrieval service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize retrieval service: {e}")
            raise RetrievalException(
                message="Failed to initialize retrieval service",
                details={"error": str(e)},
            )

    async def shutdown(self) -> None:
        """Shutdown the retrieval service and all retrievers"""
        try:
            if self.vector_retriever:
                await self.vector_retriever.shutdown()
            if self.keyword_retriever:
                await self.keyword_retriever.shutdown()
            if self.hybrid_retriever:
                await self.hybrid_retriever.shutdown()

            self._is_initialized = False
            logger.info("Retrieval service shutdown successfully")

        except Exception as e:
            logger.error(f"Error during retrieval service shutdown: {e}")

    async def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        options: Optional[RetrievalOptions] = None,
        hybrid_config: Optional[HybridRetrievalConfig] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for the given query

        Args:
            query: Query text
            method: Retrieval method - "vector", "keyword", or "hybrid" (default)
            options: Retrieval options (top_k, min_score, etc.)
            hybrid_config: Hybrid retrieval config (only for method="hybrid")

        Returns:
            RetrievalResult with list of RetrievedChunk objects

        Raises:
            RetrievalException: If retrieval fails

        Example:
            # Hybrid search (default)
            result = await service.retrieve("What is the main topic?")

            # Vector search only
            result = await service.retrieve("Query", method="vector")

            # Keyword search only
            result = await service.retrieve("Query", method="keyword")

            # With custom options
            options = RetrievalOptions(top_k=20, min_score=0.5)
            result = await service.retrieve("Query", options=options)
        """
        # Ensure initialized
        if not self._is_initialized:
            await self.initialize()

        # Validate query
        if not query or not query.strip():
            raise RetrievalException(
                message="Query cannot be empty",
                details={"query": query},
            )

        # Route to appropriate retriever
        retriever = self._get_retriever(method)

        # Execute retrieval
        if method == "hybrid" and hybrid_config:
            result = await retriever.retrieve(query, options, hybrid_config)
        else:
            result = await retriever.retrieve(query, options)

        return result

    async def retrieve_batch(
        self,
        queries: List[str],
        method: str = "hybrid",
        options: Optional[RetrievalOptions] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for multiple queries

        Args:
            queries: List of query texts
            method: Retrieval method - "vector", "keyword", or "hybrid"
            options: Retrieval options

        Returns:
            List of RetrievalResult objects (one per query)

        Raises:
            RetrievalException: If any retrieval fails
        """
        import asyncio

        # Ensure initialized
        if not self._is_initialized:
            await self.initialize()

        # Execute retrievals in parallel
        tasks = [
            self.retrieve(query, method, options) for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
                # Create empty result for failed query
                final_results.append(
                    RetrievalResult(
                        chunks=[],
                        total_results=0,
                        query=queries[i],
                        retrieval_method=method,
                        execution_time_ms=0,
                        metadata={"error": str(result)},
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def _get_retriever(self, method: str) -> BaseRetriever:
        """
        Get the appropriate retriever for the given method

        Args:
            method: Retrieval method name

        Returns:
            BaseRetriever instance

        Raises:
            RetrievalException: If method is invalid
        """
        if method == "vector":
            if not self.vector_retriever:
                raise RetrievalException(
                    message="Vector retriever not initialized"
                )
            return self.vector_retriever

        elif method == "keyword":
            if not self.keyword_retriever:
                raise RetrievalException(
                    message="Keyword retriever not initialized"
                )
            return self.keyword_retriever

        elif method == "hybrid":
            if not self.hybrid_retriever:
                raise RetrievalException(
                    message="Hybrid retriever not initialized"
                )
            return self.hybrid_retriever

        else:
            raise RetrievalException(
                message=f"Invalid retrieval method: {method}",
                details={"valid_methods": ["vector", "keyword", "hybrid"]},
            )

    async def health_check(self) -> dict:
        """
        Check health of all retrievers

        Returns:
            Dictionary with health status of each retriever
        """
        if not self._is_initialized:
            return {
                "status": "uninitialized",
                "vector": False,
                "keyword": False,
                "hybrid": False,
            }

        checks = {
            "status": "initialized",
            "vector": False,
            "keyword": False,
            "hybrid": False,
        }

        if self.vector_retriever:
            checks["vector"] = await self.vector_retriever.health_check()

        if self.keyword_retriever:
            checks["keyword"] = await self.keyword_retriever.health_check()

        if self.hybrid_retriever:
            checks["hybrid"] = await self.hybrid_retriever.health_check()

        return checks

    async def get_stats(self) -> dict:
        """
        Get retrieval service statistics

        Returns:
            Dictionary with service statistics
        """
        from backend.db.vector import get_collection_stats

        stats = {
            "initialized": self._is_initialized,
            "retrievers": {
                "vector": self.vector_retriever is not None,
                "keyword": self.keyword_retriever is not None,
                "hybrid": self.hybrid_retriever is not None,
            },
        }

        # Get Milvus stats
        milvus_stats = await get_collection_stats()
        stats["milvus"] = milvus_stats

        return stats


# Convenience functions
async def retrieve(
    query: str,
    method: str = "hybrid",
    options: Optional[RetrievalOptions] = None,
) -> RetrievalResult:
    """
    Convenience function for retrieval

    Args:
        query: Query text
        method: Retrieval method
        options: Retrieval options

    Returns:
        RetrievalResult
    """
    service = get_retrieval_service()
    return await service.retrieve(query, method, options)


async def health_check() -> dict:
    """
    Convenience function for health check

    Returns:
        Health status dictionary
    """
    service = get_retrieval_service()
    return await service.health_check()
