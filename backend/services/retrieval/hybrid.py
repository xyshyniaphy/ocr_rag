"""
Hybrid Search Retriever
Combines vector and keyword search using Reciprocal Rank Fusion (RRF)
"""

import time
from typing import Dict, List, Optional

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import RetrievalException
from backend.services.retrieval.base import BaseRetriever, RetrieverMixin
from backend.services.retrieval.keyword import KeywordRetriever
from backend.services.retrieval.models import (
    HybridRetrievalConfig,
    RetrievalOptions,
    RetrievalResult,
    RetrievedChunk,
)
from backend.services.retrieval.vector import VectorRetriever

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever, RetrieverMixin):
    """
    Hybrid search combining vector and keyword retrieval

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both semantic (vector) and lexical (keyword) search methods.

    RRF Formula:
        score(d) = sum(weight / (k + rank(d))) for each retriever

    Where:
        - k is a constant (default: 60)
        - rank(d) is the rank of document d in that retriever's results
        - weight is the importance weight for that retriever
    """

    def __init__(self):
        """Initialize the hybrid retriever"""
        super().__init__(name="hybrid")
        self.vector_retriever: Optional[VectorRetriever] = None
        self.keyword_retriever: Optional[KeywordRetriever] = None
        self.default_config = HybridRetrievalConfig(
            vector_weight=settings.HYBRID_SEARCH_VECTOR_WEIGHT,
            keyword_weight=settings.HYBRID_SEARCH_KEYWORD_WEIGHT,
        )

    async def initialize(self) -> None:
        """Initialize the hybrid retriever"""
        await super().initialize()

        # Initialize sub-retrievers
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()

        await self.vector_retriever.initialize()
        await self.keyword_retriever.initialize()

        logger.info("Hybrid retriever initialized with vector and keyword retrievers")

    async def retrieve(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
        hybrid_config: Optional[HybridRetrievalConfig] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using hybrid search

        Args:
            query: Query text
            options: Retrieval options (top_k, min_score, etc.)
            hybrid_config: Hybrid retrieval config (weights, rrf_k, etc.)

        Returns:
            RetrievalResult with list of RetrievedChunk objects

        Raises:
            RetrievalException: If retrieval fails
        """
        start_time = time.time()

        # Use default config if not provided
        if hybrid_config is None:
            hybrid_config = self.default_config

        # Default options
        if options is None:
            options = RetrievalOptions(top_k=hybrid_config.top_k)

        try:
            # Step 1: Execute both retrievers in parallel
            vector_results, keyword_results = await self._parallel_retrieve(
                query, options
            )

            # Step 2: Combine results using RRF
            combined_chunks = self._rrf_combine(
                vector_results=vector_results,
                keyword_results=keyword_results,
                vector_weight=hybrid_config.vector_weight,
                keyword_weight=hybrid_config.keyword_weight,
                rrf_k=hybrid_config.rrf_k,
            )

            # Step 3: Apply filtering and truncation
            combined_chunks = self.filter_by_score(combined_chunks, hybrid_config.min_score)
            combined_chunks = self.truncate_to_top_k(combined_chunks, hybrid_config.top_k)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Log results
            self._log_retrieval(query, len(combined_chunks), execution_time_ms)

            # Create result with metadata
            metadata = {
                "vector_count": len(vector_results),
                "keyword_count": len(keyword_results),
                "rrf_k": hybrid_config.rrf_k,
                "vector_weight": hybrid_config.vector_weight,
                "keyword_weight": hybrid_config.keyword_weight,
            }

            return self._create_result(
                chunks=combined_chunks,
                query=query,
                execution_time_ms=execution_time_ms,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RetrievalException(
                message="Hybrid search failed",
                retriever="hybrid",
                details={"query": query, "error": str(e)},
            )

    async def _parallel_retrieve(
        self,
        query: str,
        options: RetrievalOptions,
    ) -> tuple[List[RetrievedChunk], List[RetrievedChunk]]:
        """
        Execute vector and keyword retrieval in parallel

        Args:
            query: Query text
            options: Retrieval options

        Returns:
            Tuple of (vector_results, keyword_results)
        """
        import asyncio

        # Retrieve from both sources in parallel
        vector_task = self.vector_retriever.retrieve(query, options)
        keyword_task = self.keyword_retriever.retrieve(query, options)

        # Wait for both to complete
        vector_result, keyword_result = await asyncio.gather(
            vector_task,
            keyword_task,
            return_exceptions=True,
        )

        # Handle errors
        if isinstance(vector_result, Exception):
            logger.warning(f"Vector retrieval failed: {vector_result}")
            vector_result = RetrievalResult(
                chunks=[],
                total_results=0,
                query=query,
                retrieval_method="vector",
                execution_time_ms=0,
            )

        if isinstance(keyword_result, Exception):
            logger.warning(f"Keyword retrieval failed: {keyword_result}")
            keyword_result = RetrievalResult(
                chunks=[],
                total_results=0,
                query=query,
                retrieval_method="keyword",
                execution_time_ms=0,
            )

        return vector_result.chunks, keyword_result.chunks

    def _rrf_combine(
        self,
        vector_results: List[RetrievedChunk],
        keyword_results: List[RetrievedChunk],
        vector_weight: float,
        keyword_weight: float,
        rrf_k: int,
    ) -> List[RetrievedChunk]:
        """
        Combine results using Reciprocal Rank Fusion

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector results
            keyword_weight: Weight for keyword results
            rrf_k: RRF constant

        Returns:
            Combined and re-ranked list of chunks
        """
        # Dictionary to store accumulated scores
        chunk_scores: Dict[str, RetrievedChunk] = {}

        # Add vector search results
        for rank, chunk in enumerate(vector_results, start=1):
            chunk_id = chunk.chunk_id
            rrf_score = vector_weight / (rrf_k + rank)

            if chunk_id in chunk_scores:
                # Accumulate score if already exists
                chunk_scores[chunk_id].score += rrf_score
            else:
                # Create new entry
                chunk_scores[chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=rrf_score,
                    metadata=chunk.metadata,
                    source="hybrid",
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                )

        # Add keyword search results
        for rank, chunk in enumerate(keyword_results, start=1):
            chunk_id = chunk.chunk_id
            rrf_score = keyword_weight / (rrf_k + rank)

            if chunk_id in chunk_scores:
                # Accumulate score if already exists
                chunk_scores[chunk_id].score += rrf_score
            else:
                # Create new entry
                chunk_scores[chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=rrf_score,
                    metadata=chunk.metadata,
                    source="hybrid",
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                )

        # Sort by score (descending)
        combined = list(chunk_scores.values())
        combined.sort(key=lambda x: x.score, reverse=True)

        # Normalize scores to [0, 1] range
        if combined and combined[0].score > 0:
            max_score = combined[0].score
            combined = [
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    text=c.text,
                    score=c.score / max_score,  # Normalize
                    metadata=c.metadata,
                    source="hybrid",
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                )
                for c in combined
            ]

        logger.debug(
            f"RRF combined {len(vector_results)} vector + {len(keyword_results)} keyword "
            f"-> {len(combined)} unique chunks (normalized)"
        )

        return combined

    async def health_check(self) -> bool:
        """
        Check if the hybrid retriever is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check both sub-retrievers
            vector_healthy = await self.vector_retriever.health_check()
            keyword_healthy = await self.keyword_retriever.health_check()

            # At least one should be healthy
            return vector_healthy or keyword_healthy

        except Exception as e:
            logger.error(f"Hybrid retriever health check failed: {e}")
            return False
