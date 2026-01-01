"""
Vector Search Retriever
Semantic search using Milvus vector database
"""

import time
from typing import List, Optional

from backend.core.cache import get_cache_manager
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import RetrievalException
from backend.db.vector import search_similar
from backend.services.embedding import get_embedding_service
from backend.services.retrieval.base import BaseRetriever, RetrieverMixin
from backend.services.retrieval.models import (
    RetrievalOptions,
    RetrievalResult,
    RetrievedChunk,
)

logger = get_logger(__name__)


class VectorRetriever(BaseRetriever, RetrieverMixin):
    """
    Vector-based semantic search using Milvus

    Performs ANN (Approximate Nearest Neighbor) search
    using HNSW index for fast retrieval.
    """

    def __init__(self):
        """Initialize the vector retriever"""
        super().__init__(name="vector")
        self.embedding_service = None
        self.cache_manager = None

    async def initialize(self) -> None:
        """Initialize the vector retriever"""
        await super().initialize()

        # Get embedding service (it's an async function)
        from backend.services.embedding import get_embedding_service as get_emb
        self.embedding_service = await get_emb()

        # Get cache manager
        if settings.EMBEDDING_CACHE_ENABLED:
            self.cache_manager = get_cache_manager()

        logger.info("Vector retriever initialized with embedding service")

    async def retrieve(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using vector similarity search

        Args:
            query: Query text
            options: Retrieval options

        Returns:
            RetrievalResult with list of RetrievedChunk objects

        Raises:
            RetrievalException: If retrieval fails
        """
        start_time = time.time()

        # Default options
        if options is None:
            options = RetrievalOptions()

        try:
            # Step 1: Generate query embedding
            query_embedding = await self._get_query_embedding(query)

            # Step 2: Search Milvus
            chunks_data = await search_similar(
                query_embedding=query_embedding,
                top_k=options.top_k,
                document_ids=options.document_ids,
                min_score=options.min_score,
            )

            # Step 3: Convert to RetrievedChunk objects
            chunks = [
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    text=chunk["text"],
                    score=chunk["score"],
                    metadata=chunk.get("metadata"),
                    source="vector",
                    page_number=chunk.get("page_number"),
                    chunk_index=chunk.get("chunk_index"),
                )
                for chunk in chunks_data
            ]

            # Step 4: Apply filtering and truncation
            chunks = self.filter_by_score(chunks, options.min_score)
            chunks = self.truncate_to_top_k(chunks, options.top_k)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Log results
            self._log_retrieval(query, len(chunks), execution_time_ms)

            # Create result
            metadata = {
                "query_dimension": len(query_embedding),
                "index_type": "HNSW",
                "metric_type": "COSINE",
            }

            return self._create_result(
                chunks=chunks,
                query=query,
                execution_time_ms=execution_time_ms,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            raise RetrievalException(
                message="Vector search failed",
                retriever="vector",
                details={"query": query, "error": str(e)},
            )

    async def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query text

        Args:
            query: Query text

        Returns:
            Query embedding vector

        Raises:
            RetrievalException: If embedding generation fails
        """
        try:
            # Check cache first
            if self.cache_manager:
                cache_key = f"query_embedding:{query}"
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    logger.debug("Query embedding cache hit")
                    return cached

            # Generate embedding
            embedding_result = await self.embedding_service.embed_text(query)
            # Extract the actual vector list from the Embedding object
            embedding = embedding_result.embedding.vector

            # Cache result
            if self.cache_manager:
                cache_key = f"query_embedding:{query}"
                await self.cache_manager.set(
                    cache_key,
                    embedding,
                    ttl=settings.EMBEDDING_CACHE_TTL,
                )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RetrievalException(
                message="Failed to generate query embedding",
                retriever="vector",
                details={"query": query, "error": str(e)},
            )

    async def health_check(self) -> bool:
        """
        Check if the vector retriever is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if embedding service is healthy
            if not await self.embedding_service.health_check():
                return False

            # Check if Milvus is accessible
            from backend.db.vector import health_check as milvus_health_check

            return await milvus_health_check()

        except Exception as e:
            logger.error(f"Vector retriever health check failed: {e}")
            return False
