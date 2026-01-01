"""
Reranking Service
Main service for document reranking with Llama-3.2-NV model

This service provides:
- Single query reranking
- Batch query reranking
- Document relevance scoring
- Threshold-based filtering
"""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.core.cache import cache_manager
from backend.services.reranker.models import (
    RerankerOptions,
    RerankerDocument,
    RerankerResult,
    RerankingOutput,
    RerankingProcessingError,
    RerankingValidationError,
)
from backend.services.reranker.llama_nv import LlamaNVRerankerModel

logger = get_logger(__name__)


class RerankingService:
    """
    Main reranking service for document reranking

    This service manages the Llama-3.2-NV Reranker model and provides
    a high-level API for reranking retrieval results.

    Example:
        ```python
        from backend.services.reranker.service import reranking_service

        # Rerank documents for a query
        results = await reranking_service.rerank(
            query="機械学習とは何ですか？",
            documents=[
                {"text": "機械学習は...", "doc_id": "doc1"},
                {"text": "深層学習は...", "doc_id": "doc2"},
            ]
        )
        ```
    """

    def __init__(
        self,
        model: Optional[LlamaNVRerankerModel] = None,
        options: Optional[RerankerOptions] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the reranking service

        Args:
            model: LlamaNVRerankerModel instance (created automatically if None)
            options: Reranker options
            enable_cache: Whether to enable reranking caching
        """
        self.model = model or LlamaNVRerankerModel(options=options)
        self.options = self.model.options
        self.enable_cache = enable_cache and settings.EMBEDDING_CACHE_ENABLED
        self._is_initialized = False

        logger.info(
            f"RerankingService initialized (cache={self.enable_cache}, "
            f"model={self.model.model_name})"
        )

    async def initialize(self) -> None:
        """
        Initialize the reranking service

        Loads the reranker model into memory. This should be called
        before using the service, or it will be called automatically
        on the first request.
        """
        if self._is_initialized:
            logger.debug("RerankingService already initialized")
            return

        try:
            start_time = time.time()
            await self.model.load_model()
            self._is_initialized = True

            init_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"RerankingService initialized in {init_time}ms "
                f"(model={self.model.model_name})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RerankingService: {e}")
            raise RerankingProcessingError(
                f"Failed to initialize reranking service: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        options: Optional[RerankerOptions] = None,
    ) -> RerankingOutput:
        """
        Rerank documents for a query

        Args:
            query: Query text
            documents: List of document dictionaries with 'text' field
            options: Reranker options (uses service default if None)

        Returns:
            RerankingOutput with reranked and filtered results

        Raises:
            RerankingValidationError: If input validation fails
            RerankingProcessingError: If reranking fails
        """
        start_time = time.time()

        # Use provided options or service default
        rerank_options = options or self.options

        # Validate input
        if not query or not isinstance(query, str):
            raise RerankingValidationError(
                "Query must be a non-empty string",
                details={"query_type": type(query).__name__},
            )

        if not isinstance(documents, list):
            raise RerankingValidationError(
                "Documents must be a list",
                details={"documents_type": type(documents).__name__},
            )

        if not documents:
            raise RerankingValidationError(
                "Documents list cannot be empty",
                details={"query": query},
            )

        # Validate each document has 'text' field
        valid_docs = []
        doc_metadata = []

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Skipping non-document at index {i}")
                continue

            text = doc.get("text", "")
            if not text or not isinstance(text, str):
                logger.warning(f"Skipping document with invalid text at index {i}")
                continue

            valid_docs.append(text.strip())
            doc_metadata.append({
                "doc_id": doc.get("doc_id"),
                "metadata": doc.get("metadata", {}),
                "original_score": doc.get("original_score"),
            })

        if not valid_docs:
            raise RerankingValidationError(
                "No valid documents provided",
                details={"total_documents": len(documents)},
            )

        # Ensure service is initialized
        if not self._is_initialized:
            await self.initialize()

        # Perform reranking
        try:
            # Get reranked indices and scores
            reranked_indices = await self.model.rerank(
                query=query,
                documents=valid_docs,
                threshold=rerank_options.threshold,
            )

            # Limit to top_k_output
            top_indices = reranked_indices[:rerank_options.top_k_output]

            # Build results
            results = []
            for rank, (idx, score) in enumerate(top_indices, start=1):
                metadata = doc_metadata[idx]

                result = RerankerResult(
                    doc_id=metadata["doc_id"],
                    text=valid_docs[idx],
                    score=score,
                    rank=rank,
                    metadata=metadata["metadata"],
                    original_score=metadata["original_score"],
                )
                results.append(result)

            processing_time = (time.time() - start_time) * 1000

            return RerankingOutput(
                results=results,
                query=query,
                total_input=len(valid_docs),
                total_output=len(results),
                processing_time_ms=round(processing_time, 2),
                model=self.model.model_name,
                threshold_applied=rerank_options.threshold,
            )

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RerankingProcessingError(
                f"Failed to rerank documents: {str(e)}",
                details={
                    "query_length": len(query),
                    "num_documents": len(valid_docs),
                    "error_type": type(e).__name__,
                },
            )

    async def rerank_simple(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """
        Simple reranking API for lists of texts

        Args:
            query: Query text
            texts: List of document texts
            top_k: Number of top results to return (uses options default if None)

        Returns:
            List of (original_index, text, score) tuples sorted by relevance
        """
        if not self._is_initialized:
            await self.initialize()

        top_k = top_k or self.options.top_k_output

        # Prepare documents
        documents = [{"text": text} for text in texts]

        # Rerank
        output = await self.rerank(query, documents)

        # Extract results
        results = []
        for result in output.results:
            # Find original index
            original_index = texts.index(result.text)
            results.append((original_index, result.text, result.score))

        return results[:top_k]

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self._is_initialized

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the reranking service

        Returns:
            Dictionary with health status and metrics
        """
        health = {
            "status": "healthy" if self._is_initialized else "uninitialized",
            "model": self.model.model_name,
            "cache_enabled": self.enable_cache,
            "is_loaded": self.model.is_loaded,
        }

        if self._is_initialized:
            # Test reranking
            try:
                start_time = time.time()
                test_result = await self.rerank_simple(
                    query="テスト",
                    texts=["テスト文書1", "テスト文書2", "テスト文書3"],
                    top_k=2,
                )
                health["test_rerank_time_ms"] = round((time.time() - start_time) * 1000, 2)
                health["test_num_results"] = len(test_result)
            except Exception as e:
                health["status"] = "error"
                health["error"] = str(e)

        return health

    async def shutdown(self) -> None:
        """
        Shutdown the reranking service and free resources
        """
        if self._is_initialized:
            await self.model.unload()
            self._is_initialized = False
            logger.info("RerankingService shutdown complete")


# Global singleton instance
_reranking_service: Optional[RerankingService] = None


async def get_reranking_service() -> RerankingService:
    """
    Get or create the global reranking service instance

    Returns:
        RerankingService singleton
    """
    global _reranking_service

    if _reranking_service is None:
        options = RerankerOptions(
            top_k_input=settings.RERANKER_TOP_K_INPUT,
            top_k_output=settings.RERANKER_TOP_K_OUTPUT,
            threshold=settings.RERANKER_THRESHOLD,
            batch_size=32,
        )
        _reranking_service = RerankingService(options=options)
        await _reranking_service.initialize()

    return _reranking_service


# Convenience export
reranking_service: RerankingService = None  # Will be initialized by get_reranking_service()
