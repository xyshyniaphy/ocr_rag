"""
RAG Orchestration Pipeline
Main service for coordinating retrieval, reranking, and LLM generation

This service provides:
- End-to-end RAG query processing
- Stage-by-stage timing and metrics
- Error handling and recovery
- Query result caching
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.core.cache import cache_manager
from backend.core.exceptions import RAGException
from backend.services.rag.models import (
    RAGQueryOptions,
    RAGResult,
    RAGSource,
    RAGStageMetrics,
    RAGPipelineConfig,
    RAGValidationError,
    RAGProcessingError,
    RAGServiceError,
)

logger = get_logger(__name__)

# Global RAG service instance
_rag_service: Optional["RAGService"] = None


def get_rag_service() -> "RAGService":
    """
    Get the global RAG service instance

    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# Default Japanese RAG prompt
DEFAULT_RAG_SYSTEM_PROMPT = """あなたは提供されたコンテキストに基づいて質問に答えるAIアシスタントです。

# コンテキスト
{context}

# 指示
- 上記のコンテキストのみを使用して質問に答えてください
- コンテキストに情報が含まれていない場合は、「コンテキストに情報がありません」と答えてください
- 答えを简潔かつ正確にしてください
- 必要に応じてコンテキストを引用してください
- 日本語で答えてください

# 質問
{query}"""


class RAGService:
    """
    Main RAG orchestration service

    This service coordinates the complete RAG pipeline:
    1. Query understanding and validation
    2. Retrieval (vector + keyword hybrid search)
    3. Reranking (optional cross-encoder reranking)
    4. Context assembly
    5. LLM generation with retrieved contexts

    Example:
        ```python
        from backend.services.rag import get_rag_service

        service = await get_rag_service()

        result = await service.query(
            query="機械学習と深層学習の違いは何ですか？",
            options=RAGQueryOptions(top_k=10, rerank=True),
        )

        print(f"Answer: {result.answer}")
        print(f"Sources: {len(result.sources)} documents")
        ```
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """
        Initialize the RAG service

        Args:
            config: RAG pipeline configuration (uses defaults if None)
        """
        self.config = config or RAGPipelineConfig()
        self._retrieval_service = None
        self._reranking_service = None
        self._llm_service = None
        self._is_initialized = False

        logger.info(
            f"RAGService created (reranking={self.config.enable_reranking}, "
            f"cache={self.config.enable_cache})"
        )

    async def initialize(self) -> None:
        """
        Initialize the RAG service and all dependent services

        This should be called before using the service, or it will be
        called automatically on the first request.
        """
        if self._is_initialized:
            logger.debug("RAGService already initialized")
            return

        try:
            logger.info("Initializing RAG service...")

            # Import services here to avoid circular imports
            from backend.services.retrieval import get_retrieval_service
            from backend.services.reranker.service import get_reranking_service
            from backend.services.llm import get_llm_service

            # Initialize services
            self._retrieval_service = get_retrieval_service()
            await self._retrieval_service.initialize()

            if self.config.enable_reranking:
                self._reranking_service = await get_reranking_service()

            self._llm_service = await get_llm_service()

            self._is_initialized = True
            logger.info("RAGService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            raise RAGServiceError(
                f"Failed to initialize RAG service: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    async def shutdown(self) -> None:
        """Shutdown the RAG service and all dependent services"""
        try:
            if self._retrieval_service:
                await self._retrieval_service.shutdown()
            if self._reranking_service:
                await self._reranking_service.shutdown()
            if self._llm_service:
                await self._llm_service.shutdown()

            self._is_initialized = False
            logger.info("RAGService shutdown successfully")

        except Exception as e:
            logger.error(f"Error during RAGService shutdown: {e}")

    async def query(
        self,
        query: str,
        options: Optional[RAGQueryOptions] = None,
        user_id: Optional[str] = None,
    ) -> RAGResult:
        """
        Process a RAG query through the complete pipeline

        Args:
            query: User query text
            options: Query options (uses defaults if None)
            user_id: Optional user ID for logging/permissions

        Returns:
            RAGResult with answer and sources

        Raises:
            RAGValidationError: If query validation fails
            RAGProcessingError: If pipeline processing fails
        """
        # Initialize on first use
        if not self._is_initialized:
            await self.initialize()

        # Use provided options or defaults
        opts = options or RAGQueryOptions()

        # Start timing
        start_time = time.time()
        stage_timings: List[RAGStageMetrics] = []
        query_id = str(uuid.uuid4())

        # Validate query
        try:
            if not query or not query.strip():
                raise RAGValidationError("Query cannot be empty")

            query = query.strip()
            if len(query) > 500:
                raise RAGValidationError("Query too long (max 500 characters)")

            logger.info(f"Processing RAG query: {query[:100]}... (id={query_id})")

        except RAGValidationError:
            raise
        except Exception as e:
            raise RAGValidationError(f"Query validation failed: {str(e)}")

        # Stage 1: Query Understanding
        stage_start = time.time()
        try:
            normalized_query = self._normalize_query(query, opts.language)
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="query_understanding",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=True,
                    metadata={"original_length": len(query), "normalized_length": len(normalized_query)},
                )
            )
        except Exception as e:
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="query_understanding",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=False,
                    error=str(e),
                )
            )
            raise RAGProcessingError("Query understanding failed", stage="query_understanding")

        # Stage 2: Retrieval
        retrieved_chunks = []
        stage_start = time.time()
        try:
            retrieval_result = await self._retrieve(
                normalized_query,
                top_k=opts.retrieval_top_k,
                method=opts.retrieval_method,
                document_ids=opts.document_ids,
                min_score=opts.min_score,
                use_cache=opts.use_cache,
            )
            retrieved_chunks = retrieval_result.chunks
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="retrieval",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=True,
                    metadata={
                        "retrieved_count": len(retrieved_chunks),
                        "method": opts.retrieval_method,
                    },
                )
            )
        except Exception as e:
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="retrieval",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=False,
                    error=str(e),
                )
            )
            raise RAGProcessingError("Retrieval failed", stage="retrieval", details={"error": str(e)})

        # Stage 3: Reranking (optional)
        reranked_chunks = retrieved_chunks
        if opts.rerank and self.config.enable_reranking and self._reranking_service:
            stage_start = time.time()
            try:
                reranked_chunks = await self._rerank(
                    normalized_query,
                    retrieved_chunks,
                    top_k=opts.rerank_top_k,
                )
                stage_timings.append(
                    RAGStageMetrics(
                        stage_name="reranking",
                        duration_ms=(time.time() - stage_start) * 1000,
                        success=True,
                        metadata={
                            "input_count": len(retrieved_chunks),
                            "output_count": len(reranked_chunks),
                        },
                    )
                )
            except Exception as e:
                stage_timings.append(
                    RAGStageMetrics(
                        stage_name="reranking",
                        duration_ms=(time.time() - stage_start) * 1000,
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning(f"Reranking failed, using original results: {e}")
                # Continue with original results
                reranked_chunks = retrieved_chunks

        # Limit to top_k results
        reranked_chunks = reranked_chunks[: opts.top_k]

        # Stage 4: Context Assembly
        stage_start = time.time()
        try:
            contexts = self._assemble_contexts(reranked_chunks, opts.top_k)
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="context_assembly",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=True,
                    metadata={"context_count": len(contexts)},
                )
            )
        except Exception as e:
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="context_assembly",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=False,
                    error=str(e),
                )
            )
            raise RAGProcessingError("Context assembly failed", stage="context_assembly")

        # Stage 5: LLM Generation
        stage_start = time.time()
        llm_response = None
        try:
            llm_response = await self._generate(
                normalized_query,
                contexts,
                system_prompt=self.config.system_prompt,
            )
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="llm_generation",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=True,
                    metadata={
                        "llm_model": llm_response.model,
                        "tokens": llm_response.total_tokens,
                    },
                )
            )
        except Exception as e:
            stage_timings.append(
                RAGStageMetrics(
                    stage_name="llm_generation",
                    duration_ms=(time.time() - stage_start) * 1000,
                    success=False,
                    error=str(e),
                )
            )
            raise RAGProcessingError("LLM generation failed", stage="llm_generation", details={"error": str(e)})

        # Build sources
        sources = self._build_sources(reranked_chunks)

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000

        # Calculate confidence (average of top scores)
        confidence = None
        if sources:
            confidence = sum(s.score for s in sources[:3]) / min(3, len(sources))

        # Build result
        result = RAGResult(
            query=query,
            answer=llm_response.content,
            sources=sources if opts.include_sources else [],
            query_id=query_id,
            processing_time_ms=round(total_time_ms, 2),
            stage_timings=stage_timings,
            confidence=round(confidence, 3) if confidence else None,
            llm_model=llm_response.model,
            embedding_model=settings.EMBEDDING_MODEL,
            reranker_model=settings.RERANKER_MODEL_NAME if opts.rerank else None,
            metadata={
                "user_id": user_id,
                "options": opts.model_dump(),
                "total_sources": len(sources),
            },
        )

        logger.info(
            f"RAG query completed in {total_time_ms:.0f}ms "
            f"(id={query_id}, sources={len(sources)})"
        )

        return result

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        method: str,
        document_ids: Optional[Sequence[str]],
        min_score: float,
        use_cache: bool,
    ) -> Any:
        """Perform retrieval using the retrieval service"""
        from backend.services.retrieval.models import RetrievalOptions

        options = RetrievalOptions(
            top_k=top_k,
            min_score=min_score,
            document_ids=list(document_ids) if document_ids else None,
            include_metadata=True,
            use_cache=use_cache,
        )

        if method == "hybrid":
            return await self._retrieval_service.hybrid_retrieve(query, options)
        elif method == "vector":
            return await self._retrieval_service.vector_retrieve(query, options)
        elif method == "keyword":
            return await self._retrieval_service.keyword_retrieve(query, options)
        else:
            raise RAGValidationError(f"Invalid retrieval method: {method}")

    async def _rerank(
        self,
        query: str,
        chunks: List[Any],
        top_k: int,
    ) -> List[Any]:
        """Rerank chunks using the reranking service"""
        from backend.services.reranker.models import RerankerDocument

        # Prepare documents for reranking
        documents = [
            RerankerDocument(
                text=chunk.text,
                doc_id=chunk.chunk_id,
            )
            for chunk in chunks
        ]

        # Rerank
        rerank_result = await self._reranking_service.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        # Update chunks with rerank scores
        reranked_chunks = []
        for result in rerank_result.results:
            # Find original chunk
            original_chunk = next((c for c in chunks if c.chunk_id == result.doc_id), None)
            if original_chunk:
                # Create a new chunk with updated score
                import copy
                updated_chunk = copy.copy(original_chunk)
                updated_chunk.score = result.score
                reranked_chunks.append(updated_chunk)

        return reranked_chunks

    def _assemble_contexts(self, chunks: List[Any], max_contexts: int) -> List[Any]:
        """Assemble contexts from chunks for LLM generation"""
        from backend.services.llm.models import RAGContext

        contexts = [
            RAGContext(
                text=chunk.text,
                doc_id=chunk.chunk_id,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks[:max_contexts]
        ]

        return contexts

    async def _generate(
        self,
        query: str,
        contexts: List[Any],
        system_prompt: Optional[str],
    ) -> Any:
        """Generate answer using LLM service"""
        return await self._llm_service.generate_rag(
            query=query,
            contexts=contexts,
            system_prompt=system_prompt,
        )

    def _build_sources(self, chunks: List[Any]) -> List[RAGSource]:
        """Build RAGSource objects from chunks"""
        sources = []
        for chunk in chunks:
            source = RAGSource(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.metadata.get("document_title") if chunk.metadata else None,
                text=chunk.text,
                score=chunk.score,
                rerank_score=chunk.metadata.get("rerank_score") if chunk.metadata else None,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
            sources.append(source)

        return sources

    def _normalize_query(self, query: str, language: str) -> str:
        """Normalize query text"""
        import unicodedata

        # Unicode normalization
        normalized = unicodedata.normalize("NFKC", query)

        # Trim whitespace
        normalized = normalized.strip()

        return normalized

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RAG service

        Returns:
            Dictionary with health status
        """
        health = {
            "status": "unknown",
            "service": "rag",
            "initialized": self._is_initialized,
            "components": {},
            "errors": [],
        }

        try:
            if not self._is_initialized:
                health["status"] = "not_initialized"
                return health

            # Check retrieval service
            if self._retrieval_service:
                try:
                    retrieval_health = await self._retrieval_service.health_check()
                    health["components"]["retrieval"] = retrieval_health.get("status", "unknown")
                except Exception as e:
                    health["components"]["retrieval"] = "error"
                    health["errors"].append(f"Retrieval service: {str(e)}")

            # Check reranking service
            if self._reranking_service:
                try:
                    reranker_health = await self._reranking_service.health_check()
                    health["components"]["reranking"] = reranker_health.get("status", "unknown")
                except Exception as e:
                    health["components"]["reranking"] = "error"
                    health["errors"].append(f"Reranking service: {str(e)}")

            # Check LLM service
            if self._llm_service:
                try:
                    llm_health = await self._llm_service.health_check()
                    health["components"]["llm"] = llm_health.get("status", "unknown")
                except Exception as e:
                    health["components"]["llm"] = "error"
                    health["errors"].append(f"LLM service: {str(e)}")

            # Overall status
            if health["errors"]:
                health["status"] = "degraded"
            else:
                health["status"] = "healthy"

        except Exception as e:
            health["status"] = "error"
            health["errors"].append(str(e))

        return health


# Convenience function for direct querying
async def query_rag(
    query: str,
    options: Optional[RAGQueryOptions] = None,
    user_id: Optional[str] = None,
) -> RAGResult:
    """
    Convenience function for RAG queries

    Args:
        query: User query text
        options: Query options
        user_id: Optional user ID

    Returns:
        RAGResult with answer and sources

    Example:
        ```python
        from backend.services.rag import query_rag

        result = await query_rag(
            query="機械学習とは何ですか？",
            options=RAGQueryOptions(top_k=5),
        )
        ```
    """
    service = get_rag_service()
    return await service.query(query, options, user_id)
