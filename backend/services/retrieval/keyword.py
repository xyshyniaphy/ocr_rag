"""
Keyword Search Retriever
Full-text search using PostgreSQL with pg_trgm
"""

import time
from typing import List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import RetrievalException
from backend.db.session import get_db_session
from backend.models.chunk import Chunk
from backend.services.retrieval.base import BaseRetriever, RetrieverMixin
from backend.services.retrieval.models import (
    RetrievalOptions,
    RetrievalResult,
    RetrievedChunk,
)

logger = get_logger(__name__)


class KeywordRetriever(BaseRetriever, RetrieverMixin):
    """
    Keyword-based search using PostgreSQL full-text search

    Uses pg_trgm extension for language-agnostic trigram matching.
    Works well for Japanese text without requiring tokenization.
    """

    def __init__(self):
        """Initialize the keyword retriever"""
        super().__init__(name="keyword")

    async def initialize(self) -> None:
        """Initialize the keyword retriever"""
        await super().initialize()
        await self._ensure_pg_trgm_extension()
        logger.info("Keyword retriever initialized with pg_trgm extension")

    async def retrieve(
        self,
        query: str,
        options: Optional[RetrievalOptions] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using keyword search

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
            # Query PostgreSQL using trigram similarity
            chunks = await self._search_keyword(
                query=query,
                top_k=options.top_k,
                document_ids=options.document_ids,
            )

            # Apply filtering
            chunks = self.filter_by_score(chunks, options.min_score)
            chunks = self.truncate_to_top_k(chunks, options.top_k)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Log results
            self._log_retrieval(query, len(chunks), execution_time_ms)

            # Create result
            metadata = {
                "search_method": "pg_trgm",
                "query_length": len(query),
            }

            return self._create_result(
                chunks=chunks,
                query=query,
                execution_time_ms=execution_time_ms,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            raise RetrievalException(
                message="Keyword search failed",
                retriever="keyword",
                details={"query": query, "error": str(e)},
            )

    async def _search_keyword(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Search chunks by keyword using pg_trgm

        Args:
            query: Query text
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of RetrievedChunk objects with similarity scores

        Raises:
            RetrievalException: If search fails
        """
        # get_db_session() is an async generator, iterate over it
        async for session in get_db_session():
            try:
                # Build SQL query with trigram similarity
                # Use similarity() function from pg_trgm
                sql_query = text("""
                    SELECT
                        c.milvus_id as chunk_id,
                        c.document_id,
                        c.text_content as text,
                        c.page_number,
                        c.chunk_index,
                        SIMILARITY(c.text_content, :query) as score
                    FROM chunks c
                    WHERE SIMILARITY(c.text_content, :query) > 0.1
                    AND c.embedding_created_at IS NOT NULL
                """)

                # Add document ID filter if specified
                params = {"query": query}
                if document_ids:
                    # Need to construct the WHERE clause dynamically
                    doc_filter = " AND c.document_id = ANY(:document_ids)"
                    sql_query = text(sql_query.text + doc_filter)
                    params["document_ids"] = document_ids

                # Add ORDER BY and LIMIT
                sql_query = text(sql_query.text + f"""
                    ORDER BY score DESC
                    LIMIT :top_k
                """)
                params["top_k"] = top_k

                # Execute query
                result = await session.execute(sql_query, params)
                rows = result.fetchall()

                # Convert to RetrievedChunk objects
                chunks = [
                    RetrievedChunk(
                        chunk_id=row.chunk_id,
                        document_id=str(row.document_id),
                        text=row.text,
                        score=float(row.score),
                        metadata={"page": row.page_number},
                        source="keyword",
                        page_number=row.page_number,
                        chunk_index=row.chunk_index,
                    )
                    for row in rows
                ]

                logger.debug(f"Keyword search found {len(chunks)} chunks")
                return chunks

            except Exception as e:
                logger.error(f"Keyword search query failed: {e}")
                raise RetrievalException(
                    message="Database query failed",
                    retriever="keyword",
                    details={"query": query, "error": str(e)},
                )

    async def _ensure_pg_trgm_extension(self) -> None:
        """
        Ensure pg_trgm extension is installed in PostgreSQL

        Raises:
            RetrievalException: If extension installation fails
        """
        # get_db_session() is an async generator, iterate over it
        async for session in get_db_session():
            try:
                # Check if extension exists
                check_query = text("""
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'
                """)
                result = await session.execute(check_query)
                exists = result.scalar()

                if not exists:
                    logger.info("Installing pg_trgm extension...")
                    # Create extension
                    await session.execute(
                        text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                    )
                    await session.commit()
                    logger.info("pg_trgm extension installed successfully")
                else:
                    logger.debug("pg_trgm extension already installed")

            except Exception as e:
                logger.warning(f"Failed to ensure pg_trgm extension: {e}")
                # Don't fail - keyword search may still work
                # but with reduced performance

    async def health_check(self) -> bool:
        """
        Check if the keyword retriever is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if PostgreSQL is accessible
            async for session in get_db_session():
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Keyword retriever health check failed: {e}")
            return False
