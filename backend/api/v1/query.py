"""
Query API Routes
RAG query endpoints and search
"""

import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger
from backend.core.exceptions import ValidationException, RAGException
from backend.db.session import get_db_session
from backend.db.models import Document as DocumentModel
from backend.models.query import (
    QueryRequest,
    QueryResponse,
    SourceReference,
    SearchRequest,
    SearchResponse,
)
from backend.services.rag import (
    get_rag_service,
    RAGQueryOptions,
    RAGValidationError,
    RAGProcessingError,
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Submit a query to the RAG system

    - **query**: User query text (1-500 characters)
    - **document_ids**: Optional filter by specific documents
    - **top_k**: Number of sources to retrieve (1-20)
    - **include_sources**: Include source chunks in response
    - **language**: Query language (default: ja)
    - **stream**: Stream response tokens (not implemented yet)
    - **rerank**: Apply reranker (default: true)
    """
    try:
        # Build RAG query options
        options = RAGQueryOptions(
            top_k=request.top_k,
            retrieval_top_k=min(request.top_k * 2, 50),  # Retrieve more for reranking
            rerank_top_k=request.top_k,
            rerank=request.rerank,
            retrieval_method="hybrid",
            document_ids=request.document_ids,
            include_sources=request.include_sources,
            use_cache=True,
            language=request.language,
        )

        # Get RAG service and process query
        rag_service = await get_rag_service()
        rag_result = await rag_service.query(
            query=request.query,
            options=options,
            user_id=None,  # TODO: Get from authenticated user
        )

        # Convert RAG sources to SourceReference format
        sources = [
            SourceReference(
                document_id=str(source.document_id),
                document_title=source.document_title or "Unknown Document",
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                chunk_text=source.text,
                relevance_score=source.score,
            )
            for source in rag_result.sources
        ]

        # Build stage timings dict from RAGStageMetrics
        stage_timings = {
            stage.stage_name: int(stage.duration_ms)
            for stage in rag_result.stage_timings
        }

        # Create query record for database
        from backend.db.models import Query as QueryModel

        query_record = QueryModel(
            id=uuid.UUID(rag_result.query_id) if rag_result.query_id else uuid.uuid4(),
            user_id=None,  # TODO: Get from authenticated user
            query_text=request.query,
            query_language=request.language,
            query_type="hybrid",
            top_k=request.top_k,
            retrieved_count=len(sources),
            answer=rag_result.answer,
            confidence=rag_result.confidence,
            sources=[s.model_dump() for s in sources],
            processing_time_ms=int(rag_result.processing_time_ms),
            stage_timings_ms=stage_timings,
            llm_model=rag_result.llm_model,
            embedding_model=rag_result.embedding_model,
        )

        db.add(query_record)
        await db.commit()

        return QueryResponse(
            query_id=str(query_record.id),
            query=request.query,
            answer=rag_result.answer,
            sources=sources,
            processing_time_ms=int(rag_result.processing_time_ms),
            stage_timings_ms=stage_timings,
            confidence=rag_result.confidence,
            timestamp=datetime.utcnow().isoformat(),
        )

    except RAGValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "details": e.details},
        )
    except RAGProcessingError as e:
        logger.error(f"RAG processing error: {e.message} (stage={e.stage})")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "stage": e.stage, "details": e.details},
        )
    except RAGException as e:
        logger.error(f"RAG error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.exception(f"Unexpected error in query endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An unexpected error occurred", "error": str(e)},
        )


@router.get("/documents/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Full-text search across document metadata

    - **q**: Search query
    - **limit**: Results per page (1-100)
    - **offset**: Results to skip
    """
    from sqlalchemy import select, func, or_

    # Build search query
    search_query = (
        select(DocumentModel)
        .where(
            or_(
                DocumentModel.title.ilike(f"%{q}%"),
                DocumentModel.author.ilike(f"%{q}%"),
                DocumentModel.filename.ilike(f"%{q}%"),
            )
        )
        .where(DocumentModel.deleted_at.is_(None))
    )

    # Get total count
    count_query = select(func.count()).select_from(search_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get results
    search_query = search_query.offset(offset).limit(limit)
    result = await db.execute(search_query)
    documents = result.scalars().all()

    results = [
        {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "title": doc.title,
            "upload_date": doc.uploaded_at.isoformat(),
            "page_count": doc.page_count,
            "match_score": 0.8,  # Mock score
            "tags": doc.keywords,
        }
        for doc in documents
    ]

    return SearchResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=results,
    )
