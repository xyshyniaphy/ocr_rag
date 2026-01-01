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
from backend.core.exceptions import ValidationException
from backend.db.session import get_db_session
from backend.db.models import Document as DocumentModel
from backend.models.query import (
    QueryRequest,
    QueryResponse,
    SourceReference,
    SearchRequest,
    SearchResponse,
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
    import time
    from sqlalchemy import select

    # Start timer
    start_time = time.time()
    stage_timings = {}

    # TODO: Implement actual RAG pipeline
    # For now, return a mock response

    # Stage 1: Query understanding
    stage_start = time.time()
    normalized_query = request.query.strip()
    stage_timings["query_understanding"] = int((time.time() - stage_start) * 1000)

    # Stage 2: Vector search (mock)
    stage_start = time.time()
    # TODO: Implement actual vector search
    stage_timings["vector_search"] = int((time.time() - stage_start) * 1000)

    # Stage 3: Reranking (mock)
    stage_start = time.time()
    # TODO: Implement actual reranking
    stage_timings["reranking"] = int((time.time() - stage_start) * 1000)

    # Stage 4: Context assembly (mock)
    stage_start = time.time()
    # TODO: Implement actual context assembly
    stage_timings["context_assembly"] = int((time.time() - stage_start) * 1000)

    # Stage 5: LLM generation (mock)
    stage_start = time.time()
    # TODO: Implement actual LLM generation
    answer = "This is a placeholder response. The RAG pipeline is not yet implemented."
    stage_timings["llm_generation"] = int((time.time() - stage_start) * 1000)

    total_time = int((time.time() - start_time) * 1000)

    # Mock sources
    sources = [
        SourceReference(
            document_id="00000000-0000-0000-0000-000000000000",
            document_title="Sample Document",
            page_number=1,
            chunk_index=0,
            chunk_text="Sample text content...",
            relevance_score=0.9,
        )
    ]

    # Create query record
    from backend.db.models import Query as QueryModel

    query_record = QueryModel(
        id=uuid.uuid4(),
        user_id=None,  # TODO: Get from authenticated user
        query_text=request.query,
        query_language=request.language,
        query_type="hybrid",
        top_k=request.top_k,
        retrieved_count=len(sources),
        answer=answer,
        confidence=0.85,
        sources=[s.model_dump() for s in sources],
        processing_time_ms=total_time,
        stage_timings_ms=stage_timings,
        llm_model="qwen2.5:14b-instruct-q4_K_M",
        embedding_model="sbintuitions/sarashina-embedding-v1-1b",
    )

    db.add(query_record)
    await db.commit()

    return QueryResponse(
        query_id=str(query_record.id),
        query=request.query,
        answer=answer,
        sources=sources,
        processing_time_ms=total_time,
        stage_timings_ms=stage_timings,
        confidence=0.85,
        timestamp=datetime.utcnow().isoformat(),
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
