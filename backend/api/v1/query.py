"""
Query API Routes
RAG query endpoints and search
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger
from backend.core.exceptions import ValidationException, RAGException, AuthenticationException, NotFoundException
from backend.core.security import verify_access_token
from backend.db.session import get_db_session
from backend.db.models import Document as DocumentModel, User as UserModel
from backend.models.query import (
    QueryRequest,
    QueryResponse,
    QueryListResponse,
    SourceReference,
    SearchRequest,
    SearchResponse,
    QueryFeedbackRequest,
)
from backend.services.rag import (
    get_rag_service,
    RAGQueryOptions,
    RAGValidationError,
    RAGProcessingError,
)
from sqlalchemy import select

logger = get_logger(__name__)
router = APIRouter()


async def get_current_user_required(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session),
) -> UserModel:
    """
    Required dependency to get current user from JWT token
    Raises AuthenticationException if not authenticated
    """
    if not authorization:
        raise AuthenticationException(message="Missing authorization header")

    if not authorization.startswith("Bearer "):
        raise AuthenticationException(message="Invalid authorization header format")

    try:
        token = authorization.split(" ")[1]
        payload = verify_access_token(token)

        result = await db.execute(
            select(UserModel).where(UserModel.id == payload["sub"])
        )
        user = result.scalar_one_or_none()

        if user and user.is_active:
            return user

        raise AuthenticationException(message="User not found or inactive")
    except AuthenticationException:
        raise
    except Exception as e:
        raise AuthenticationException(message=f"Authentication failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    current_user: UserModel = Depends(get_current_user_required),
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
        rag_service = get_rag_service()
        rag_result = await rag_service.query(
            query=request.query,
            options=options,
            user_id=str(current_user.id),
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
                rerank_score=source.rerank_score,  # Add rerank_score
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
        import json

        query_record = QueryModel(
            id=uuid.UUID(rag_result.query_id) if rag_result.query_id else uuid.uuid4(),
            user_id=current_user.id,
            query_text=request.query,
            query_language=request.language,
            query_type="hybrid",
            top_k=request.top_k,
            retrieved_count=len(sources),
            answer=rag_result.answer,
            confidence=rag_result.confidence,
            sources=json.dumps([s.model_dump() for s in sources], ensure_ascii=False),  # Serialize to JSON string
            processing_time_ms=int(rag_result.processing_time_ms),
            stage_timings_ms=json.dumps(stage_timings, ensure_ascii=False),  # Serialize to JSON string
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
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": e.message, "code": e.code, "details": e.details},
        )
    except HTTPException:
        # Re-raise HTTPExceptions (including FastAPI validation errors)
        raise
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


@router.get("/queries", response_model=QueryListResponse)
async def list_queries(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user_required),
):
    """
    Get current user's query history
    
    - **limit**: Results per page (1-100)
    - **offset**: Results to skip
    """
    from backend.db.models import Query as QueryModel
    from sqlalchemy import select, func, desc
    
    # Count total queries for user
    count_result = await db.execute(
        select(func.count())
        .select_from(select(QueryModel).where(QueryModel.user_id == current_user.id).subquery())
    )
    total = count_result.scalar()
    
    # Get user's queries
    result = await db.execute(
        select(QueryModel)
        .where(QueryModel.user_id == current_user.id)
        .order_by(desc(QueryModel.created_at))
        .offset(offset)
        .limit(limit)
    )
    queries = result.scalars().all()
    
    # Format results
    results = []
    for q in queries:
        import json
        results.append({
            "query_id": str(q.id),
            "query_text": q.query_text,
            "query_language": q.query_language,
            "answer": q.answer,
            "created_at": q.created_at.isoformat(),
            "processing_time_ms": q.processing_time_ms,
            "confidence": q.confidence,
            "retrieved_count": q.retrieved_count,
            "llm_model": q.llm_model,
        })
    
    return QueryListResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=results,
    )


@router.post("/queries/{query_id}/feedback")
async def submit_query_feedback(
    query_id: str,
    feedback: QueryFeedbackRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user_required),
):
    """
    Submit feedback on a query result
    
    - **user_rating**: Rating from 1-5
    - **is_helpful**: Was the answer helpful?
    - **user_feedback**: Additional feedback text
    """
    from backend.db.models import Query as QueryModel
    import uuid
    
    # Validate query ID
    try:
        query_uuid = uuid.UUID(query_id)
    except ValueError:
        raise ValidationException(
            message="Invalid query ID format",
            details={"query_id": query_id}
        )
    
    # Get query
    result = await db.execute(
        select(QueryModel).where(QueryModel.id == query_uuid)
    )
    query = result.scalar_one_or_none()
    
    if not query:
        raise NotFoundException("Query")
    
    # Verify query belongs to user
    if query.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update feedback
    query.feedback_rating = feedback.user_rating
    query.feedback_is_helpful = feedback.is_helpful
    query.feedback_text = feedback.user_feedback
    query.feedback_submitted_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Feedback submitted for query {query_id} by user {current_user.email}")
    
    return {"message": "Feedback submitted successfully"}


@router.get("/queries/{query_id}")
async def get_query(
    query_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user_required),
):
    """
    Get details of a specific query
    
    Returns the full query with sources and timing information
    """
    from backend.db.models import Query as QueryModel
    import uuid
    import json
    
    # Validate query ID
    try:
        query_uuid = uuid.UUID(query_id)
    except ValueError:
        raise ValidationException(
            message="Invalid query ID format",
            details={"query_id": query_id}
        )
    
    # Get query
    result = await db.execute(
        select(QueryModel).where(QueryModel.id == query_uuid)
    )
    query = result.scalar_one_or_none()
    
    if not query:
        raise NotFoundException("Query")
    
    # Verify query belongs to user
    if query.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Parse sources and timings
    try:
        sources = json.loads(query.sources) if query.sources else []
        stage_timings = json.loads(query.stage_timings_ms) if query.stage_timings_ms else {}
    except json.JSONDecodeError:
        sources = []
        stage_timings = {}
    
    return {
        "query_id": str(query.id),
        "query_text": query.query_text,
        "query_language": query.query_language,
        "query_type": query.query_type,
        "answer": query.answer,
        "sources": sources,
        "confidence": query.confidence,
        "processing_time_ms": query.processing_time_ms,
        "stage_timings_ms": stage_timings,
        "retrieved_count": query.retrieved_count,
        "top_k": query.top_k,
        "llm_model": query.llm_model,
        "embedding_model": query.embedding_model,
        "created_at": query.created_at.isoformat(),
        "feedback_rating": query.feedback_rating,
        "feedback_is_helpful": query.feedback_is_helpful,
        "feedback_text": query.feedback_text,
        "feedback_submitted_at": query.feedback_submitted_at.isoformat() if query.feedback_submitted_at else None,
    }
