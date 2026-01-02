"""
Documents API Routes
Document upload, status, list, and deletion
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging import get_logger
from backend.core.exceptions import NotFoundException, ValidationException
from backend.db.session import get_db_session
from backend.db.models import Document as DocumentModel, User as UserModel
from backend.models.auth import UserResponse
from backend.models.document import DocumentResponse, DocumentStatusResponse, DocumentListResponse
from backend.storage.client import upload_file, get_presigned_url, BUCKET_RAW_PDFS

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a PDF document for processing

    - **file**: PDF file (max 50MB)
    - **metadata**: Optional JSON metadata
    """
    import hashlib
    import json
    from backend.core.config import settings

    # Validate file type
    if file.content_type != "application/pdf":
        raise ValidationException(
            message="Invalid file type",
            details={"content_type": file.content_type, "expected": "application/pdf"},
        )

    # Read file
    content = await file.read()

    # Check file size
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValidationException(
            message="File too large",
            details={"max_size_mb": settings.MAX_UPLOAD_SIZE_MB},
        )

    # Calculate hash
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicates
    from sqlalchemy import select
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.file_hash == file_hash)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return {
            "document_id": str(existing.id),
            "status": "duplicate",
            "message": "Document already exists",
            "existing_document_id": str(existing.id),
        }

    # Parse metadata
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    # Create document record
    document = DocumentModel(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),  # TODO: Get from authenticated user
        filename=file.filename,
        file_size_bytes=len(content),
        file_hash=file_hash,
        content_type=file.content_type,
        title=metadata_dict.get("title", file.filename),
        author=metadata_dict.get("author"),
        keywords=json.dumps(metadata_dict.get("keywords", [])) if metadata_dict.get("keywords") else None,
        language=metadata_dict.get("language", "ja"),
        category=metadata_dict.get("category"),
        metadata=metadata_dict,
        status="pending",
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Upload to MinIO
    object_name = f"{document.id}/{file.filename}"
    await upload_file(BUCKET_RAW_PDFS, object_name, content, file.content_type)

    # Update storage path
    document.storage_path = f"{BUCKET_RAW_PDFS}/{object_name}"
    await db.commit()

    logger.info(f"Document uploaded: {document.id} - {file.filename}")

    # TODO: Trigger background processing

    return {
        "document_id": str(document.id),
        "status": "pending",
        "filename": file.filename,
        "file_size_bytes": len(content),
        "content_type": file.content_type,
        "upload_timestamp": datetime.utcnow().isoformat(),
        "message": "Document uploaded successfully. Processing started.",
    }


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get document details"""
    from sqlalchemy import select

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Get owner info
    owner_info = {
        "user_id": str(document.owner_id),
        "name": "Unknown",  # TODO: Fetch from user table
    }

    response_data = DocumentResponse.model_validate(document).model_dump()
    response_data["owner"] = owner_info

    return DocumentResponse(**response_data)


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get document processing status"""
    from sqlalchemy import select

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Calculate progress
    progress = 0
    current_stage = None

    if document.status == "pending":
        progress = 0
        current_stage = "queued"
    elif document.status == "processing":
        progress = 50
        current_stage = document.ocr_status or "processing"
    elif document.status == "completed":
        progress = 100
        current_stage = "completed"
    elif document.status == "failed":
        progress = 0
        current_stage = "failed"

    return DocumentStatusResponse(
        document_id=str(document.id),
        status=document.status,
        progress=progress,
        current_stage=current_stage,
        stages={
            "upload": {"status": "completed"},
            "ocr": {"status": document.ocr_status or "pending"},
            "chunking": {"status": "pending"},
            "embedding": {"status": "pending"},
        },
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """List documents with pagination"""
    from sqlalchemy import select, func

    # Build query
    query = select(DocumentModel)

    if status:
        query = query.where(DocumentModel.status == status)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get documents
    query = query.order_by(DocumentModel.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a document"""
    from sqlalchemy import select

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Soft delete
    document.deleted_at = datetime.utcnow()
    await db.commit()

    logger.info(f"Document deleted: {document_id}")

    return None


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Download the original PDF file"""
    from fastapi.responses import Response
    from sqlalchemy import select

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Generate presigned URL
    if document.storage_path:
        bucket, object_name = document.storage_path.split("/", 1)
        url = await get_presigned_url(bucket, object_name, expires=3600)
        return {"download_url": url}

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/{document_id}/thumbnail")
async def get_document_thumbnail(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get document thumbnail"""
    from fastapi.responses import Response
    from sqlalchemy import select

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Generate presigned URL
    if document.thumbnail_path:
        bucket, object_name = document.thumbnail_path.split("/", 1)
        url = await get_presigned_url(bucket, object_name, expires=3600)
        return {"thumbnail_url": url}

    raise HTTPException(status_code=404, detail="Thumbnail not found")
