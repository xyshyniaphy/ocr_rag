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
from backend.core.permissions import PermissionChecker
from backend.db.session import get_db_session
from backend.db.models import Document as DocumentModel, User as UserModel
from backend.models.auth import UserResponse
from backend.models.document import DocumentResponse, DocumentStatusResponse, DocumentListResponse
from backend.storage.client import upload_file, get_presigned_url, BUCKET_RAW_PDFS
from backend.api.dependencies import get_current_user

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
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

    # Check for duplicates (excluding soft-deleted documents)
    from sqlalchemy import select
    result = await db.execute(
        select(DocumentModel).where(
            DocumentModel.file_hash == file_hash,
            DocumentModel.deleted_at.is_(None)
        )
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

    # Create document record with authenticated user as owner
    document = DocumentModel(
        id=uuid.uuid4(),
        owner_id=current_user.id,  # Use authenticated user's ID
        filename=file.filename,
        file_size_bytes=len(content),
        file_hash=file_hash,
        content_type=file.content_type,
        title=metadata_dict.get("title", file.filename),
        author=metadata_dict.get("author"),
        keywords=json.dumps(metadata_dict.get("keywords", [])) if metadata_dict.get("keywords") else None,
        language=metadata_dict.get("language", "ja"),
        category=metadata_dict.get("category"),
        doc_metadata=json.dumps(metadata_dict) if metadata_dict else None,
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

    logger.info(f"Document uploaded: {document.id} - {file.filename} by user {current_user.email}")

    # Trigger background processing
    from backend.tasks.document_tasks import process_document
    process_document.delay(str(document.id))

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
    current_user: UserModel = Depends(get_current_user),
):
    """Get document details"""
    from sqlalchemy import select

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check view permission
    await PermissionChecker.require_permission(
        db, doc_uuid, current_user.id, "can_view", current_user.role
    )

    # Use from_db_model to handle proper field mapping
    owner_info = {
        "id": str(document.owner_id),
        "email": current_user.email if document.owner_id == current_user.id else "",
        "full_name": current_user.full_name if document.owner_id == current_user.id else "Unknown",
    }

    return DocumentResponse.from_db_model(document, owner_info)


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get document processing status"""
    from sqlalchemy import select

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
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
        current_stage = "processing"
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
            "ocr": {"status": "pending"},
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
    status: Optional[str] = Query(None, description="Filter by document status"),
    category: Optional[str] = Query(None, description="Filter by document category"),
    date_from: Optional[str] = Query(None, description="Filter documents created after this date (ISO 8601 format)"),
    date_to: Optional[str] = Query(None, description="Filter documents created before this date (ISO 8601 format)"),
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """
    List documents with pagination and filtering

    Only returns documents the user has view access to.

    Filters:
    - status: Filter by document status (pending, processing, completed, failed)
    - category: Filter by document category
    - date_from: Filter documents created after this date (ISO 8601 format)
    - date_to: Filter documents created before this date (ISO 8601 format)
    """
    from sqlalchemy import select, func, or_
    from datetime import datetime

    # Get accessible document IDs for current user
    accessible_ids = await PermissionChecker.get_accessible_documents(
        db, current_user.id, "can_view", limit=None, offset=0
    )

    # Build query (exclude soft-deleted documents)
    query = select(DocumentModel).where(
        DocumentModel.id.in_(accessible_ids),
        DocumentModel.deleted_at.is_(None)
    )

    # Apply filters
    if status:
        query = query.where(DocumentModel.status == status)

    if category:
        query = query.where(DocumentModel.category == category)

    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.where(DocumentModel.created_at >= from_date)
        except ValueError:
            raise ValidationException(
                message="Invalid date format",
                details={"date_from": date_from, "expected_format": "ISO 8601"}
            )

    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.where(DocumentModel.created_at <= to_date)
        except ValueError:
            raise ValidationException(
                message="Invalid date format",
                details={"date_to": date_to, "expected_format": "ISO 8601"}
            )

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
        results=[DocumentResponse.from_db_model(doc) for doc in documents],
    )


@router.delete("/all", status_code=status.HTTP_200_OK)
async def delete_all_documents(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete all documents (hard delete with cascade)

    This will hard-delete ALL documents in the system, including:
    - Files from MinIO (object storage)
    - Chunks from Milvus (vector database)
    - Document and chunk records from PostgreSQL (CASCADE)

    This operation is irreversible and should be used with caution.
    """
    from sqlalchemy import select
    import asyncio

    # Get all documents with their IDs and storage paths
    result = await db.execute(select(DocumentModel))
    documents = result.scalars().all()

    if not documents:
        return {
            "deleted_count": 0,
            "message": "No documents to delete"
        }

    total_count = len(documents)
    document_ids = [str(doc.id) for doc in documents]
    storage_paths = [doc.storage_path for doc in documents]

    # 1. Delete all files from MinIO in parallel
    if storage_paths:
        delete_tasks = []
        for storage_path in storage_paths:
            if storage_path:
                try:
                    from backend.storage.client import delete_file
                    bucket, object_name = storage_path.split('/', 1)
                    delete_tasks.append(delete_file(bucket, object_name))
                except Exception as e:
                    logger.warning(f"Failed to prepare MinIO deletion for {storage_path}: {e}")

        if delete_tasks:
            try:
                await asyncio.gather(*delete_tasks, return_exceptions=True)
                logger.info(f"Deleted {len(delete_tasks)} files from MinIO")
            except Exception as e:
                logger.warning(f"Some MinIO deletions failed: {e}")

    # 2. Delete all chunks from Milvus (batch delete by document)
    try:
        from backend.db.vector.client import delete_by_document
        for doc_id in document_ids:
            try:
                await delete_by_document(doc_id)
            except Exception as e:
                logger.warning(f"Failed to delete chunks for document {doc_id} from Milvus: {e}")
        logger.info(f"Deleted chunks for {total_count} documents from Milvus")
    except Exception as e:
        logger.warning(f"Failed to delete chunks from Milvus: {e}")

    # 3. Delete all documents from PostgreSQL (CASCADE auto-deletes chunk records)
    for document in documents:
        await db.delete(document)
    await db.commit()

    logger.info(f"All documents deleted: {total_count} documents hard-deleted")

    return {
        "deleted_count": total_count,
        "message": f"Successfully deleted {total_count} document(s)"
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """Delete a document (soft delete)"""
    from sqlalchemy import select

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check delete permission
    await PermissionChecker.require_permission(
        db, doc_uuid, current_user.id, "can_delete", current_user.role
    )

    # Soft delete: Set deleted_at timestamp instead of removing from database
    document.deleted_at = datetime.utcnow()
    await db.commit()

    logger.info(f"Document soft-deleted: {document_id} by user {current_user.email}")

    return None


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """Download the original PDF file"""
    from fastapi.responses import Response
    from sqlalchemy import select

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check download permission
    await PermissionChecker.require_permission(
        db, doc_uuid, current_user.id, "can_download", current_user.role
    )

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

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
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
