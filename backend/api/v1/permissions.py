"""
Permission Management API Routes
Grant, revoke, and list permissions for documents
"""

import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from backend.core.exceptions import NotFoundException, ValidationException
from backend.core.permissions import PermissionChecker
from backend.db.session import get_db_session
from backend.db.models import User as UserModel, Document as DocumentModel
from backend.models.permission import Permission as PermissionModel
from backend.api.dependencies import get_current_user

logger = get_logger(__name__)
router = APIRouter()


# Pydantic models for permission requests/responses
class GrantPermissionRequest(BaseModel):
    """Request model for granting permission"""
    user_id: str = Field(..., description="User ID to grant permission to")
    permission: str = Field(..., description="Permission type: can_view, can_download, can_delete")


class RevokePermissionRequest(BaseModel):
    """Request model for revoking permission"""
    user_id: str = Field(..., description="User ID to revoke permission from")
    permission: str = Field(..., description="Permission type to revoke")


class PermissionResponse(BaseModel):
    """Response model for permission"""
    permission_id: str
    document_id: str
    user_id: str
    permission: str
    granted_by: str
    granted_at: str


class DocumentPermissionsListResponse(BaseModel):
    """Response model for listing document permissions"""
    document_id: str
    permissions: List[PermissionResponse]


@router.get("/{document_id}", response_model=DocumentPermissionsListResponse)
async def list_document_permissions(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """
    List all permissions for a document

    Only the document owner or admin can list permissions.
    """
    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise ValidationException(
            message="Invalid document ID format",
            details={"document_id": document_id, "expected_format": "UUID"}
        )

    # Get document
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check if user is owner or admin
    if document.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only document owner or admin can list permissions"
        )

    # Get all permissions for this document
    result = await db.execute(
        select(PermissionModel).where(PermissionModel.resource_id == str(doc_uuid))
    )
    permissions = result.scalars().all()

    return DocumentPermissionsListResponse(
        document_id=document_id,
        permissions=[
            PermissionResponse(
                permission_id=str(perm.id),
                document_id=str(perm.resource_id),
                user_id=str(perm.user_id),
                permission=perm.permission_type,
                granted_by=str(perm.granted_by) if perm.granted_by else "",
                granted_at=perm.created_at.isoformat()
            )
            for perm in permissions
        ]
    )


@router.post("/{document_id}/grant", response_model=PermissionResponse, status_code=status.HTTP_201_CREATED)
async def grant_permission(
    document_id: str,
    request: GrantPermissionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Grant a permission to a user on a document

    Only the document owner or admin can grant permissions.

    Valid permissions: can_view, can_download, can_delete
    """
    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
        target_user_uuid = uuid.UUID(request.user_id)
    except ValueError as e:
        raise ValidationException(
            message="Invalid UUID format",
            details={"error": str(e)}
        )

    # Validate permission type
    valid_permissions = ["can_view", "can_download", "can_delete"]
    if request.permission not in valid_permissions:
        raise ValidationException(
            message="Invalid permission type",
            details={
                "permission": request.permission,
                "valid_permissions": valid_permissions
            }
        )

    # Get document
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check if current user is owner or admin
    if document.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only document owner or admin can grant permissions"
        )

    # Check if target user exists
    result = await db.execute(
        select(UserModel).where(UserModel.id == target_user_uuid)
    )
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise NotFoundException("User")

    # Check if permission already exists
    result = await db.execute(
        select(PermissionModel).where(
            PermissionModel.resource_id == str(doc_uuid),
            PermissionModel.user_id == str(target_user_uuid),
            PermissionModel.permission_type == request.permission
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Permission already granted"
        )

    # Create permission
    permission = PermissionModel(
        resource_type="document",
        resource_id=str(doc_uuid),
        user_id=str(target_user_uuid),
        permission_type=request.permission,
        granted_by=str(current_user.id)
    )

    db.add(permission)
    await db.commit()
    await db.refresh(permission)

    logger.info(
        f"Permission granted: {request.permission} on document {document_id} "
        f"to user {request.user_id} by {current_user.email}"
    )

    return PermissionResponse(
        permission_id=str(permission.id),
        document_id=str(permission.resource_id),
        user_id=str(permission.user_id),
        permission=permission.permission_type,
        granted_by=str(permission.granted_by) if permission.granted_by else "",
        granted_at=permission.created_at.isoformat()
    )


@router.delete("/{document_id}/revoke")
async def revoke_permission(
    document_id: str,
    request: RevokePermissionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Revoke a permission from a user on a document

    Only the document owner or admin can revoke permissions.
    """
    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
        target_user_uuid = uuid.UUID(request.user_id)
    except ValueError as e:
        raise ValidationException(
            message="Invalid UUID format",
            details={"error": str(e)}
        )

    # Get document
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise NotFoundException("Document")

    # Check if current user is owner or admin
    if document.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only document owner or admin can revoke permissions"
        )

    # Check if permission exists
    result = await db.execute(
        select(PermissionModel).where(
            PermissionModel.resource_id == str(doc_uuid),
            PermissionModel.user_id == str(target_user_uuid),
            PermissionModel.permission_type == request.permission
        )
    )
    permission = result.scalar_one_or_none()

    if not permission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )

    # Delete permission
    await db.delete(permission)
    await db.commit()

    logger.info(
        f"Permission revoked: {request.permission} on document {document_id} "
        f"from user {request.user_id} by {current_user.email}"
    )

    return None


@router.get("/user/{user_id}", response_model=List[PermissionResponse])
async def list_user_permissions(
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: UserModel = Depends(get_current_user),
):
    """
    List all permissions for a user

    Users can only view their own permissions unless they are admin.
    """
    # Validate UUID format
    try:
        target_user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise ValidationException(
            message="Invalid user ID format",
            details={"user_id": user_id, "expected_format": "UUID"}
        )

    # Check if current user is the target user or admin
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own permissions"
        )

    # Get all permissions for this user
    result = await db.execute(
        select(PermissionModel).where(PermissionModel.user_id == user_id)
    )
    permissions = result.scalars().all()

    return [
        PermissionResponse(
            permission_id=str(perm.id),
            document_id=str(perm.resource_id),
            user_id=str(perm.user_id),
            permission=perm.permission_type,
            granted_by=str(perm.granted_by) if perm.granted_by else "",
            granted_at=perm.created_at.isoformat()
        )
        for perm in permissions
    ]
