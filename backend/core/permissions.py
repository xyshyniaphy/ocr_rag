"""
Permission Service
Document-level access control (ACL) enforcement
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from backend.core.logging import get_logger
from backend.core.exceptions import PermissionException
from backend.models.permission import Permission

logger = get_logger(__name__)


class PermissionChecker:
    """Check and enforce document-level permissions"""

    # Default permissions for document owners
    OWNER_PERMISSIONS = {
        "can_view": True,
        "can_download": True,
        "can_delete": True,
        "can_share": True,
    }

    # Default permissions for admin role
    ADMIN_PERMISSIONS = {
        "can_view": True,
        "can_download": True,
        "can_delete": True,
        "can_share": True,
    }

    # Default permissions for regular users (no access unless granted)
    DEFAULT_PERMISSIONS = {
        "can_view": False,
        "can_download": False,
        "can_delete": False,
        "can_share": False,
    }

    @staticmethod
    async def get_user_role(db: AsyncSession, user_id: uuid.UUID) -> str:
        """Get user's role from database"""
        from backend.db.models import User

        result = await db.execute(
            select(User.role).where(User.id == user_id)
        )
        role = result.scalar_one_or_none()
        return role or "user"

    @staticmethod
    async def check_permission(
        db: AsyncSession,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        permission: str,
        user_role: Optional[str] = None,
    ) -> bool:
        """
        Check if user has specific permission on document

        Args:
            db: Database session
            document_id: Document to check
            user_id: User to check
            permission: Permission type (can_view, can_download, can_delete, can_share)
            user_role: User's role (optional, will fetch if not provided)

        Returns:
            True if user has permission, False otherwise
        """
        # Get user role if not provided
        if user_role is None:
            user_role = await PermissionChecker.get_user_role(db, user_id)

        # Admin bypass - admins have all permissions
        if user_role == "admin":
            logger.debug(f"Admin {user_id} granted {permission} on {document_id}")
            return True

        # Check if user is owner
        from backend.db.models import Document

        result = await db.execute(
            select(Document.owner_id).where(Document.id == document_id)
        )
        owner_id = result.scalar_one_or_none()

        if owner_id and owner_id == user_id:
            logger.debug(f"Owner {user_id} granted {permission} on {document_id}")
            return True

        # Check explicit permissions
        result = await db.execute(
            select(Permission).where(
                Permission.document_id == document_id,
                Permission.user_id == user_id,
            )
        )
        perm = result.scalar_one_or_none()

        if perm:
            has_permission = getattr(perm, permission, False)
            # Check if permission is not expired
            if has_permission and perm.expires_at:
                from datetime import datetime
                if datetime.utcnow() > perm.expires_at:
                    logger.warning(f"Permission expired for user {user_id} on document {document_id}")
                    return False
            logger.debug(f"User {user_id} {'granted' if has_permission else 'denied'} {permission} on {document_id}")
            return has_permission

        # Check role-based permissions
        result = await db.execute(
            select(Permission).where(
                Permission.document_id == document_id,
                Permission.role == user_role,
            )
        )
        role_perm = result.scalar_one_or_none()

        if role_perm:
            has_permission = getattr(role_perm, permission, False)
            logger.debug(f"Role {user_role} {'granted' if has_permission else 'denied'} {permission} on {document_id}")
            return has_permission

        logger.debug(f"User {user_id} denied {permission} on {document_id} (no explicit permission)")
        return False

    @staticmethod
    async def require_permission(
        db: AsyncSession,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        permission: str,
        user_role: Optional[str] = None,
    ) -> None:
        """
        Require permission or raise exception

        Raises:
            PermissionException: If user doesn't have required permission
        """
        has_permission = await PermissionChecker.check_permission(
            db, document_id, user_id, permission, user_role
        )

        if not has_permission:
            raise PermissionException(
                message=f"Access denied: Missing '{permission}' permission",
                details={
                    "document_id": str(document_id),
                    "required_permission": permission,
                },
            )

    @staticmethod
    async def grant_permission(
        db: AsyncSession,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        granted_by: uuid.UUID,
        can_view: bool = False,
        can_download: bool = False,
        can_delete: bool = False,
        can_share: bool = False,
        expires_at: Optional[datetime] = None,
    ) -> Permission:
        """Grant explicit permission to user on document"""
        # Check if granter has share permission
        await PermissionChecker.require_permission(db, document_id, granted_by, "can_share")

        # Check if permission already exists
        result = await db.execute(
            select(Permission).where(
                Permission.document_id == document_id,
                Permission.user_id == user_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing permission
            existing.can_view = can_view
            existing.can_download = can_download
            existing.can_delete = can_delete
            existing.can_share = can_share
            existing.expires_at = expires_at
            existing.granted_by = granted_by
            await db.commit()
            await db.refresh(existing)
            logger.info(f"Updated permission for user {user_id} on document {document_id}")
            return existing

        # Create new permission
        permission = Permission(
            document_id=document_id,
            user_id=user_id,
            can_view=can_view,
            can_download=can_download,
            can_delete=can_delete,
            can_share=can_share,
            expires_at=expires_at,
            granted_by=granted_by,
        )
        db.add(permission)
        await db.commit()
        await db.refresh(permission)
        logger.info(f"Granted permission to user {user_id} on document {document_id}")
        return permission

    @staticmethod
    async def revoke_permission(
        db: AsyncSession,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        revoked_by: uuid.UUID,
    ) -> bool:
        """Revoke permission from user on document"""
        # Check if revoker has share permission
        await PermissionChecker.require_permission(db, document_id, revoked_by, "can_share")

        result = await db.execute(
            select(Permission).where(
                Permission.document_id == document_id,
                Permission.user_id == user_id,
            )
        )
        permission = result.scalar_one_or_none()

        if permission:
            await db.delete(permission)
            await db.commit()
            logger.info(f"Revoked permission from user {user_id} on document {document_id}")
            return True

        return False

    @staticmethod
    async def list_permissions(
        db: AsyncSession,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> List[Permission]:
        """List all permissions on a document (requires can_share permission)"""
        # Check if user has share permission
        await PermissionChecker.require_permission(db, document_id, user_id, "can_view")

        result = await db.execute(
            select(Permission).where(Permission.document_id == document_id)
        )
        permissions = result.scalars().all()
        return list(permissions)

    @staticmethod
    async def get_accessible_documents(
        db: AsyncSession,
        user_id: uuid.UUID,
        permission: str = "can_view",
        limit: int = 100,
        offset: int = 0,
    ) -> List[uuid.UUID]:
        """
        Get list of document IDs user has access to

        Returns documents where:
        - User is owner
        - User has explicit permission
        - User's role has permission
        """
        from backend.db.models import Document

        # Get user role
        user_role = await PermissionChecker.get_user_role(db, user_id)

        # Admin can access all documents
        if user_role == "admin":
            result = await db.execute(
                select(Document.id).offset(offset).limit(limit)
            )
            return [row[0] for row in result.all()]

        # Get owned documents
        result = await db.execute(
            select(Document.id)
            .where(Document.owner_id == user_id)
            .offset(offset).limit(limit)
        )
        owned_ids = {row[0] for row in result.all()}

        # Get documents with explicit user permissions
        result = await db.execute(
            select(Permission.document_id)
            .where(
                Permission.user_id == user_id,
                getattr(Permission, permission) == True,
            )
        )
        explicit_ids = {row[0] for row in result.all()}

        # Get documents with role permissions
        result = await db.execute(
            select(Permission.document_id)
            .where(
                Permission.role == user_role,
                getattr(Permission, permission) == True,
            )
        )
        role_ids = {row[0] for row in result.all()}

        # Combine all accessible document IDs
        all_ids = owned_ids | explicit_ids | role_ids

        return list(all_ids)
