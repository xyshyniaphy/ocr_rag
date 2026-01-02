#!/usr/bin/env python3
"""
Unit Tests for Permission System
Tests for backend/core/permissions.py
"""

import pytest
import uuid
import unittest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from backend.core.permissions import PermissionChecker
from backend.core.exceptions import PermissionException


@pytest.mark.unit
class TestPermissionChecker:
    """Test PermissionChecker class methods"""

    @pytest.mark.asyncio
    async def test_admin_bypass_all_permissions(self):
        """Test admin role bypasses all permission checks"""
        mock_db = AsyncMock()

        # Admin should bypass all checks
        result = await PermissionChecker.check_permission(
            db=mock_db,
            document_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            permission="can_delete",
            user_role="admin"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_owner_has_all_permissions(self):
        """Test document owner has all permissions"""
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # Mock database response - user is owner
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user_id

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        # Owner should have all permissions
        result = await PermissionChecker.check_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            permission="can_delete",
            user_role="user"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_non_owner_without_permission_denied(self):
        """Test non-owner without explicit permission is denied"""
        doc_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # Mock database - document exists with different owner
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = owner_id

        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = None

        mock_role_perm_result = MagicMock()
        mock_role_perm_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [mock_owner_result, mock_perm_result, mock_role_perm_result]

        result = await PermissionChecker.check_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            permission="can_view",
            user_role="user"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_explicit_permission_granted(self):
        """Test explicit permission grants access"""
        doc_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # Mock database - document exists with different owner
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = owner_id

        # Mock permission record
        mock_permission = MagicMock()
        mock_permission.can_view = True
        mock_permission.expires_at = None
        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = mock_permission

        mock_role_perm_result = MagicMock()
        mock_role_perm_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [mock_owner_result, mock_perm_result, mock_role_perm_result]

        result = await PermissionChecker.check_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            permission="can_view",
            user_role="user"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_wrong_permission_denied(self):
        """Test having one permission doesn't grant others"""
        doc_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # Mock database - document exists with different owner
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = owner_id

        # User has can_view but requesting can_delete
        mock_permission = MagicMock()
        mock_permission.can_view = True
        mock_permission.can_delete = False
        mock_permission.expires_at = None
        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = mock_permission

        mock_role_perm_result = MagicMock()
        mock_role_perm_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [mock_owner_result, mock_perm_result, mock_role_perm_result]

        result = await PermissionChecker.check_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            permission="can_delete",
            user_role="user"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_require_permission_success(self):
        """Test require_permission succeeds when user has permission"""
        mock_db = AsyncMock()

        # Admin should not raise exception
        await PermissionChecker.require_permission(
            db=mock_db,
            document_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            permission="can_delete",
            user_role="admin"
        )

    @pytest.mark.asyncio
    async def test_require_permission_denied_raises_exception(self):
        """Test require_permission raises exception when denied"""
        doc_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # Mock database - document exists with different owner
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = owner_id

        # No permissions
        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = None

        mock_role_perm_result = MagicMock()
        mock_role_perm_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [mock_owner_result, mock_perm_result, mock_role_perm_result]

        # Should raise PermissionException
        with pytest.raises(PermissionException):
            await PermissionChecker.require_permission(
                db=mock_db,
                document_id=doc_id,
                user_id=user_id,
                permission="can_delete",
                user_role="user"
            )

    @pytest.mark.asyncio
    async def test_get_accessible_documents_empty(self):
        """Test getting accessible documents when user has none"""
        user_id = uuid.uuid4()

        # Mock role check (get_user_role uses scalar_one_or_none)
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = "user"

        # Mock owned documents (empty)
        mock_owned_result = MagicMock()
        mock_owned_result.all.return_value = []

        # Mock explicit permissions (empty)
        mock_explicit_result = MagicMock()
        mock_explicit_result.all.return_value = []

        # Mock role permissions (empty)
        mock_role_perm_result = MagicMock()
        mock_role_perm_result.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [
            mock_role_result,
            mock_owned_result,
            mock_explicit_result,
            mock_role_perm_result
        ]

        result = await PermissionChecker.get_accessible_documents(
            db=mock_db,
            user_id=user_id,
            permission="can_view",
            limit=10,
            offset=0
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_accessible_documents_with_results(self):
        """Test getting accessible documents returns IDs"""
        user_id = uuid.uuid4()
        doc_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

        # Mock role check (get_user_role uses scalar_one_or_none)
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = "user"

        # Mock owned documents
        mock_owned_result = MagicMock()
        mock_owned_result.all.return_value = [(doc_ids[0],), (doc_ids[1],)]

        # Mock explicit permissions
        mock_explicit_result = MagicMock()
        mock_explicit_result.all.return_value = [(doc_ids[2],)]

        # Mock role permissions (empty)
        mock_role_perm_result = MagicMock()
        mock_role_perm_result.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [
            mock_role_result,
            mock_owned_result,
            mock_explicit_result,
            mock_role_perm_result
        ]

        result = await PermissionChecker.get_accessible_documents(
            db=mock_db,
            user_id=user_id,
            permission="can_view",
            limit=10,
            offset=0
        )

        assert len(result) == 3
        assert doc_ids[0] in result
        assert doc_ids[1] in result
        assert doc_ids[2] in result

    @pytest.mark.asyncio
    async def test_get_accessible_documents_admin_gets_all(self):
        """Test admin gets all documents"""
        user_id = uuid.uuid4()
        doc_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

        # Mock role check - admin (get_user_role uses scalar_one_or_none)
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = "admin"

        # Mock all documents query
        mock_all_result = MagicMock()
        mock_all_result.all.return_value = [(doc_ids[0],), (doc_ids[1],), (doc_ids[2],)]

        mock_db = AsyncMock()
        # First call: get_user_role
        # Second call: select all document IDs
        mock_db.execute.side_effect = [mock_role_result, mock_all_result]

        result = await PermissionChecker.get_accessible_documents(
            db=mock_db,
            user_id=user_id,
            permission="can_view",
            limit=10,
            offset=0
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full SQLAlchemy setup - tested in integration tests")
    async def test_grant_permission_calls_permission_check(self):
        """Test granting permission calls permission check for granter

        NOTE: This test requires full SQLAlchemy table mappings for Permission model.
        The actual functionality is tested in integration tests.
        """
        pass

    @pytest.mark.asyncio
    async def test_revoke_permission_deletes_record(self):
        """Test revoking permission deletes database record"""
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        admin_id = uuid.uuid4()

        # Mock require_permission -> check_permission internal calls:
        # 1. get_user_role (checks revoker's role)
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = "user"
        # 2. owner check (checks if revoker is owner)
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = admin_id  # admin_id is owner
        # 3. permission check for deletion (in revoke_permission)
        mock_permission = MagicMock(id=uuid.uuid4())
        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = mock_permission

        mock_db = AsyncMock()
        mock_db.delete = AsyncMock()  # Make delete async
        mock_db.commit = AsyncMock()
        mock_db.execute.side_effect = [mock_role_result, mock_owner_result, mock_perm_result]

        result = await PermissionChecker.revoke_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            revoked_by=admin_id
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_permission_silent(self):
        """Test revoking non-existent permission doesn't raise error"""
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        admin_id = uuid.uuid4()

        # Mock require_permission -> check_permission internal calls:
        # 1. get_user_role (checks revoker's role)
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = "user"
        # 2. owner check (checks if revoker is owner)
        mock_owner_result = MagicMock()
        mock_owner_result.scalar_one_or_none.return_value = admin_id  # admin_id is owner
        # 3. permission check for deletion (in revoke_permission) - returns None
        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.side_effect = [mock_role_result, mock_owner_result, mock_perm_result]

        result = await PermissionChecker.revoke_permission(
            db=mock_db,
            document_id=doc_id,
            user_id=user_id,
            revoked_by=admin_id
        )

        # Should return False for not found
        assert result is False


@pytest.mark.unit
class TestPermissionException:
    """Test PermissionException class"""

    def test_permission_exception_creation(self):
        """Test PermissionException can be created"""
        exc = PermissionException(
            message="Access denied",
            details={"document_id": str(uuid.uuid4())}
        )

        assert exc.message == "Access denied"
        assert exc.details["document_id"] is not None
        assert exc.code == "permission_denied"

    def test_permission_exception_default_message(self):
        """Test PermissionException default message"""
        exc = PermissionException()

        assert exc.message == "Permission denied"
        assert exc.code == "permission_denied"
