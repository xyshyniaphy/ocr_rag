#!/usr/bin/env python3
"""
Integration Tests for Authentication API
Tests for backend/api/v1/auth.py endpoints
"""

import pytest
import uuid
from httpx import AsyncClient
from datetime import datetime
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestAuthAPIRegistration:
    """Test user registration endpoint"""

    @pytest.mark.asyncio
    async def test_register_new_user_returns_200(self, client: AsyncClient):
        """Test successful user registration returns 200"""
        user_data = {
            "email": f"newuser_{uuid.uuid4().hex[:8]}@example.com",
            "password": "SecurePassword123!",
            "full_name": "New Test User"
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 200
        result = response.json()
        assert "access_token" in result
        assert "refresh_token" in result
        assert "user" in result
        assert result["token_type"] == "Bearer"
        assert "expires_in" in result
        assert result["user"]["email"] == user_data["email"]
        assert result["user"]["full_name"] == user_data["full_name"]

    @pytest.mark.asyncio
    async def test_register_with_duplicate_email_returns_400(self, client: AsyncClient, test_user: dict):
        """Test registering with existing email returns error"""
        # First registration should succeed
        response1 = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        assert response1.status_code in [200, 201]

        # Second registration with same email should fail
        response2 = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        assert response2.status_code in [400, 422]
        result = response2.json()
        assert "error" in result or "detail" in result

    @pytest.mark.asyncio
    async def test_register_with_invalid_email_returns_422(self, client: AsyncClient):
        """Test registration with invalid email format returns validation error"""
        user_data = {
            "email": "invalid-email-format",
            "password": "SecurePassword123!",
            "full_name": "Test User"
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 400
        result = response.json()
        assert "error" in result or "detail" in result

    @pytest.mark.asyncio
    async def test_register_with_short_password_returns_422(self, client: AsyncClient):
        """Test registration with short password returns validation error"""
        user_data = {
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
            "password": "short",  # Too short
            "full_name": "Test User"
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 400
        result = response.json()
        assert "error" in result or "detail" in result

    @pytest.mark.asyncio
    async def test_register_with_missing_fields_returns_422(self, client: AsyncClient):
        """Test registration with missing required fields returns validation error"""
        user_data = {
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com"
            # Missing password and full_name
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 400
        result = response.json()
        assert "error" in result or "detail" in result

    @pytest.mark.asyncio
    async def test_register_with_optional_fields(self, client: AsyncClient):
        """Test registration with optional display_name and role fields"""
        user_data = {
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User",
            "display_name": "Testy",
            "role": "user"
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["user"]["email"] == user_data["email"]
        assert result["user"]["role"] == user_data["role"]


@pytest.mark.integration
class TestAuthAPILogin:
    """Test user login endpoint"""

    @pytest.mark.asyncio
    async def test_login_with_valid_credentials_returns_200(self, client: AsyncClient, test_user: dict):
        """Test successful login returns JWT tokens"""
        # First register the user
        await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        # Then login
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert "access_token" in result
        assert "refresh_token" in result
        assert "user" in result
        assert result["token_type"] == "Bearer"
        assert "expires_in" in result
        assert isinstance(result["access_token"], str)
        assert len(result["access_token"]) > 0

    @pytest.mark.asyncio
    async def test_login_with_invalid_email_returns_401(self, client: AsyncClient):
        """Test login with non-existent email returns 401"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "SomePassword123!"
            }
        )

        assert response.status_code == 401
        result = response.json()
        assert "detail" in result or "error" in result

    @pytest.mark.asyncio
    async def test_login_with_invalid_password_returns_401(self, client: AsyncClient, test_user: dict):
        """Test login with wrong password returns 401"""
        # First register the user
        await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        # Then login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": "WrongPassword123!"
            }
        )

        assert response.status_code == 401
        result = response.json()
        assert "error" in result
        assert result["error"]["message"] == "Invalid email or password"

    @pytest.mark.asyncio
    async def test_login_with_missing_fields_returns_422(self, client: AsyncClient):
        """Test login with missing fields returns validation error"""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com"}  # Missing password
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_login_with_inactive_account_returns_403(self, client: AsyncClient, db_session):
        """Test login with inactive account returns 403"""
        from backend.db.models import User
        from backend.core.security import get_password_hash
        import uuid

        # Create inactive user
        user = User(
            id=uuid.uuid4(),
            email=f"inactive_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("Password123!"),
            full_name="Inactive User",
            is_active=False
        )
        db_session.add(user)
        await db_session.commit()

        # Try to login
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": user.email,
                "password": "Password123!"
            }
        )

        assert response.status_code == 403
        result = response.json()
        assert "error" in result
        assert result["error"]["message"] == "Account is disabled"


@pytest.mark.integration
class TestAuthAPIRefresh:
    """Test token refresh endpoint"""

    @pytest.mark.asyncio
    async def test_refresh_with_valid_token_returns_200(self, client: AsyncClient, test_user: dict):
        """Test token refresh with valid refresh token"""
        # First login to get tokens
        login_response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]

        # Refresh the tokens
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        assert response.status_code == 200
        result = response.json()
        assert "access_token" in result
        assert "refresh_token" in result
        assert "user" in result
        assert result["token_type"] == "Bearer"
        assert isinstance(result["access_token"], str)

    @pytest.mark.asyncio
    async def test_refresh_with_invalid_token_returns_401(self, client: AsyncClient):
        """Test refresh with invalid token returns 401"""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_refresh_token_12345"}
        )

        assert response.status_code in [401, 422]
        result = response.json()
        assert "detail" in result or "error" in result

    @pytest.mark.asyncio
    async def test_refresh_with_missing_token_returns_422(self, client: AsyncClient):
        """Test refresh without token returns validation error"""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={}  # Missing refresh_token field
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_refresh_generates_new_tokens(self, client: AsyncClient, test_user: dict):
        """Test that refresh generates new access token"""
        import asyncio

        # First login
        login_response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        tokens = login_response.json()
        old_access_token = tokens["access_token"]
        old_refresh_token = tokens["refresh_token"]

        # Wait 1 second to ensure different exp timestamp
        await asyncio.sleep(1)

        # Refresh
        refresh_response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": old_refresh_token}
        )
        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()

        # New tokens should be different
        assert new_tokens["access_token"] != old_access_token
        # New refresh token may or may not be different (implementation choice)


@pytest.mark.integration
class TestAuthAPILogout:
    """Test logout endpoint"""

    @pytest.mark.asyncio
    async def test_logout_returns_200(self, client: AsyncClient):
        """Test logout returns success message"""
        response = await client.post("/api/v1/auth/logout")

        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert result["message"] == "Successfully logged out"


@pytest.mark.integration
class TestAuthAPIMe:
    """Test current user endpoint"""

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_token_returns_200(self, client: AsyncClient, auth_headers: dict):
        """Test getting current user with valid token"""
        response = await client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        # Note: API does not expose user ID in responses for security
        assert "email" in result
        assert "full_name" in result
        assert "role" in result

    @pytest.mark.asyncio
    async def test_get_current_user_without_token_returns_401(self, client: AsyncClient):
        """Test getting current user without token returns 401"""
        response = await client.get("/api/v1/auth/me")

        assert response.status_code == 401
        result = response.json()
        assert "detail" in result or "error" in result

    @pytest.mark.asyncio
    async def test_get_current_user_with_invalid_token_returns_401(self, client: AsyncClient):
        """Test getting current user with invalid token returns 401"""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token_12345"}
        )

        assert response.status_code == 401
        result = response.json()
        assert "detail" in result or "error" in result

    @pytest.mark.asyncio
    async def test_get_current_user_with_malformed_header_returns_401(self, client: AsyncClient):
        """Test getting current user with malformed auth header returns 401"""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "InvalidFormat token123"}
        )

        assert response.status_code == 401
        result = response.json()
        assert "detail" in result or "error" in result


@pytest.mark.integration
class TestAuthAPIResponseStructure:
    """Test response structure and data types"""

    @pytest.mark.asyncio
    async def test_token_response_structure(self, client: AsyncClient, test_user: dict):
        """Test TokenResponse has all required fields"""
        response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        assert response.status_code == 200
        result = response.json()

        # Required fields
        required_fields = ["access_token", "refresh_token", "token_type", "expires_in", "user"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # User object fields
        # Note: API does not expose user ID in responses for security
        user_fields = ["email", "full_name", "role", "is_active", "created_at"]
        for field in user_fields:
            assert field in result["user"], f"Missing user field: {field}"

    @pytest.mark.asyncio
    async def test_user_response_fields(self, client: AsyncClient, auth_headers: dict):
        """Test UserResponse has all required fields"""
        response = await client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Check field types
        # Note: API does not expose user ID in responses for security
        assert isinstance(result["email"], str)
        assert isinstance(result["full_name"], str)
        assert isinstance(result["role"], str)
        assert isinstance(result["is_active"], bool)


@pytest.mark.integration
class TestAuthAPISecurity:
    """Test security aspects of auth API"""

    @pytest.mark.asyncio
    async def test_password_is_hashed_in_database(self, client: AsyncClient, test_user: dict, db_session):
        """Test that passwords are properly hashed, not stored in plaintext"""
        from sqlalchemy import select
        from backend.db.models import User

        # Register user
        response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        # Note: API does not expose user ID, so we query by email
        user_email = test_user["email"]

        # Check database
        result = await db_session.execute(
            select(User).where(User.email == user_email)
        )
        user = result.scalar_one_or_none()

        assert user is not None
        assert user.hashed_password != test_user["password"]
        assert user.hashed_password.startswith("$2b$")  # bcrypt hash

    @pytest.mark.asyncio
    async def test_jwt_token_structure(self, client: AsyncClient, test_user: dict):
        """Test JWT tokens have proper structure"""
        import jwt

        # Register and get tokens
        response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        tokens = response.json()

        # Decode access token (without verification, just check structure)
        access_token = tokens["access_token"]
        parts = access_token.split(".")
        assert len(parts) == 3, "JWT should have 3 parts (header.payload.signature)"

    @pytest.mark.asyncio
    async def test_login_updates_last_login_at(self, client: AsyncClient, test_user: dict, db_session):
        """Test that successful login updates last_login_at timestamp"""
        from sqlalchemy import select
        from backend.db.models import User
        from datetime import datetime, timedelta

        # Register user
        register_response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        # Note: API does not expose user ID, so we query by email
        user_email = test_user["email"]

        # Get initial last_login_at
        result = await db_session.execute(
            select(User).where(User.email == user_email)
        )
        user = result.scalar_one_or_none()
        initial_last_login = user.last_login_at

        # Wait a bit and login again
        import asyncio
        await asyncio.sleep(0.1)

        await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )

        # Check that last_login_at was updated
        await db_session.refresh(user)
        assert user.last_login_at is not None
        if initial_last_login:
            assert user.last_login_at > initial_last_login


@pytest.mark.integration
class TestAuthAPIEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_register_with_very_long_name(self, client: AsyncClient):
        """Test registration with very long full name"""
        user_data = {
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
            "password": "SecurePassword123!",
            "full_name": "A" * 255  # Max length
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        # Should succeed or fail gracefully
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_register_with_unicode_email(self, client: AsyncClient):
        """Test registration with Unicode characters in email"""
        user_data = {
            "email": f"test{uuid.uuid4().hex[:8]}@例え.jp",  # Japanese TLD
            "password": "SecurePassword123!",
            "full_name": "テストユーザー"  # Japanese name
        }

        response = await client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        # Email validation may reject this
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_multiple_refreshes(self, client: AsyncClient, test_user: dict):
        """Test multiple sequential token refreshes"""
        # Login
        login_response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        tokens = login_response.json()

        # Refresh multiple times
        for i in range(3):
            refresh_response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": tokens["refresh_token"]}
            )
            assert refresh_response.status_code == 200
            tokens = refresh_response.json()
            assert "access_token" in tokens
            assert "refresh_token" in tokens

    @pytest.mark.asyncio
    async def test_concurrent_login_requests(self, client: AsyncClient, test_user: dict):
        """Test handling concurrent login requests"""
        import asyncio

        # Register user first
        await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        # Send concurrent login requests
        async def login():
            return await client.post(
                "/api/v1/auth/login",
                json={
                    "email": test_user["email"],
                    "password": test_user["password"]
                }
            )

        results = await asyncio.gather(login(), login(), login())

        # All should succeed
        for response in results:
            assert response.status_code == 200
            assert "access_token" in response.json()


@pytest.mark.integration
class TestAuthAPIProfileManagement:
    """Test user profile management endpoints"""

    @pytest.mark.asyncio
    async def test_update_profile_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful profile update"""
        # Update profile
        update_data = {
            "full_name": "Updated Name",
            "display_name": "UpdUser"
        }
        response = await client.put(
            "/api/v1/auth/me",
            headers=auth_headers,
            json=update_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["full_name"] == "Updated Name"
        assert result["display_name"] == "UpdUser"

    @pytest.mark.asyncio
    async def test_update_profile_partial(self, client: AsyncClient, auth_headers: dict):
        """Test partial profile update (only some fields)"""
        # Update only display_name
        update_data = {
            "display_name": "PartialUpdate"
        }
        response = await client.put(
            "/api/v1/auth/me",
            headers=auth_headers,
            json=update_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["display_name"] == "PartialUpdate"
        # full_name should remain unchanged

    @pytest.mark.asyncio
    async def test_update_profile_requires_auth(self, client: AsyncClient):
        """Test profile update requires authentication"""
        response = await client.put(
            "/api/v1/auth/me",
            json={"full_name": "No Auth"}
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_update_profile_empty_body(self, client: AsyncClient, auth_headers: dict):
        """Test profile update with empty body (no changes)"""
        response = await client.put(
            "/api/v1/auth/me",
            headers=auth_headers,
            json={}
        )

        # Should succeed with no changes
        assert response.status_code == 200


@pytest.mark.integration
class TestAuthAPIPasswordChange:
    """Test password change endpoint"""

    @pytest.mark.asyncio
    async def test_change_password_success(self, client: AsyncClient):
        """Test successful password change"""
        # First register a user
        user_data = {
            "email": f"password_change_{uuid.uuid4().hex[:8]}@example.com",
            "password": "OldPassword123!",
            "full_name": "Password Changer"
        }
        register_response = await client.post("/api/v1/auth/register", json=user_data)
        assert register_response.status_code == 200
        tokens = register_response.json()
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Change password
        password_data = {
            "old_password": "OldPassword123!",
            "new_password": "NewPassword456!"
        }
        response = await client.put(
            "/api/v1/auth/me/password",
            headers=auth_headers,
            json=password_data
        )

        assert response.status_code == 200
        result = response.json()
        assert "message" in result

        # Verify new password works for login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": user_data["email"], "password": "NewPassword456!"}
        )
        assert login_response.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(self, client: AsyncClient, auth_headers: dict):
        """Test password change with incorrect old password"""
        password_data = {
            "old_password": "WrongOldPassword123!",
            "new_password": "NewPassword456!"
        }
        response = await client.put(
            "/api/v1/auth/me/password",
            headers=auth_headers,
            json=password_data
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_change_password_same_as_old(self, client: AsyncClient, auth_headers: dict):
        """Test password change when new password same as old"""
        # Assuming test user password is "SecurePassword123!"
        password_data = {
            "old_password": "SecurePassword123!",
            "new_password": "SecurePassword123!"
        }
        response = await client.put(
            "/api/v1/auth/me/password",
            headers=auth_headers,
            json=password_data
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_change_password_requires_auth(self, client: AsyncClient):
        """Test password change requires authentication"""
        response = await client.put(
            "/api/v1/auth/me/password",
            json={"old_password": "old", "new_password": "new"}
        )

        assert response.status_code == 401


@pytest.mark.integration
class TestAuthAPIAdminEndpoints:
    """Test admin-only user management endpoints"""

    @pytest.mark.asyncio
    async def test_list_users_as_admin(self, client: AsyncClient, db_session):
        """Test listing all users as admin"""
        from backend.db.models import User
        from backend.core.security import get_password_hash

        # Create admin user
        admin = User(
            id=uuid.uuid4(),
            email=f"admin_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("AdminPass123!"),
            full_name="Admin User",
            role="admin",
            is_active=True
        )
        db_session.add(admin)
        await db_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": admin.email, "password": "AdminPass123!"}
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # List all users
        response = await client.get(
            "/api/v1/auth/users",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_users_as_regular_user_forbidden(self, client: AsyncClient, auth_headers: dict):
        """Test listing users as regular user returns 403"""
        # Regular user (created by auth_headers fixture) tries to list users
        response = await client.get(
            "/api/v1/auth/users",
            headers=auth_headers
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_user_as_admin(self, client: AsyncClient, db_session):
        """Test deleting a user as admin"""
        from backend.db.models import User
        from backend.core.security import get_password_hash
        from sqlalchemy import select

        # Create admin user
        admin = User(
            id=uuid.uuid4(),
            email=f"admin_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("AdminPass123!"),
            full_name="Admin User",
            role="admin",
            is_active=True
        )
        db_session.add(admin)

        # Create target user
        target_user = User(
            id=uuid.uuid4(),
            email=f"target_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("TargetPass123!"),
            full_name="Target User",
            role="user",
            is_active=True
        )
        db_session.add(target_user)
        await db_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": admin.email, "password": "AdminPass123!"}
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Delete target user
        response = await client.delete(
            f"/api/v1/auth/users/{target_user.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert "message" in result

        # Verify user is deactivated (soft delete)
        # Note: Need to expire session cache to see changes from API's transaction
        target_user_id = target_user.id  # Save ID before expiring
        db_session.expire_all()
        result = await db_session.execute(select(User).where(User.id == target_user_id))
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.is_active is False

    @pytest.mark.asyncio
    async def test_delete_user_as_regular_user_forbidden(self, client: AsyncClient, db_session):
        """Test deleting user as regular user returns 403"""
        from backend.db.models import User
        from backend.core.security import get_password_hash

        # Create target user
        target_user = User(
            id=uuid.uuid4(),
            email=f"target_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("TargetPass123!"),
            full_name="Target User",
            role="user",
            is_active=True
        )
        db_session.add(target_user)
        await db_session.commit()

        # Create regular user
        regular_user_data = {
            "email": f"regular_{uuid.uuid4().hex[:8]}@example.com",
            "password": "RegularPass123!",
            "full_name": "Regular User"
        }
        register_response = await client.post("/api/v1/auth/register", json=regular_user_data)
        assert register_response.status_code == 200
        tokens = register_response.json()
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Regular user tries to delete
        response = await client.delete(
            f"/api/v1/auth/users/{target_user.id}",
            headers=auth_headers
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_self_prevented(self, client: AsyncClient, db_session):
        """Test admin cannot delete their own account"""
        from backend.db.models import User
        from backend.core.security import get_password_hash

        # Create admin user
        admin = User(
            id=uuid.uuid4(),
            email=f"admin_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("AdminPass123!"),
            full_name="Admin User",
            role="admin",
            is_active=True
        )
        db_session.add(admin)
        await db_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": admin.email, "password": "AdminPass123!"}
        )
        tokens = login_response.json()
        auth_headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Try to delete self
        response = await client.delete(
            f"/api/v1/auth/users/{admin.id}",
            headers=auth_headers
        )

        assert response.status_code == 400
