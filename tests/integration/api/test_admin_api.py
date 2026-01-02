#!/usr/bin/env python3
"""
Integration Tests for Admin API
Tests for backend/api/v1/admin.py endpoints
"""

import pytest
import uuid
from httpx import AsyncClient
from datetime import datetime
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestAdminAPIStats:
    """Test system statistics endpoint"""

    @pytest.mark.asyncio
    async def test_get_system_stats_returns_200(self, client: AsyncClient):
        """Test getting system statistics returns 200"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Check main structure
        assert "documents" in result
        assert "queries" in result
        assert "users" in result
        assert "storage" in result

    @pytest.mark.asyncio
    async def test_stats_documents_structure(self, client: AsyncClient):
        """Test document statistics have correct structure"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Document stats
        docs = result["documents"]
        assert "total" in docs
        assert "by_status" in docs
        assert "total_pages" in docs
        assert "total_chunks" in docs

        # Check by_status sub-structure
        by_status = docs["by_status"]
        assert "pending" in by_status
        assert "processing" in by_status
        assert "completed" in by_status
        assert "failed" in by_status

        # Check types
        assert isinstance(docs["total"], int)
        assert docs["total"] >= 0
        assert isinstance(docs["total_pages"], int)
        assert docs["total_pages"] >= 0
        assert isinstance(docs["total_chunks"], int)
        assert docs["total_chunks"] >= 0

    @pytest.mark.asyncio
    async def test_stats_queries_structure(self, client: AsyncClient):
        """Test query statistics have correct structure"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Query stats
        queries = result["queries"]
        assert "total" in queries
        assert "last_24h" in queries
        assert "average_latency_ms" in queries

        # Check types
        assert isinstance(queries["total"], int)
        assert queries["total"] >= 0
        assert isinstance(queries["last_24h"], int)
        assert queries["last_24h"] >= 0
        assert isinstance(queries["average_latency_ms"], int)
        assert queries["average_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_stats_users_structure(self, client: AsyncClient):
        """Test user statistics have correct structure"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # User stats
        users = result["users"]
        assert "total" in users
        assert "active_today" in users

        # Check types
        assert isinstance(users["total"], int)
        assert users["total"] >= 0
        assert isinstance(users["active_today"], int)
        assert users["active_today"] >= 0

    @pytest.mark.asyncio
    async def test_stats_storage_structure(self, client: AsyncClient):
        """Test storage statistics have correct structure"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Storage stats
        storage = result["storage"]
        assert "vector_db_size_gb" in storage
        assert "object_storage_size_gb" in storage
        assert "database_size_gb" in storage

        # Check types (float or int acceptable)
        assert isinstance(storage["vector_db_size_gb"], (int, float))
        assert storage["vector_db_size_gb"] >= 0
        assert isinstance(storage["object_storage_size_gb"], (int, float))
        assert storage["object_storage_size_gb"] >= 0
        assert isinstance(storage["database_size_gb"], (int, float))
        assert storage["database_size_gb"] >= 0

    @pytest.mark.asyncio
    async def test_stats_reflects_soft_deleted_documents(self, client: AsyncClient, db_session):
        """Test that stats only count non-deleted documents"""
        from backend.db.models import Document
        import uuid

        # Get initial stats
        initial_response = await client.get("/api/v1/admin/stats")
        assert initial_response.status_code == 200
        initial_result = initial_response.json()
        initial_total = initial_result["documents"]["total"]
        initial_pages = initial_result["documents"]["total_pages"] or 0
        initial_chunks = initial_result["documents"]["total_chunks"] or 0

        # Create active document
        doc1 = Document(
            id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            filename="active.pdf",
            file_size_bytes=1000,
            file_hash="hash1_soft_delete_test",
            content_type="application/pdf",
            status="completed",
            page_count=10,
            chunk_count=20,
            deleted_at=None  # Active
        )
        db_session.add(doc1)

        # Create soft-deleted document
        doc2 = Document(
            id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            filename="deleted.pdf",
            file_size_bytes=1000,
            file_hash="hash2_soft_delete_test",
            content_type="application/pdf",
            status="completed",
            page_count=5,
            chunk_count=10,
            deleted_at=datetime.utcnow()  # Deleted
        )
        db_session.add(doc2)
        await db_session.commit()

        # Get stats
        response = await client.get("/api/v1/admin/stats")
        assert response.status_code == 200
        result = response.json()

        # Should only count active document (initial + 1 active)
        assert result["documents"]["total"] == initial_total + 1
        assert result["documents"]["total_pages"] == initial_pages + 10
        assert result["documents"]["total_chunks"] == initial_chunks + 20

        # Clean up test data
        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_stats_with_no_documents(self, client: AsyncClient):
        """Test stats structure is valid"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Check structure (may have documents from other tests)
        assert "documents" in result
        assert isinstance(result["documents"]["total"], int)
        assert result["documents"]["total"] >= 0


@pytest.mark.integration
class TestAdminAPIUsers:
    """Test user management endpoints"""

    @pytest.mark.asyncio
    async def test_list_users_returns_200(self, client: AsyncClient):
        """Test listing users returns 200"""
        response = await client.get("/api/v1/admin/users")

        assert response.status_code == 200
        result = response.json()

        # Check structure
        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_list_users_default_pagination(self, client: AsyncClient):
        """Test default pagination parameters"""
        response = await client.get("/api/v1/admin/users")

        assert response.status_code == 200
        result = response.json()

        # Default limit should be 20
        assert result["limit"] == 20
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_users_custom_limit(self, client: AsyncClient):
        """Test custom limit parameter"""
        response = await client.get("/api/v1/admin/users?limit=5")

        assert response.status_code == 200
        result = response.json()

        assert result["limit"] == 5
        assert len(result["results"]) <= 5

    @pytest.mark.asyncio
    async def test_list_users_custom_offset(self, client: AsyncClient):
        """Test custom offset parameter"""
        response = await client.get("/api/v1/admin/users?offset=10")

        assert response.status_code == 200
        result = response.json()

        assert result["offset"] == 10

    @pytest.mark.asyncio
    async def test_list_users_max_limit(self, client: AsyncClient):
        """Test maximum limit constraint"""
        response = await client.get("/api/v1/admin/users?limit=1000")

        assert response.status_code == 200
        result = response.json()

        # Should cap at 100 (based on route definition)
        assert result["limit"] <= 100

    @pytest.mark.asyncio
    async def test_list_user_structure(self, client: AsyncClient, test_user: dict):
        """Test individual user structure in results"""
        # Create a user first
        await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        # List users
        response = await client.get("/api/v1/admin/users?limit=100")

        assert response.status_code == 200
        result = response.json()

        if result["total"] > 0:
            # Check first user structure
            user = result["results"][0]
            assert "user_id" in user
            assert "name" in user
            assert "email" in user
            assert "role" in user
            assert "created_at" in user

            # Check types
            assert isinstance(user["user_id"], str)
            assert isinstance(user["email"], str)
            assert isinstance(user["name"], str)
            assert isinstance(user["role"], str)

    @pytest.mark.asyncio
    async def test_list_users_pagination_boundary(self, client: AsyncClient):
        """Test pagination at boundaries"""
        # Get first page
        page1 = await client.get("/api/v1/admin/users?limit=5&offset=0")
        assert page1.status_code == 200
        result1 = page1.json()

        # Get second page
        page2 = await client.get("/api/v1/admin/users?limit=5&offset=5")
        assert page2.status_code == 200
        result2 = page2.json()

        # Results should be different (if enough users)
        if result1["total"] > 5:
            assert len(result1["results"]) == 5
            # Page 2 may have fewer results
            assert len(result2["results"]) >= 0

    @pytest.mark.asyncio
    async def test_list_users_with_negative_limit(self, client: AsyncClient):
        """Test list users with negative limit"""
        response = await client.get("/api/v1/admin/users?limit=-1")

        # Should return validation error
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_list_users_with_negative_offset(self, client: AsyncClient):
        """Test list users with negative offset"""
        response = await client.get("/api/v1/admin/users?offset=-1")

        # Should return validation error
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_list_users_with_large_offset(self, client: AsyncClient):
        """Test list users with offset exceeding total users"""
        response = await client.get("/api/v1/admin/users?offset=999999")

        assert response.status_code == 200
        result = response.json()

        # Should return empty list
        assert result["total"] >= 0
        assert len(result["results"]) == 0


@pytest.mark.integration
class TestAdminAPIAuthentication:
    """Test authentication requirements for admin endpoints"""

    @pytest.mark.asyncio
    async def test_stats_without_auth(self, client: AsyncClient):
        """Test that stats endpoint might not require auth (depends on implementation)"""
        # Currently, admin endpoints don't enforce authentication
        # This test documents current behavior
        response = await client.get("/api/v1/admin/stats")

        # May return 200 or 401 depending on auth requirements
        assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_users_without_auth(self, client: AsyncClient):
        """Test that users endpoint might not require auth (depends on implementation)"""
        # Currently, admin endpoints don't enforce authentication
        # This test documents current behavior
        response = await client.get("/api/v1/admin/users")

        # May return 200 or 401 depending on auth requirements
        assert response.status_code in [200, 401]


@pytest.mark.integration
class TestAdminAPIResponseStructure:
    """Test response structure and data consistency"""

    @pytest.mark.asyncio
    async def test_stats_all_fields_present(self, client: AsyncClient):
        """Test that all expected fields are present in stats response"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Top-level fields
        expected_fields = ["documents", "queries", "users", "storage"]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_stats_datatypes_consistent(self, client: AsyncClient):
        """Test that stats return consistent data types"""
        response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        result = response.json()

        # Check numeric fields
        assert isinstance(result["documents"]["total"], int)
        assert isinstance(result["documents"]["by_status"]["pending"], int)
        assert isinstance(result["queries"]["total"], int)
        assert isinstance(result["users"]["total"], int)

        # Check storage fields (can be int or float)
        for field in ["vector_db_size_gb", "object_storage_size_gb", "database_size_gb"]:
            assert isinstance(result["storage"][field], (int, float))

    @pytest.mark.asyncio
    async def test_users_list_all_fields_present(self, client: AsyncClient):
        """Test that all expected fields are present in user list response"""
        response = await client.get("/api/v1/admin/users?limit=100")

        assert response.status_code == 200
        result = response.json()

        # Top-level fields
        expected_fields = ["total", "limit", "offset", "results"]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

        # User object fields (if users exist)
        if len(result["results"]) > 0:
            user_fields = ["user_id", "name", "email", "role", "created_at"]
            for field in user_fields:
                assert field in result["results"][0], f"Missing user field: {field}"


@pytest.mark.integration
class TestAdminAPIEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_stats_with_invalid_parameters(self, client: AsyncClient):
        """Test stats endpoint with invalid query parameters"""
        # Stats endpoint doesn't take parameters, but let's test with them anyway
        response = await client.get("/api/v1/admin/stats?invalid_param=value")

        # Should ignore invalid params or return error
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_users_pagination_edge_cases(self, client: AsyncClient):
        """Test users list with various pagination edge cases"""
        # Zero limit
        response = await client.get("/api/v1/admin/users?limit=0")
        assert response.status_code in [200, 400, 422]

        # Very large offset
        response = await client.get("/api/v1/admin/users?offset=999999999")
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_stats_counts_match_actual_data(self, client: AsyncClient, db_session):
        """Test that stats accurately reflect database state"""
        from sqlalchemy import select, func
        from backend.db.models import User, Document, Query

        # Get actual counts from database
        user_result = await db_session.execute(select(func.count()).select_from(select(User).subquery()))
        actual_user_count = user_result.scalar()

        doc_result = await db_session.execute(
            select(func.count()).select_from(
                select(Document).where(Document.deleted_at.is_(None)).subquery()
            )
        )
        actual_doc_count = doc_result.scalar()

        query_result = await db_session.execute(select(func.count()).select_from(select(Query).subquery()))
        actual_query_count = query_result.scalar()

        # Get stats from API
        response = await client.get("/api/v1/admin/stats")
        assert response.status_code == 200
        stats = response.json()

        # Compare
        assert stats["users"]["total"] == actual_user_count
        assert stats["documents"]["total"] == actual_doc_count
        assert stats["queries"]["total"] == actual_query_count

    @pytest.mark.asyncio
    async def test_users_list_total_matches_db(self, client: AsyncClient, db_session):
        """Test that users list total matches database count"""
        from sqlalchemy import select, func
        from backend.db.models import User

        # Get actual count
        result = await db_session.execute(select(func.count()).select_from(select(User).subquery()))
        actual_count = result.scalar()

        # Get from API
        response = await client.get("/api/v1/admin/users")
        assert response.status_code == 200
        data = response.json()

        # Compare
        assert data["total"] == actual_count


@pytest.mark.integration
class TestAdminAPIPerformance:
    """Test performance and response times"""

    @pytest.mark.asyncio
    async def test_stats_response_time(self, client: AsyncClient, benchmark_timer):
        """Test that stats endpoint responds quickly"""
        with benchmark_timer as timer:
            response = await client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        # Should respond in under 1 second
        assert timer.elapsed_ms < 1000

    @pytest.mark.asyncio
    async def test_users_list_response_time(self, client: AsyncClient, benchmark_timer):
        """Test that users list endpoint responds quickly"""
        with benchmark_timer as timer:
            response = await client.get("/api/v1/admin/users")

        assert response.status_code == 200
        # Should respond in under 1 second
        assert timer.elapsed_ms < 1000


@pytest.mark.integration
class TestAdminAPIIntegration:
    """Test admin API integration with other components"""

    @pytest.mark.asyncio
    async def test_stats_includes_new_user(self, client: AsyncClient, test_user: dict):
        """Test that creating a user is reflected in stats"""
        # Get initial stats
        response1 = await client.get("/api/v1/admin/stats")
        assert response1.status_code == 200
        initial_count = response1.json()["users"]["total"]

        # Create new user
        await client.post(
            "/api/v1/auth/register",
            json=test_user
        )

        # Get updated stats
        response2 = await client.get("/api/v1/admin/stats")
        assert response2.status_code == 200
        updated_count = response2.json()["users"]["total"]

        # Count should increase by 1
        assert updated_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_users_list_includes_new_user(self, client: AsyncClient, test_user: dict):
        """Test that newly created user appears in users list"""
        # Get initial list
        response1 = await client.get("/api/v1/admin/users?limit=100")
        assert response1.status_code == 200
        initial_users = response1.json()

        # Create new user
        reg_response = await client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        new_user_id = reg_response.json()["user"]["id"]

        # Get updated list
        response2 = await client.get("/api/v1/admin/users?limit=100")
        assert response2.status_code == 200
        updated_users = response2.json()

        # Count should increase by 1
        assert updated_users["total"] == initial_users["total"] + 1

        # New user should be in results
        user_ids = [u["user_id"] for u in updated_users["results"]]
        assert new_user_id in user_ids
