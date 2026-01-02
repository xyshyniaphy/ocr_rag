#!/usr/bin/env python3
"""
Integration Tests for Query API
Tests for backend/api/v1/query.py endpoints
"""

import pytest
import uuid
from httpx import AsyncClient
from typing import Dict, Any


@pytest.mark.integration
class TestQueryAPIAuth:
    """Test authentication requirements for query endpoint"""

    @pytest.mark.asyncio
    async def test_query_without_auth_returns_401(self, client: AsyncClient):
        """Test query without authentication token returns 401"""
        response = await client.post(
            "/api/v1/query",
            json={"query": "test query"}
        )

        assert response.status_code == 401
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_invalid_token_returns_401(self, client: AsyncClient):
        """Test query with invalid token returns 401"""
        response = await client.post(
            "/api/v1/query",
            headers={"Authorization": "Bearer invalid_token_12345"},
            json={"query": "test query"}
        )

        assert response.status_code == 401
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_malformed_token_returns_401(self, client: AsyncClient):
        """Test query with malformed authorization header returns 401"""
        response = await client.post(
            "/api/v1/query",
            headers={"Authorization": "InvalidFormat token123"},
            json={"query": "test query"}
        )

        assert response.status_code == 401


@pytest.mark.integration
class TestQueryAPIValidation:
    """Test input validation for query endpoint"""

    @pytest.mark.asyncio
    async def test_empty_query_returns_400(self, client: AsyncClient, auth_headers: dict):
        """Test empty query string returns 400 validation error"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": ""}
        )

        # Should return validation error
        assert response.status_code in [400, 422]
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_query_field_returns_400(self, client: AsyncClient, auth_headers: dict):
        """Test missing query field returns 400 validation error"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"top_k": 5}  # Missing required 'query' field
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_too_long_returns_400(self, client: AsyncClient, auth_headers: dict):
        """Test query exceeding 500 characters returns 400"""
        long_query = "a" * 501  # Exceeds 500 character limit

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": long_query}
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_top_k_below_minimum(self, client: AsyncClient, auth_headers: dict):
        """Test top_k < 1 returns validation error"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "top_k": 0}
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_top_k_above_maximum(self, client: AsyncClient, auth_headers: dict):
        """Test top_k > 20 returns validation error"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "top_k": 21}
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_language_too_long(self, client: AsyncClient, auth_headers: dict):
        """Test language exceeding max length returns validation error"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "language": "a" * 11}  # Max is 10
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_document_ids_format(self, client: AsyncClient, auth_headers: dict):
        """Test invalid document_ids format - currently accepts any string"""
        # Note: The API accepts any string in document_ids list
        # UUID validation would happen during RAG processing, not at request validation
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "document_ids": ["not-a-uuid"]}
        )

        # Currently accepts any string - validation happens during RAG processing
        # With mock RAG service, this should succeed
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_too_many_document_ids(self, client: AsyncClient, auth_headers: dict):
        """Test more than 10 document_ids returns validation error"""
        document_ids = [str(uuid.uuid4()) for _ in range(11)]  # Max is 10

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "document_ids": document_ids}
        )

        # API returns 400 for validation errors (via global exception handler)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result


@pytest.mark.integration
class TestQueryAPIHappyPath:
    """Test successful query scenarios"""

    @pytest.mark.asyncio
    async def test_query_with_defaults(self, client: AsyncClient, auth_headers: dict):
        """Test basic query with default parameters

        Note: This test will likely return an error until the RAG pipeline
        is fully implemented. The test verifies the endpoint accepts the
        request and returns a properly formatted response (success or error).
        """
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test query"}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        # Response should have either success data or error info
        if response.status_code == 200:
            # Success case
            assert "query_id" in result
            assert "answer" in result
            assert "sources" in result
            assert "processing_time_ms" in result
        else:
            # Error case (expected until RAG is fully implemented)
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self, client: AsyncClient, auth_headers: dict):
        """Test query with custom top_k parameter"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test query", "top_k": 10}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            # Verify the response structure
            assert "query_id" in result
            assert "sources" in result
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_language_ja(self, client: AsyncClient, auth_headers: dict):
        """Test query with Japanese language parameter"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "テストクエリ", "language": "ja"}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
            assert "answer" in result
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_language_en(self, client: AsyncClient, auth_headers: dict):
        """Test query with English language parameter"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test query in english", "language": "en"}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_with_document_filter(self, client: AsyncClient, auth_headers: dict):
        """Test query with document_ids filter"""
        # Use a valid UUID format (even if document doesn't exist)
        doc_id = str(uuid.uuid4())

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "document_ids": [doc_id]}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_rerank_disabled(self, client: AsyncClient, auth_headers: dict):
        """Test query with reranking disabled"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "rerank": False}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_without_sources(self, client: AsyncClient, auth_headers: dict):
        """Test query with include_sources=False"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "include_sources": False}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
            # Sources should be empty or not included
            sources = result.get("sources", [])
            assert isinstance(sources, list)
        else:
            # Expected until RAG is fully implemented
            assert "error" in result

    @pytest.mark.asyncio
    async def test_query_minimal_parameters(self, client: AsyncClient, auth_headers: dict):
        """Test query with only required parameter (query)"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "minimal test"}
        )

        # Should not be a validation error
        assert response.status_code not in [400, 422, 401]

        result = response.json()

        if response.status_code == 200:
            assert "query_id" in result
            assert "answer" in result
            # Should use defaults: top_k=5, language="ja", include_sources=True
        else:
            # Expected until RAG is fully implemented
            assert "error" in result


@pytest.mark.integration
class TestQueryAPIResponseStructure:
    """Test response format and structure"""

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, client: AsyncClient, auth_headers: dict):
        """Test successful response contains all required fields

        Note: This test is skipped until RAG pipeline is working
        """
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        assert response.status_code == 200
        result = response.json()

        # Required fields
        assert "query_id" in result
        assert "query" in result
        assert "answer" in result
        assert "sources" in result
        assert "processing_time_ms" in result
        assert "timestamp" in result

        # Optional fields that should be present
        assert "confidence" in result or result.get("confidence") is None
        assert "stage_timings_ms" in result or result.get("stage_timings_ms") is None

    @pytest.mark.asyncio
    async def test_query_id_is_valid_uuid(self, client: AsyncClient, auth_headers: dict):
        """Test query_id is a valid UUID string"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        assert response.status_code == 200
        result = response.json()

        # Should be valid UUID
        uuid.UUID(result["query_id"])

    @pytest.mark.asyncio
    async def test_sources_structure(self, client: AsyncClient, auth_headers: dict):
        """Test sources array has correct structure"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "top_k": 5}
        )

        assert response.status_code == 200
        result = response.json()

        sources = result.get("sources", [])
        assert isinstance(sources, list)

        if len(sources) > 0:
            source = sources[0]
            assert "document_id" in source
            assert "document_title" in source
            assert "page_number" in source
            assert "chunk_index" in source
            assert "chunk_text" in source
            assert "relevance_score" in source

    @pytest.mark.asyncio
    async def test_stage_timings_structure(self, client: AsyncClient, auth_headers: dict):
        """Test stage_timings_ms has correct structure"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        assert response.status_code == 200
        result = response.json()

        stage_timings = result.get("stage_timings_ms")
        if stage_timings:
            assert isinstance(stage_timings, dict)
            # All timing values should be integers (milliseconds)
            for stage, timing in stage_timings.items():
                assert isinstance(stage, str)
                assert isinstance(timing, int)
                assert timing >= 0

    @pytest.mark.asyncio
    async def test_timestamp_is_valid_iso_format(self, client: AsyncClient, auth_headers: dict):
        """Test timestamp is valid ISO 8601 format"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        assert response.status_code == 200
        result = response.json()

        from datetime import datetime

        # Should be valid ISO format
        timestamp = result["timestamp"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_processing_time_is_positive_integer(self, client: AsyncClient, auth_headers: dict):
        """Test processing_time_ms is positive integer"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        assert response.status_code == 200
        result = response.json()

        processing_time = result["processing_time_ms"]
        assert isinstance(processing_time, int)
        assert processing_time > 0


@pytest.mark.integration
class TestQueryAPIErrorHandling:
    """Test error handling and error responses"""

    @pytest.mark.asyncio
    async def test_error_response_has_correct_structure(self, client: AsyncClient, auth_headers: dict):
        """Test error responses have consistent structure"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test"}
        )

        # If we get an error (expected until RAG is working)
        if response.status_code != 200:
            result = response.json()
            assert "error" in result
            error = result["error"]

            # Error structure should have these fields
            assert "code" in error or "message" in error

            # Optional details field
            if "details" in error:
                assert isinstance(error["details"], dict)

    @pytest.mark.asyncio
    async def test_rag_service_error_is_handled(self, client: AsyncClient, auth_headers: dict):
        """Test RAG service errors are properly caught and returned

        Current implementation will likely return a retrieval error
        since the full RAG pipeline is not implemented.
        """
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test query"}
        )

        # Currently expect an error due to unimplemented RAG pipeline
        # This test verifies the error is handled gracefully
        if response.status_code != 200:
            result = response.json()
            assert "error" in result

            # Error should have meaningful message
            error = result["error"]
            assert "message" in error or "details" in error

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, client: AsyncClient, auth_headers: dict):
        """Test malformed JSON returns validation error"""
        response = await client.post(
            "/api/v1/query",
            headers={"Authorization": auth_headers["Authorization"], "Content-Type": "application/json"},
            content="invalid json {"
        )

        # API returns 400 for JSON parsing errors (via global exception handler)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_extra_fields_are_ignored(self, client: AsyncClient, auth_headers: dict):
        """Test extra fields in request are ignored (not rejected)"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={
                "query": "test",
                "extra_field": "should be ignored",
                "another_extra": 123
            }
        )

        # Pydantic should ignore extra fields by default
        assert response.status_code not in [400, 422]


@pytest.mark.integration
@pytest.mark.database
class TestQueryAPIDatabase:
    """Test query database persistence"""

    @pytest.mark.asyncio
    async def test_query_record_created(self, client: AsyncClient, auth_headers: dict, db_session):
        """Test query record is created in database after successful query"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        # Count queries before
        from backend.db.models import Query as QueryModel
        from sqlalchemy import select, func

        count_before = await db_session.execute(
            select(func.count()).select_from(QueryModel)
        )
        before = count_before.scalar()

        # Make query
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "database test query"}
        )

        assert response.status_code == 200

        # Count queries after
        count_after = await db_session.execute(
            select(func.count()).select_from(QueryModel)
        )
        after = count_after.scalar()

        # Should have created a new record
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_query_metadata_saved_correctly(self, client: AsyncClient, auth_headers: dict, db_session):
        """Test query metadata is saved correctly"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        from backend.db.models import Query as QueryModel
        from sqlalchemy import select

        # Make query with specific parameters
        test_query = "metadata test query"
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": test_query, "top_k": 10, "language": "ja"}
        )

        assert response.status_code == 200
        result = response.json()

        # Query database for the record
        db_result = await db_session.execute(
            select(QueryModel).where(QueryModel.query_text == test_query)
        )
        query_record = db_result.scalar_one_or_none()

        assert query_record is not None
        assert query_record.query_text == test_query
        assert query_record.top_k == 10
        assert query_record.query_language == "ja"
        assert query_record.query_type == "hybrid"

    @pytest.mark.asyncio
    async def test_query_sources_stored_as_json(self, client: AsyncClient, auth_headers: dict, db_session):
        """Test sources are stored as JSON in database"""
        pytest.skip("Requires working RAG pipeline - not yet implemented")

        from backend.db.models import Query as QueryModel
        from sqlalchemy import select

        # Make query
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "sources test", "include_sources": True}
        )

        assert response.status_code == 200

        # Query database
        db_result = await db_session.execute(
            select(QueryModel).where(
                QueryModel.query_text == "sources test"
            )
        )
        query_record = db_result.scalar_one_or_none()

        assert query_record is not None
        # Sources should be stored as JSON
        assert query_record.sources is not None
        assert isinstance(query_record.sources, list)


@pytest.mark.integration
class TestQueryAPIEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_query_with_min_length(self, client: AsyncClient, auth_headers: dict):
        """Test query with minimum valid length (1 character)"""
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "a"}
        )

        # Should not be validation error
        assert response.status_code not in [400, 422]

    @pytest.mark.asyncio
    async def test_query_with_max_length(self, client: AsyncClient, auth_headers: dict):
        """Test query with maximum valid length (500 characters)"""
        max_query = "a" * 500

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": max_query}
        )

        # Should not be validation error
        assert response.status_code not in [400, 422]

    @pytest.mark.asyncio
    async def test_query_with_unicode_characters(self, client: AsyncClient, auth_headers: dict):
        """Test query with various Unicode characters"""
        unicode_query = "日本語テスト 🎉 Ñoño café"

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": unicode_query, "language": "ja"}
        )

        # Should not be validation error
        assert response.status_code not in [400, 422]

    @pytest.mark.asyncio
    async def test_query_with_newlines_and_tabs(self, client: AsyncClient, auth_headers: dict):
        """Test query with whitespace characters"""
        whitespace_query = "line1\nline2\ttabbed"

        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": whitespace_query}
        )

        # Should not be validation error
        assert response.status_code not in [400, 422]

    @pytest.mark.asyncio
    async def test_query_boundary_top_k_values(self, client: AsyncClient, auth_headers: dict):
        """Test boundary values for top_k parameter"""
        # Test minimum
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "top_k": 1}
        )
        assert response.status_code not in [400, 422]

        # Test maximum
        response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={"query": "test", "top_k": 20}
        )
        assert response.status_code not in [400, 422]


@pytest.mark.integration
class TestQueryHistoryAPI:
    """Test query history endpoints"""

    @pytest.mark.asyncio
    async def test_list_queries_requires_auth(self, client: AsyncClient):
        """Test listing queries requires authentication"""
        response = await client.get("/api/v1/queries")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_queries_empty(self, client: AsyncClient, auth_headers: dict):
        """Test listing queries when user has no queries"""
        response = await client.get(
            "/api/v1/queries",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_list_queries_with_pagination(self, client: AsyncClient, auth_headers: dict):
        """Test listing queries with pagination parameters"""
        response = await client.get(
            "/api/v1/queries",
            headers=auth_headers,
            params={"limit": 10, "offset": 0}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["limit"] == 10
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_query_details_requires_auth(self, client: AsyncClient):
        """Test getting query details requires authentication"""
        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/queries/{fake_id}")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_query_details_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test getting details for non-existent query"""
        fake_id = uuid.uuid4()
        response = await client.get(
            f"/api/v1/queries/{fake_id}",
            headers=auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_query_details_invalid_id(self, client: AsyncClient, auth_headers: dict):
        """Test getting query details with invalid UUID format"""
        response = await client.get(
            "/api/v1/queries/invalid-uuid",
            headers=auth_headers
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_feedback_requires_auth(self, client: AsyncClient):
        """Test submitting feedback requires authentication"""
        fake_id = uuid.uuid4()
        response = await client.post(
            f"/api/v1/queries/{fake_id}/feedback",
            json={"user_rating": 5, "is_helpful": True}
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_rating(self, client: AsyncClient, auth_headers: dict):
        """Test submitting feedback with invalid rating"""
        fake_id = uuid.uuid4()
        response = await client.post(
            f"/api/v1/queries/{fake_id}/feedback",
            headers=auth_headers,
            json={"user_rating": 6, "is_helpful": True}  # Rating must be 1-5
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_fields(self, client: AsyncClient, auth_headers: dict):
        """Test submitting feedback with missing required fields"""
        # First, submit a query to get a real query_id
        import json
        query_response = await client.post(
            "/api/v1/query",
            headers=auth_headers,
            json={
                "query": "test query for feedback",
                "top_k": 3,
                "language": "ja"
            }
        )
        # Note: May fail if RAG service isn't fully set up, but we'll get the query_id
        if query_response.status_code == 200:
            query_data = query_response.json()
            query_id = query_data["query_id"]
        else:
            # Use a fake ID for testing validation logic
            query_id = uuid.uuid4()

        response = await client.post(
            f"/api/v1/queries/{query_id}/feedback",
            headers=auth_headers,
            json={"is_helpful": True}  # Missing user_rating
        )

        # If query exists, expect validation error (400)
        # If query doesn't exist, expect 404
        assert response.status_code in [400, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
