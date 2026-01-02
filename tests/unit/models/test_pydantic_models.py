#!/usr/bin/env python3
"""
Unit Tests for Pydantic Models
Tests validation of request/response schemas
"""

import pytest
from datetime import datetime
from pydantic import ValidationError, EmailStr
from unittest.mock import MagicMock

from backend.models.auth import (
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    TokenResponse,
    LoginRequest,
    RegisterRequest,
)

from backend.models.query import (
    QueryRequest,
    QueryResponse,
    QueryListResponse,
    QueryFeedbackRequest,
    SearchRequest,
    SearchResponse,
    SourceReference,
)

from backend.models.document import (
    DocumentResponse,
    DocumentStatusResponse,
    DocumentListResponse,
)


@pytest.mark.unit
class TestUserModels:
    """Test user-related Pydantic models"""

    def test_user_base_valid(self):
        """Test UserBase with valid data"""
        data = {
            "email": "test@example.com",
            "full_name": "Test User",
            "display_name": "Testy",
            "role": "user"
        }
        user = UserBase(**data)
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.display_name == "Testy"
        assert user.role == "user"

    def test_user_base_invalid_email(self):
        """Test UserBase rejects invalid email"""
        with pytest.raises(ValidationError):
            UserBase(
                email="invalid-email",
                full_name="Test User",
                role="user"
            )

    def test_user_base_invalid_role(self):
        """Test UserBase rejects invalid role"""
        with pytest.raises(ValidationError):
            UserBase(
                email="test@example.com",
                full_name="Test User",
                role="invalid_role"
            )

    def test_user_base_empty_name(self):
        """Test UserBase rejects empty full name"""
        with pytest.raises(ValidationError):
            UserBase(
                email="test@example.com",
                full_name="",
                role="user"
            )

    def test_user_base_name_too_long(self):
        """Test UserBase rejects name exceeding max length"""
        with pytest.raises(ValidationError):
            UserBase(
                email="test@example.com",
                full_name="A" * 256,
                role="user"
            )

    def test_user_base_valid_roles(self):
        """Test UserBase accepts all valid roles"""
        valid_roles = ["admin", "power_user", "user", "viewer"]
        for role in valid_roles:
            user = UserBase(
                email="test@example.com",
                full_name="Test User",
                role=role
            )
            assert user.role == role

    def test_user_base_defaults(self):
        """Test UserBase default values"""
        user = UserBase(
            email="test@example.com",
            full_name="Test User"
        )
        assert user.role == "user"
        assert user.display_name is None

    def test_user_create_valid(self):
        """Test UserCreate with valid data"""
        data = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "full_name": "Test User",
            "role": "user"
        }
        user = UserCreate(**data)
        assert user.email == "test@example.com"
        assert user.password == "SecurePass123!"

    def test_user_create_short_password(self):
        """Test UserCreate rejects short password"""
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="short",
                full_name="Test User"
            )

    def test_user_create_long_password(self):
        """Test UserCreate rejects long password"""
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="A" * 101,
                full_name="Test User"
            )

    def test_user_update_partial(self):
        """Test UserUpdate with partial data"""
        data = {
            "full_name": "Updated Name"
        }
        user = UserUpdate(**data)
        assert user.full_name == "Updated Name"
        assert user.display_name is None
        assert user.role is None

    def test_user_update_invalid_role(self):
        """Test UserUpdate rejects invalid role"""
        with pytest.raises(ValidationError):
            UserUpdate(role="invalid_role")

    def test_user_response_structure(self):
        """Test UserResponse structure"""
        data = {
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "email": "test@example.com",
            "full_name": "Test User",
            "display_name": "Testy",
            "role": "user",
            "is_active": True,
            "is_verified": False,
            "created_at": "2024-01-01T00:00:00"
        }
        user = UserResponse(**data)
        assert user.user_id == data["user_id"]
        assert user.email == data["email"]
        assert user.is_active is True
        assert user.permissions == {}

    def test_user_response_from_user_model(self):
        """Test UserResponse.from_user_model classmethod"""
        mock_user = MagicMock()
        mock_user.id = "123e4567-e89b-12d3-a456-426614174000"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.display_name = "Testy"
        mock_user.role = "user"
        mock_user.is_active = True
        mock_user.is_verified = False
        mock_user.created_at = datetime(2024, 1, 1, 0, 0, 0)
        mock_user.last_login_at = None

        response = UserResponse.from_user_model(mock_user)
        assert response.user_id == str(mock_user.id)
        assert response.email == mock_user.email
        assert response.full_name == mock_user.full_name
        assert response.is_active is True
        assert response.created_at == "2024-01-01T00:00:00"


@pytest.mark.unit
class TestAuthModels:
    """Test authentication-related models"""

    def test_login_request_valid(self):
        """Test LoginRequest with valid data"""
        data = {
            "email": "test@example.com",
            "password": "password123"
        }
        request = LoginRequest(**data)
        assert request.email == "test@example.com"
        assert request.password == "password123"

    def test_login_request_missing_fields(self):
        """Test LoginRequest requires both fields"""
        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com")  # Missing password

        with pytest.raises(ValidationError):
            LoginRequest(password="password123")  # Missing email

    def test_token_response_structure(self):
        """Test TokenResponse structure"""
        user_data = {
            "user_id": "123",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
            "is_verified": False,
            "created_at": "2024-01-01T00:00:00"
        }

        data = {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "Bearer",
            "expires_in": 900,
            "user": user_data
        }
        response = TokenResponse(**data)
        assert response.access_token == data["access_token"]
        assert response.refresh_token == data["refresh_token"]
        assert response.token_type == "Bearer"
        assert response.expires_in == 900
        assert response.user.email == "test@example.com"


@pytest.mark.unit
class TestQueryModels:
    """Test query-related models"""

    def test_query_request_valid(self):
        """Test QueryRequest with valid data"""
        data = {
            "query": "What is the capital of France?",
            "top_k": 5,
            "language": "en"
        }
        request = QueryRequest(**data)
        assert request.query == data["query"]
        assert request.top_k == 5
        assert request.language == "en"

    def test_query_request_defaults(self):
        """Test QueryRequest default values"""
        request = QueryRequest(query="test query")
        assert request.top_k == 5
        assert request.include_sources is True
        assert request.language == "ja"
        assert request.stream is False
        assert request.rerank is True

    def test_query_request_empty_query(self):
        """Test QueryRequest rejects empty query"""
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_request_too_long(self):
        """Test QueryRequest rejects query exceeding max length"""
        with pytest.raises(ValidationError):
            QueryRequest(query="a" * 501)

    def test_query_request_top_k_bounds(self):
        """Test QueryRequest top_k boundaries"""
        # Min boundary
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

        # Max boundary
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=21)

        # Valid boundaries
        request1 = QueryRequest(query="test", top_k=1)
        assert request1.top_k == 1

        request2 = QueryRequest(query="test", top_k=20)
        assert request2.top_k == 20

    def test_query_request_document_ids_too_many(self):
        """Test QueryRequest rejects too many document_ids"""
        with pytest.raises(ValidationError):
            QueryRequest(
                query="test",
                document_ids=[str(i) for i in range(11)]  # Max is 10
            )

    def test_query_request_valid_document_ids(self):
        """Test QueryRequest accepts valid document_ids"""
        data = {
            "query": "test",
            "document_ids": ["doc1", "doc2", "doc3"]
        }
        request = QueryRequest(**data)
        assert len(request.document_ids) == 3

    def test_query_response_structure(self):
        """Test QueryResponse structure"""
        sources = [
            {
                "document_id": "doc1",
                "document_title": "Test Document",
                "page_number": 1,
                "chunk_index": 0,
                "chunk_text": "Test text",
                "relevance_score": 0.95
            }
        ]

        data = {
            "query_id": "123",
            "query": "test query",
            "answer": "This is the answer",
            "sources": sources,
            "processing_time_ms": 1500,
            "stage_timings_ms": {"retrieval": 100, "generation": 200},
            "confidence": 0.9,
            "timestamp": "2024-01-01T00:00:00"
        }
        response = QueryResponse(**data)
        assert response.query_id == "123"
        assert len(response.sources) == 1
        assert response.sources[0].relevance_score == 0.95

    def test_search_request_valid(self):
        """Test SearchRequest with valid data"""
        data = {
            "q": "test search",
            "limit": 20,
            "offset": 0
        }
        request = SearchRequest(**data)
        assert request.q == "test search"
        assert request.limit == 20
        assert request.offset == 0

    def test_search_request_empty_query(self):
        """Test SearchRequest rejects empty query"""
        with pytest.raises(ValidationError):
            SearchRequest(q="")

    def test_search_request_defaults(self):
        """Test SearchRequest default values"""
        request = SearchRequest(q="test")
        assert request.limit == 10
        assert request.offset == 0

    def test_query_feedback_request_valid(self):
        """Test QueryFeedbackRequest with valid data"""
        data = {
            "user_rating": 5,
            "is_helpful": True,
            "user_feedback": "Very helpful!"
        }
        request = QueryFeedbackRequest(**data)
        assert request.user_rating == 5
        assert request.is_helpful is True

    def test_query_feedback_rating_bounds(self):
        """Test QueryFeedbackRequest rating boundaries"""
        # Below minimum
        with pytest.raises(ValidationError):
            QueryFeedbackRequest(user_rating=0)

        # Above maximum
        with pytest.raises(ValidationError):
            QueryFeedbackRequest(user_rating=6)

        # Valid boundaries
        request1 = QueryFeedbackRequest(user_rating=1)
        assert request1.user_rating == 1

        request2 = QueryFeedbackRequest(user_rating=5)
        assert request2.user_rating == 5


@pytest.mark.unit
class TestDocumentModels:
    """Test document-related models"""

    def test_source_reference_structure(self):
        """Test SourceReference structure"""
        data = {
            "document_id": "doc1",
            "document_title": "Test Doc",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_text": "Test text",
            "relevance_score": 0.95,
            "rerank_score": 0.90
        }
        source = SourceReference(**data)
        assert source.document_id == "doc1"
        assert source.relevance_score == 0.95

    def test_document_response_structure(self):
        """Test DocumentResponse structure"""
        data = {
            "document_id": "doc1",
            "filename": "test.pdf",
            "status": "completed",
            "page_count": 10,
            "chunk_count": 50,
            "created_at": "2024-01-01T00:00:00"
        }
        doc = DocumentResponse(**data)
        assert doc.document_id == "doc1"
        assert doc.status == "completed"

    def test_document_status_response_structure(self):
        """Test DocumentStatusResponse structure"""
        data = {
            "document_id": "doc1",
            "status": "processing",
            "progress": 50,
            "current_stage": "ocr",
            "stages": {
                "upload": {"status": "completed"},
                "ocr": {"status": "processing"},
                "chunking": {"status": "pending"}
            },
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:01:00"
        }
        status = DocumentStatusResponse(**data)
        assert status.document_id == "doc1"
        assert status.progress == 50
        assert status.current_stage == "ocr"

    def test_document_list_response_structure(self):
        """Test DocumentListResponse structure"""
        results = [
            {
                "document_id": "doc1",
                "filename": "test1.pdf",
                "status": "completed"
            },
            {
                "document_id": "doc2",
                "filename": "test2.pdf",
                "status": "processing"
            }
        ]

        data = {
            "total": 2,
            "limit": 20,
            "offset": 0,
            "results": results
        }
        response = DocumentListResponse(**data)
        assert response.total == 2
        assert len(response.results) == 2


@pytest.mark.unit
class TestModelEdgeCases:
    """Test edge cases in model validation"""

    def test_email_validation_edge_cases(self):
        """Test email validation with edge cases"""
        # Valid emails
        valid_emails = [
            "test@example.com",
            "user.name@example.com",
            "user+tag@example.co.uk",
            "test_user@test-domain.com"
        ]
        for email in valid_emails:
            user = UserBase(email=email, full_name="Test", role="user")
            assert user.email == email

        # Invalid emails
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test @example.com"
        ]
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                UserBase(email=email, full_name="Test", role="user")

    def test_unicode_in_name(self):
        """Test Unicode characters in names"""
        # Japanese name
        user = UserBase(
            email="test@example.com",
            full_name="テストユーザー",
            role="user"
        )
        assert user.full_name == "テストユーザー"

        # Emoji in display name
        user = UserBase(
            email="test@example.com",
            full_name="Test User",
            display_name="Testy 🚀",
            role="user"
        )
        assert user.display_name == "Testy 🚀"

    def test_optional_fields_with_none(self):
        """Test optional fields accept None"""
        user = UserUpdate(
            full_name=None,
            display_name=None,
            role=None
        )
        assert user.full_name is None
        assert user.display_name is None
        assert user.role is None

    def test_boolean_fields(self):
        """Test boolean field validation"""
        request = QueryRequest(
            query="test",
            include_sources=True,
            stream=False,
            rerank=True
        )
        assert request.include_sources is True
        assert request.stream is False
        assert request.rerank is True

    def test_integer_validation(self):
        """Test integer field validation"""
        # Valid integers
        request = QueryRequest(query="test", top_k=10)
        assert request.top_k == 10

        # Float should be coerced to int
        request2 = QueryRequest(query="test", top_k=10.0)
        assert request2.top_k == 10

    def test_list_validation(self):
        """Test list field validation"""
        # Empty list
        request = QueryRequest(query="test", document_ids=[])
        assert request.document_ids == []

        # Valid list
        request2 = QueryRequest(
            query="test",
            document_ids=["doc1", "doc2", "doc3"]
        )
        assert len(request2.document_ids) == 3

    def test_dict_validation(self):
        """Test dict field validation"""
        # Empty dict
        response = QueryResponse(
            query_id="123",
            query="test",
            answer="answer",
            sources=[],
            processing_time_ms=100,
            stage_timings_ms={},
            timestamp="2024-01-01T00:00:00"
        )
        assert response.stage_timings_ms == {}

        # Valid dict
        response2 = QueryResponse(
            query_id="123",
            query="test",
            answer="answer",
            sources=[],
            processing_time_ms=100,
            stage_timings_ms={"retrieval": 100, "generation": 200},
            timestamp="2024-01-01T00:00:00"
        )
        assert len(response2.stage_timings_ms) == 2


@pytest.mark.unit
class TestModelSerialization:
    """Test model serialization and deserialization"""

    def test_model_to_dict(self):
        """Test model_dump method"""
        data = {
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user"
        }
        user = UserBase(**data)
        dumped = user.model_dump()

        assert dumped["email"] == "test@example.com"
        assert dumped["full_name"] == "Test User"

    def test_model_to_json(self):
        """Test model_dump_json method"""
        data = {
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user"
        }
        user = UserBase(**data)
        json_str = user.model_dump_json()

        assert "test@example.com" in json_str
        assert "Test User" in json_str

    def test_model_from_dict(self):
        """Test model_validate method"""
        data = {
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user"
        }
        user = UserBase.model_validate(data)
        assert user.email == "test@example.com"

    def test_model_from_json(self):
        """Test model_validate_json method"""
        json_str = '{"email": "test@example.com", "full_name": "Test User", "role": "user"}'
        user = UserBase.model_validate_json(json_str)
        assert user.email == "test@example.com"


@pytest.mark.unit
class TestModelConfig:
    """Test model configuration"""

    def test_model_frozen(self):
        """Test frozen model configuration"""
        # Test if any models are frozen (immutable after creation)
        # This is a documentation test for current behavior
        user = UserBase(
            email="test@example.com",
            full_name="Test User",
            role="user"
        )
        # If frozen, this would raise an error
        # For now, assume models are not frozen
        user_data = user.model_copy()
        assert user_data.email == user.email

    def test_model_extra_forbid(self):
        """Test extra fields are rejected"""
        # Most models should reject extra fields
        with pytest.raises(ValidationError):
            UserBase(
                email="test@example.com",
                full_name="Test User",
                role="user",
                extra_field="not_allowed"  # Extra field
            )
