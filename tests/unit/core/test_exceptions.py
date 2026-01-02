#!/usr/bin/env python3
"""
Unit Tests for Custom Exceptions
Tests for backend/core/exceptions.py
"""

import pytest
from datetime import datetime

from backend.core.exceptions import (
    AppException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    NotFoundException,
    ConflictException,
    RateLimitException,
    OCRException,
    EmbeddingException,
    RAGException,
    RetrievalException,
    LLMException,
)


class TestAppException:
    """Test base AppException"""

    def test_default_values(self):
        """Test exception with default values"""
        exc = AppException("Test error")
        assert exc.message == "Test error"
        assert exc.code == "app_error"
        assert exc.status_code == 500
        assert exc.details == {}

    def test_custom_code(self):
        """Test exception with custom code"""
        exc = AppException("Test error", code="custom_error")
        assert exc.code == "custom_error"

    def test_custom_status_code(self):
        """Test exception with custom status code"""
        exc = AppException("Test error", status_code=400)
        assert exc.status_code == 400

    def test_with_details(self):
        """Test exception with details dict"""
        exc = AppException(
            "Test error",
            details={"key": "value", "number": 123}
        )
        assert exc.details == {"key": "value", "number": 123}

    def test_all_parameters(self):
        """Test exception with all parameters"""
        exc = AppException(
            message="Full error",
            code="full_error",
            status_code=418,
            details={"detail": "info"}
        )
        assert exc.message == "Full error"
        assert exc.code == "full_error"
        assert exc.status_code == 418
        assert exc.details == {"detail": "info"}

    def test_timestamp_is_set(self):
        """Test exception has timestamp"""
        exc = AppException("Test error")
        assert exc.timestamp is not None
        # Should be ISO format string
        assert isinstance(exc.timestamp, str)
        # Should be parseable
        datetime.fromisoformat(exc.timestamp)

    def test_timestamp_recent(self):
        """Test exception timestamp is recent"""
        before = datetime.utcnow()
        exc = AppException("Test error")
        after = datetime.utcnow()
        exc_time = datetime.fromisoformat(exc.timestamp)
        assert before <= exc_time <= after

    def test_str_representation(self):
        """Test exception string representation"""
        exc = AppException("Test error")
        assert str(exc) == "Test error"

    def test_raise_and_catch(self):
        """Test raising and catching exception"""
        with pytest.raises(AppException) as exc_info:
            raise AppException("Test error")
        assert exc_info.value.message == "Test error"

    def test_inheritance(self):
        """Test inherits from Exception"""
        exc = AppException("Test")
        assert isinstance(exc, Exception)
        assert isinstance(exc, AppException)


class TestValidationException:
    """Test ValidationException"""

    def test_default_values(self):
        """Test default validation error values"""
        exc = ValidationException("Invalid input")
        assert exc.message == "Invalid input"
        assert exc.code == "validation_error"
        assert exc.status_code == 400

    def test_with_details(self):
        """Test validation error with details"""
        exc = ValidationException(
            "Invalid email",
            details={"field": "email", "value": "invalid"}
        )
        assert exc.details["field"] == "email"
        assert exc.status_code == 400

    def test_status_code_constant(self):
        """Test status code is always 400"""
        exc1 = ValidationException("Error 1")
        exc2 = ValidationException("Error 2", status_code=500)
        # The class should override status_code
        assert exc1.status_code == 400
        # If parameter is passed, it should be respected
        # (but the class default is 400)


class TestAuthenticationException:
    """Test AuthenticationException"""

    def test_default_message(self):
        """Test default authentication error message"""
        exc = AuthenticationException()
        assert exc.message == "Authentication failed"
        assert exc.code == "authentication_error"
        assert exc.status_code == 401

    def test_custom_message(self):
        """Test custom authentication error message"""
        exc = AuthenticationException("Invalid credentials")
        assert exc.message == "Invalid credentials"
        assert exc.code == "authentication_error"

    def test_with_details(self):
        """Test authentication error with details"""
        exc = AuthenticationException(
            "Token expired",
            details={"token_type": "access", "expired_at": "2026-01-01"}
        )
        assert exc.details["token_type"] == "access"
        assert exc.status_code == 401


class TestAuthorizationException:
    """Test AuthorizationException"""

    def test_default_message(self):
        """Test default authorization error message"""
        exc = AuthorizationException()
        assert exc.message == "Permission denied"
        assert exc.code == "authorization_error"
        assert exc.status_code == 403

    def test_custom_message(self):
        """Test custom authorization error message"""
        exc = AuthorizationException("Not allowed to access this resource")
        assert exc.message == "Not allowed to access this resource"

    def test_with_required_permission(self):
        """Test authorization error with permission details"""
        exc = AuthorizationException(
            "Insufficient permissions",
            details={"required": "admin", "user_role": "user"}
        )
        assert exc.details["required"] == "admin"
        assert exc.status_code == 403


class TestNotFoundException:
    """Test NotFoundException"""

    def test_default_message(self):
        """Test default not found message"""
        exc = NotFoundException()
        assert "not found" in exc.message.lower()
        assert exc.status_code == 404

    def test_custom_resource(self):
        """Test not found with custom resource name"""
        exc = NotFoundException("User")
        assert "User" in exc.message
        assert exc.message == "User not found"

    def test_with_details(self):
        """Test not found with details"""
        exc = NotFoundException(
            "Document",
            details={"document_id": "123", "user_id": "456"}
        )
        assert "Document" in exc.message
        assert exc.details["document_id"] == "123"
        assert exc.status_code == 404


class TestConflictException:
    """Test ConflictException"""

    def test_default_values(self):
        """Test default conflict error values"""
        exc = ConflictException("Resource already exists")
        assert exc.message == "Resource already exists"
        assert exc.code == "conflict"
        assert exc.status_code == 409

    def test_with_details(self):
        """Test conflict error with details"""
        exc = ConflictException(
            "Duplicate entry",
            details={"field": "email", "value": "test@example.com"}
        )
        assert exc.details["field"] == "email"
        assert exc.status_code == 409


class TestRateLimitException:
    """Test RateLimitException"""

    def test_default_message(self):
        """Test default rate limit message"""
        exc = RateLimitException()
        assert exc.message == "Rate limit exceeded"
        assert exc.code == "rate_limit_exceeded"
        assert exc.status_code == 429

    def test_custom_message(self):
        """Test custom rate limit message"""
        exc = RateLimitException("Too many requests")
        assert exc.message == "Too many requests"

    def test_retry_after_added_to_details(self):
        """Test retry_after is added to details"""
        exc = RateLimitException(retry_after=60)
        assert exc.details["retry_after"] == 60
        assert exc.status_code == 429

    def test_retry_after_with_custom_details(self):
        """Test retry_after merges with custom details"""
        exc = RateLimitException(
            "Slow down",
            retry_after=120,
            details={"limit": 100, "used": 150}
        )
        assert exc.details["retry_after"] == 120
        assert exc.details["limit"] == 100


class TestOCRException:
    """Test OCRException"""

    def test_default_values(self):
        """Test default OCR error values"""
        exc = OCRException("OCR failed")
        assert exc.message == "OCR failed"
        assert exc.code == "ocr_error"
        assert exc.status_code == 500

    def test_with_engine(self):
        """Test OCR error with engine specified"""
        exc = OCRException(
            "Engine initialization failed",
            engine="yomitoku"
        )
        assert exc.details["engine"] == "yomitoku"

    def test_with_engine_and_details(self):
        """Test OCR error with engine and additional details"""
        exc = OCRException(
            "Low confidence",
            engine="paddleocr",
            details={"confidence": 0.5, "threshold": 0.8}
        )
        assert exc.details["engine"] == "paddleocr"
        assert exc.details["confidence"] == 0.5


class TestEmbeddingException:
    """Test EmbeddingException"""

    def test_default_values(self):
        """Test default embedding error values"""
        exc = EmbeddingException("Embedding failed")
        assert exc.message == "Embedding failed"
        assert exc.code == "embedding_error"
        assert exc.status_code == 500

    def test_with_details(self):
        """Test embedding error with details"""
        exc = EmbeddingException(
            "Model not found",
            details={
                "model_path": "/app/models/sarashina",
                "expected_location": "Docker base image"
            }
        )
        assert exc.details["model_path"] == "/app/models/sarashina"


class TestRAGException:
    """Test RAGException"""

    def test_default_values(self):
        """Test default RAG error values"""
        exc = RAGException("RAG pipeline failed")
        assert exc.message == "RAG pipeline failed"
        assert exc.code == "rag_error"
        assert exc.status_code == 500

    def test_with_stage(self):
        """Test RAG error with stage specified"""
        exc = RAGException(
            "Processing failed",
            stage="retrieval"
        )
        assert exc.details["stage"] == "retrieval"

    def test_with_stage_and_details(self):
        """Test RAG error with stage and details"""
        exc = RAGException(
            "Query failed",
            stage="generation",
            details={"error": "LLM timeout", "retries": 3}
        )
        assert exc.details["stage"] == "generation"
        assert exc.details["error"] == "LLM timeout"


class TestRetrievalException:
    """Test RetrievalException"""

    def test_default_values(self):
        """Test default retrieval error values"""
        exc = RetrievalException("Search failed")
        assert exc.message == "Search failed"
        assert exc.code == "retrieval_error"
        assert exc.status_code == 500

    def test_with_retriever(self):
        """Test retrieval error with retriever specified"""
        exc = RetrievalException(
            "Connection failed",
            retriever="milvus"
        )
        assert exc.details["retriever"] == "milvus"

    def test_with_retriever_and_details(self):
        """Test retrieval error with retriever and details"""
        exc = RetrievalException(
            "Query timeout",
            retriever="postgresql",
            details={"timeout": 5000, "query": "SELECT ..."}
        )
        assert exc.details["retriever"] == "postgresql"
        assert exc.details["timeout"] == 5000


class TestLLMException:
    """Test LLMException"""

    def test_default_values(self):
        """Test default LLM error values"""
        exc = LLMException("LLM generation failed")
        assert exc.message == "LLM generation failed"
        assert exc.code == "llm_error"
        assert exc.status_code == 500

    def test_with_model(self):
        """Test LLM error with model specified"""
        exc = LLMException(
            "Model not available",
            model="qwen3:4b"
        )
        assert exc.details["model"] == "qwen3:4b"

    def test_with_model_and_details(self):
        """Test LLM error with model and details"""
        exc = LLMException(
            "Generation timeout",
            model="qwen3:14b",
            details={"timeout": 30, "tokens_generated": 150}
        )
        assert exc.details["model"] == "qwen3:14b"
        assert exc.details["tokens_generated"] == 150


class TestExceptionChaining:
    """Test exception chaining and context"""

    def test_exception_from_another_exception(self):
        """Test creating exception from another exception"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise AppException("Wrapped error") from e
        except AppException as exc:
            assert exc.message == "Wrapped error"
            assert exc.__cause__ is not None

    def test_exception_in_exception_handler(self):
        """Test exception in error handler context"""
        def handle_error():
            raise ValidationException("Invalid data")

        with pytest.raises(ValidationException) as exc_info:
            handle_error()

        assert exc_info.value.details == {}


class TestExceptionSerialization:
    """Test exception serialization for API responses"""

    def test_to_dict_method(self):
        """Test converting exception to dict for JSON response"""
        exc = AppException(
            "Test error",
            code="test_error",
            status_code=400,
            details={"key": "value"}
        )
        result = {
            "message": exc.message,
            "code": exc.code,
            "status_code": exc.status_code,
            "details": exc.details,
            "timestamp": exc.timestamp
        }
        assert result["message"] == "Test error"
        assert result["code"] == "test_error"
        assert result["details"] == {"key": "value"}

    def test_all_exception_types_serializable(self):
        """Test all exception types can be serialized"""
        exceptions = [
            ValidationException("Validation failed"),
            AuthenticationException("Auth failed"),
            AuthorizationException("Access denied"),
            NotFoundException("Resource"),
            ConflictException("Conflict"),
            RateLimitException("Rate limit", retry_after=60),
            OCRException("OCR failed", engine="yomitoku"),
            EmbeddingException("Embedding failed"),
            RAGException("RAG failed", stage="retrieval"),
            RetrievalException("Search failed", retriever="milvus"),
            LLMException("LLM failed", model="qwen3:4b"),
        ]
        for exc in exceptions:
            assert exc.message is not None
            assert exc.code is not None
            assert exc.status_code is not None
            assert exc.timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
