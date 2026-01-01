"""
Custom Exceptions
Application-specific exception classes
"""

from datetime import datetime
from typing import Any, Dict, Optional


class AppException(Exception):
    """Base application exception"""

    def __init__(
        self,
        message: str,
        code: str = "app_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)


class ValidationException(AppException):
    """Validation error exception"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="validation_error",
            status_code=400,
            details=details,
        )


class AuthenticationException(AppException):
    """Authentication error exception"""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="authentication_error",
            status_code=401,
            details=details,
        )


class AuthorizationException(AppException):
    """Authorization error exception"""

    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="authorization_error",
            status_code=403,
            details=details,
        )


class NotFoundException(AppException):
    """Resource not found exception"""

    def __init__(
        self,
        resource: str = "Resource",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"{resource} not found",
            code="not_found",
            status_code=404,
            details=details,
        )


class ConflictException(AppException):
    """Resource conflict exception"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="conflict",
            status_code=409,
            details=details,
        )


class RateLimitException(AppException):
    """Rate limit exceeded exception"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            status_code=429,
            details=details,
        )


class OCRException(AppException):
    """OCR processing error exception"""

    def __init__(
        self,
        message: str,
        engine: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if engine:
            details["engine"] = engine
        super().__init__(
            message=message,
            code="ocr_error",
            status_code=500,
            details=details,
        )


class EmbeddingException(AppException):
    """Embedding generation error exception"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="embedding_error",
            status_code=500,
            details=details,
        )


class RAGException(AppException):
    """RAG pipeline error exception"""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if stage:
            details["stage"] = stage
        super().__init__(
            message=message,
            code="rag_error",
            status_code=500,
            details=details,
        )


class RetrievalException(AppException):
    """Retrieval/search error exception"""

    def __init__(
        self,
        message: str,
        retriever: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if retriever:
            details["retriever"] = retriever
        super().__init__(
            message=message,
            code="retrieval_error",
            status_code=500,
            details=details,
        )


class LLMException(AppException):
    """LLM generation error exception"""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(
            message=message,
            code="llm_error",
            status_code=500,
            details=details,
        )
