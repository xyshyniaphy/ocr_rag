"""
Conftest for API integration tests
Defines fixtures specific to API testing

Note: These tests use ASGI transport for testing without requiring a running server.
"""

import pytest
import pytest_asyncio
import uuid
import os
import sys
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from backend.main import app
from backend.core.logging import get_logger
from backend.db.session import init_db

logger = get_logger(__name__)

# Mock RAG service instance - created once and reused
_mock_rag_service = None


def get_mock_rag_service():
    """Get or create the mock RAG service"""
    global _mock_rag_service
    if _mock_rag_service is None:
        from backend.services.rag.models import RAGResult, RAGSource, RAGStageMetrics

        mock_result = RAGResult(
            query_id=str(uuid.uuid4()),
            query="Test query",
            answer="This is a test answer from the mock RAG service.",
            sources=[
                RAGSource(
                    chunk_id="test_chunk_001",
                    document_id=str(uuid.uuid4()),
                    document_title="Test Document",
                    page_number=1,
                    chunk_index=0,
                    text="Sample text from document",
                    score=0.95,
                    rerank_score=0.97
                )
            ],
            confidence=0.9,
            processing_time_ms=150,
            stage_timings=[
                RAGStageMetrics(stage_name="retrieval", duration_ms=50, success=True),
                RAGStageMetrics(stage_name="reranking", duration_ms=30, success=True),
                RAGStageMetrics(stage_name="generation", duration_ms=70, success=True),
            ],
            llm_model="test-model",
            embedding_model="test-embedding"
        )

        _mock_rag_service = AsyncMock()
        _mock_rag_service.query = AsyncMock(return_value=mock_result)

    return _mock_rag_service


# Patch get_rag_service at module import level
sys.modules['backend.api.v1.query'].__dict__['get_rag_service'] = lambda: get_mock_rag_service()


@pytest_asyncio.fixture(autouse=True)
async def init_database():
    """
    Auto-initialize database for each test.
    This is safe because init_db() checks if tables already exist.
    """
    await init_db()


@pytest_asyncio.fixture
async def db_session():
    """
    Test database session.
    Creates a new session for each test and rolls back changes.
    """
    from backend.db.session import async_session_maker

    async with async_session_maker() as session:
        yield session
        # Rollback all changes after test
        await session.rollback()


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """
    Test HTTP client using ASGI transport
    Tests the FastAPI app directly without requiring a running server
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict:
    """
    Authentication headers for API requests
    Creates a test user and returns JWT token in headers
    """
    test_user = {
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User"
    }

    # Try to register user
    response = await client.post(
        "/api/v1/auth/register",
        json=test_user
    )

    # If registration fails (user might exist), try login
    if response.status_code not in [200, 201]:
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )

    if response.status_code in [200, 201]:
        token = response.json().get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}

    # Return empty headers if auth fails
    return {}
