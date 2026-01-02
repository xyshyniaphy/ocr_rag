"""
Pytest Configuration and Fixtures
Shared fixtures and configuration for all tests
"""

import os
import sys
import asyncio
import pytest
import uuid
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.core.config import settings
from backend.db.session import async_session_maker, engine


# ============================================
# PYTEST CONFIGURATION
# ============================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (medium speed)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (slow)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "external: Tests requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Add markers based on test file path
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add gpu marker to GPU-specific tests
        if item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.slow)


# ============================================
# ASYNCIO CONFIGURATION
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# HTTP CLIENT FIXTURES
# ============================================

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Test HTTP client for FastAPI app
    Uses ASGI transport for testing without running server
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def auth_headers(client: AsyncClient, test_user: dict) -> dict:
    """
    Authentication headers for API requests
    Creates a test user and returns JWT token in headers
    """
    # Register user
    response = client.post(
        "/api/v1/auth/register",
        json=test_user
    )
    if response.status_code not in [200, 201]:
        # User might already exist, try logging in
        response = client.post(
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


# ============================================
# DATABASE FIXTURES
# ============================================

@pytest.fixture
async def db_session():
    """
    Test database session
    Creates a new session for each test and rolls back changes
    """
    async with async_session_maker() as session:
        yield session
        # Rollback all changes after test
        await session.rollback()


@pytest.fixture
async def clean_db(db_session):
    """
    Clean database before and after tests
    Removes all test data
    """
    # Clean before test
    await db_session.execute("DELETE FROM backend.models.chunk")
    await db_session.execute("DELETE FROM backend.db.models.Query")
    await db_session.execute("DELETE FROM backend.db.models.Document")
    await db_session.execute("DELETE FROM backend.db.models.User")

    yield

    # Clean after test
    await db_session.execute("DELETE FROM backend.models.chunk")
    await db_session.execute("DELETE FROM backend.db.models.Query")
    await db_session.execute("DELETE FROM backend.db.models.Document")
    await db_session.execute("DELETE FROM backend.db.models.User")


@pytest.fixture
async def test_user_data(db_session):
    """
    Create test user in database
    Returns user data dictionary
    """
    user_data = {
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "password": "test_password_123",
        "full_name": "Test User"
    }
    return user_data


# ============================================
# TEST DATA FIXTURES
# ============================================

@pytest.fixture
def test_user():
    """
    Test user data for authentication
    Uses unique email to avoid conflicts
    """
    return {
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "password": "test_password_123",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_text_japanese():
    """Sample Japanese text for testing"""
    return "これはテストテキストです。日本語の埋め込みをテストしています。"


@pytest.fixture
def sample_texts_japanese():
    """Sample Japanese texts for batch testing"""
    return [
        "最初のテキストです。",
        "二番目のテキストです。少し長めの文章を用意しています。",
        "人工知能は現代の技術において非常に重要な役割を果たしています。",
        "自然言語処理はAIの一分野です。",
    ]


@pytest.fixture
def sample_pdf():
    """
    Sample PDF bytes for testing
    Loads from test fixtures directory
    """
    pdf_path = Path(__file__).parent / "sample.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            return f.read()

    # Return minimal PDF if no file exists
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    # Return normalized vector of correct dimension
    import numpy as np
    vec = np.random.rand(1792)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def sample_query():
    """Sample query for RAG testing"""
    return "ドキュメントの内容について教えて"


# ============================================
# MOCK FIXTURES
# ============================================

@pytest.fixture
def mock_llm_response():
    """
    Mock LLM response
    Returns a mock generate function
    """
    async def mock_generate(text: str, context: list = None, **kwargs):
        from backend.services.llm.schema import LLMResponse
        return LLMResponse(
            text=f"Mocked response for: {text[:50]}...",
            tokens=100,
            model="mock-model",
            processing_time_ms=50
        )
    return mock_generate


@pytest.fixture
def mock_ocr_result():
    """
    Mock OCR result
    Returns a mock OCR processing result
    """
    from backend.services.ocr.schema import OCRResult, OCRPage
    return OCRResult(
        total_pages=1,
        markdown="## Extracted Text\n\nThis is sample text from OCR.",
        confidence=0.95,
        processing_time_ms=1000,
        engine_used="yomitoku",
        pages=[
            OCRPage(
                page_number=1,
                text="Extracted text",
                confidence=0.95,
                blocks=[]
            )
        ],
        warnings=[]
    )


@pytest.fixture
def mock_embedding_result():
    """
    Mock embedding result
    Returns a mock embedding with vector
    """
    from backend.services.embedding.schema import EmbeddingResult, EmbeddingVector
    import numpy as np

    vec = np.random.rand(1792)
    vec = vec / np.linalg.norm(vec)

    return EmbeddingResult(
        embedding=EmbeddingVector(
            vector=vec.tolist(),
            text_hash="abc123",
            dimension=1792
        ),
        token_count=50,
        processing_time_ms=100
    )


@pytest.fixture
def mock_search_results():
    """
    Mock search results
    Returns mock document chunks from vector search
    """
    from backend.services.retrieval.schema import SearchResult
    return [
        SearchResult(
            chunk_id="chunk1",
            document_id="doc1",
            text="First result text",
            score=0.95,
            metadata={"page": 1}
        ),
        SearchResult(
            chunk_id="chunk2",
            document_id="doc1",
            text="Second result text",
            score=0.85,
            metadata={"page": 2}
        ),
    ]


# ============================================
# SERVICE FIXTURES
# =============================================

@pytest.fixture
async def embedding_service():
    """
    Embedding service fixture
    Initializes the embedding service for testing
    Uses CPU to avoid GPU requirement
    """
    from backend.services.embedding import get_embedding_service

    # Override device to CPU for tests
    with patch.object(settings, "EMBEDDING_DEVICE", "cpu"):
        service = await get_embedding_service()
        yield service
        # Cleanup
        await service.unload()


@pytest.fixture
async def ocr_service():
    """
    OCR service fixture
    Initializes the OCR service for testing
    """
    from backend.services.ocr import OCRService

    service = OCRService(engine="yomitoku")
    yield service
    # Cleanup
    await service.unload_all()


@pytest.fixture
async def reranker_service():
    """
    Reranker service fixture
    Initializes the reranker service for testing
    Uses CPU to avoid GPU requirement
    """
    from backend.services.reranker import get_reranking_service

    # Override device to CPU for tests
    with patch.object(settings, "RERANKER_DEVICE", "cpu"):
        service = await get_reranking_service()
        yield service


# ============================================
# ENVIRONMENT FIXTURES
# ============================================

@pytest.fixture
def test_env_vars():
    """
    Set test environment variables
    Automatically restores original values after test
    """
    original_env = os.environ.copy()

    # Set test environment variables
    test_vars = {
        "ENVIRONMENT": "development",
        "DEBUG": "True",
        "LOG_LEVEL": "DEBUG",
        "EMBEDDING_DEVICE": "cpu",
        "RERANKER_DEVICE": "cpu",
        "OCR_ENGINE": "yomitoku",
    }

    os.environ.update(test_vars)
    yield test_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_config_file(tmp_path):
    """
    Create temporary .env file for testing
    """
    env_file = tmp_path / ".env"
    env_file.write_text("""
ENVIRONMENT=testing
DEBUG=True
SECRET_KEY=test_secret_key_for_testing_only_32chars
POSTGRES_DB=test_db
POSTGRES_USER=test_user
POSTGRES_PASSWORD=test_pass
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
""")
    return env_file


# ============================================
# SKIP FIXTURES
# ============================================

@pytest.fixture
def skip_if_no_gpu():
    """
    Skip test if GPU is not available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def skip_if_no_milvus():
    """
    Skip test if Milvus is not available
    """
    try:
        from pymilvus import connections
        connections.connect("test", host="localhost", port="19530")
        connections.disconnect("test")
    except Exception:
        pytest.skip("Milvus not available")


@pytest.fixture
def skip_if_no_redis():
    """
    Skip test if Redis is not available
    """
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        r.close()
    except Exception:
        pytest.skip("Redis not available")


# ============================================
# PERFORMANCE FIXTURES
# ============================================

@pytest.fixture
def benchmark_timer():
    """
    Timer fixture for performance testing
    Returns a context manager that measures execution time
    """
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

        @property
        def elapsed_ms(self):
            if self.elapsed is not None:
                return self.elapsed * 1000
            return None

    return Timer


# ============================================
# TEST HELPERS
# ============================================

@pytest.fixture
def assert_valid_embedding():
    """
    Helper to validate embedding vectors
    Returns a function that checks embedding properties
    """
    def _assert(vector, expected_dim=1792, normalized=True):
        assert isinstance(vector, list), "Vector must be a list"
        assert len(vector) == expected_dim, f"Vector dimension mismatch: {len(vector)} != {expected_dim}"

        if normalized:
            import numpy as np
            norm = np.linalg.norm(vector)
            assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"

    return _assert


@pytest.fixture
def assert_valid_response():
    """
    Helper to validate API responses
    Returns a function that checks response properties
    """
    def _assert(response, expected_status=200):
        assert response.status_code == expected_status, \
            f"Expected {expected_status}, got {response.status_code}: {response.text}"

        # Check JSON is valid
        if "application/json" in response.headers.get("content-type", ""):
            data = response.json()
            assert isinstance(data, dict), "Response must be a dict"
            return data
        return response.text

    return _assert


# ============================================
# CLEANUP FIXTURES
# ============================================

@pytest.fixture(autouse=True)
def cleanup_test_resources():
    """
    Automatically clean up resources after each test
    Runs automatically for all tests
    """
    yield

    # Force garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
