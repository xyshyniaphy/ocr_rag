# OCR RAG System - Test Plan

**Version**: 1.0.0
**Date**: 2026-01-02
**Status**: 📋 Planning

---

## Table of Contents

1. [Overview](#overview)
2. [Test Strategy](#test-strategy)
3. [Testing Pyramid](#testing-pyramid)
4. [Unit Tests](#unit-tests)
5. [Integration Tests](#integration-tests)
6. [E2E Tests](#e2e-tests)
7. [Performance Tests](#performance-tests)
8. [Test Data & Fixtures](#test-data--fixtures)
9. [CI/CD Integration](#cicd-integration)
10. [Test Execution](#test-execution)

---

## Overview

This test plan covers the **Japanese OCR RAG System** - a production-grade Retrieval-Augmented Generation system optimized for Japanese PDF document processing.

### Test Scope

| Component | Coverage | Priority |
|-----------|----------|----------|
| **Core** (`backend/core/`) | Config, Logging, Security, Cache | P0 (Critical) |
| **Services** (`backend/services/`) | OCR, Embedding, Reranker, LLM, RAG | P0 (Critical) |
| **API** (`backend/api/`) | REST endpoints, WebSocket | P1 (High) |
| **Database** (`backend/db/`) | PostgreSQL, Milvus repositories | P1 (High) |
| **Tasks** (`backend/tasks/`) | Celery background jobs | P2 (Medium) |
| **Storage** (`backend/storage/`) | MinIO client | P2 (Medium) |

### Out of Scope

- Third-party library testing (HuggingFace, Ollama, etc.)
- Infrastructure testing (Docker, Kubernetes)
- Load/Stress testing beyond performance targets

---

## Test Strategy

### Testing Principles

1. **Fast Feedback**: Unit tests should run in < 5 seconds
2. **Isolation**: Each test should be independent
3. **Deterministic**: No random data or external dependencies in unit tests
4. **Maintainable**: Clear test names, DRY principles
5. **Realistic**: Use production-like data in integration tests

### Test Framework

| Component | Tool | Version |
|-----------|------|---------|
| **Test Runner** | pytest | 7.4.0 |
| **Async Support** | pytest-asyncio | 0.21.0 |
| **Coverage** | pytest-cov | 4.1.0 |
| **Fixtures** | pytest fixtures | builtin |
| **Mocking** | pytest-mock | 3.11.1 |
| **HTTP Testing** | httpx | 0.25.0 |
| **Database** | pytest-postgresql | 5.0.0 |

### Test Categories

```
tests/
├── unit/           # Fast, isolated tests (<5s)
├── integration/    # Service integration tests (<30s)
├── e2e/            # Full pipeline tests (<2min)
├── performance/    # Performance benchmarks
├── manual/         # Manual test scripts (existing)
└── fixtures/       # Test data and fixtures
```

---

## Testing Pyramid

```
                    /\
                   /  \
                  / E2E \          10 tests (slow, expensive)
                 /------\
                /        \
               / INTEGRATION\    100 tests (medium speed)
              /--------------\
             /                \
            /    UNIT TESTS    \  1000+ tests (fast, cheap)
           /--------------------\
```

### Target Metrics

| Level | Count | Duration | Coverage |
|-------|-------|----------|----------|
| **Unit** | 1000+ | <5s | 90%+ |
| **Integration** | 100 | <30s | 70%+ |
| **E2E** | 10 | <2min | Critical paths |
| **Overall** | 1100+ | <2min | 80%+ |

---

## Unit Tests

### Core Components (`tests/unit/core/`)

#### 1. Configuration Tests (`test_config.py`)

```python
# Test cases for backend/core/config.py

class TestSettings:
    """Test Settings configuration"""

    def test_default_settings():
        """Test default values are set correctly"""
        assert settings.APP_NAME == "Japanese OCR RAG System"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.ENVIRONMENT in ["development", "staging", "production"]

    def test_postgres_url_property():
        """Test POSTGRES_URL property construction"""
        expected = f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        assert settings.POSTGRES_URL == expected

    def test_environment_validation():
        """Test ENVIRONMENT validator"""
        with pytest.raises(ValueError):
            Settings(ENVIRONMENT="invalid")

    def test_log_level_validation():
        """Test LOG_LEVEL validator"""
        with pytest.raises(ValueError):
            Settings(LOG_LEVEL="INVALID")

    def test_secret_key_min_length():
        """Test SECRET_KEY minimum length"""
        with pytest.raises(ValidationError):
            Settings(SECRET_KEY="short")

    def test_cors_origins_default():
        """Test CORS_ORIGINS defaults"""
        assert "http://localhost:8501" in settings.CORS_ORIGINS
        assert "http://localhost:8000" in settings.CORS_ORIGINS
```

#### 2. Exception Tests (`test_exceptions.py`)

```python
# Test cases for backend/core/exceptions.py

class TestAppException:
    """Test base exception"""

    def test_exception_creation():
        """Test creating base exception"""
        exc = AppException("Test error")
        assert exc.message == "Test error"
        assert exc.code == "app_error"
        assert exc.status_code == 500

    def test_exception_with_details():
        """Test exception with details"""
        exc = AppException(
            "Test error",
            details={"key": "value"}
        )
        assert exc.details == {"key": "value"}

    def test_exception_timestamp():
        """Test exception has timestamp"""
        exc = AppException("Test error")
        assert exc.timestamp is not None


class TestValidationException:
    """Test validation exception"""

    def test_status_code():
        """Test status code is 400"""
        exc = ValidationException("Invalid input")
        assert exc.status_code == 400
        assert exc.code == "validation_error"


class TestAuthenticationException:
    """Test authentication exception"""

    def test_status_code():
        """Test status code is 401"""
        exc = AuthenticationException()
        assert exc.status_code == 401
        assert exc.code == "authentication_error"


class TestNotFoundException:
    """Test not found exception"""

    def test_message_format():
        """Test message includes resource name"""
        exc = NotFoundException("User")
        assert "User" in exc.message
        assert exc.status_code == 404


class TestRateLimitException:
    """Test rate limit exception"""

    def test_retry_after_in_details():
        """Test retry_after is added to details"""
        exc = RateLimitException(retry_after=60)
        assert exc.details["retry_after"] == 60
        assert exc.status_code == 429
```

#### 3. Logging Tests (`test_logging.py`)

```python
# Test cases for backend/core/logging.py

class TestLogging:
    """Test logging configuration"""

    def test_logger_creation():
        """Test logger is created"""
        logger = get_logger(__name__)
        assert logger is not None

    def test_log_levels():
        """Test different log levels"""
        logger = get_logger("test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_log_format():
        """Test log format includes required fields"""
        # Capture logs and verify format
        pass

    def test_contextual_logging():
        """Test contextual logging with extra fields"""
        logger = get_logger("test", user_id="123", request_id="456")
        logger.info("Test with context")
```

#### 4. Security Tests (`test_security.py`)

```python
# Test cases for backend/core/security.py

class TestPasswordHashing:
    """Test password hashing utilities"""

    def test_hash_password():
        """Test password hashing"""
        password = "test_password"
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > 50

    def test_verify_password():
        """Test password verification"""
        password = "test_password"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True
        assert verify_password("wrong", hashed) is False


class TestJWT:
    """Test JWT token utilities"""

    def test_create_access_token():
        """Test access token creation"""
        data = {"sub": "user@example.com"}
        token = create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 50

    def test_decode_access_token():
        """Test token decoding"""
        data = {"sub": "user@example.com"}
        token = create_access_token(data)
        decoded = decode_access_token(token)
        assert decoded["sub"] == "user@example.com"

    def test_token_expiration():
        """Test token expires after time"""
        # Create token with very short expiration
        pass


class TestSanitization:
    """Test input sanitization"""

    def test_sanitize_html():
        """Test HTML sanitization"""
        dirty = "<script>alert('xss')</script>"
        clean = sanitize_html(dirty)
        assert "<script>" not in clean

    def test_sanitize_sql():
        """Test SQL injection prevention"""
        pass
```

#### 5. Cache Tests (`test_cache.py`)

```python
# Test cases for backend/core/cache.py

class TestCache:
    """Test caching utilities"""

    @pytest.mark.asyncio
    async def test_cache_set_get():
        """Test setting and getting cache values"""
        await cache.set("key", "value", ttl=60)
        value = await cache.get("key")
        assert value == "value"

    @pytest.mark.asyncio
    async def test_cache_expiration():
        """Test cache expires after TTL"""
        await cache.set("key", "value", ttl=1)
        await asyncio.sleep(2)
        value = await cache.get("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_delete():
        """Test deleting cache values"""
        await cache.set("key", "value")
        await cache.delete("key")
        value = await cache.get("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_clear():
        """Test clearing all cache"""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
```

### Services Unit Tests (`tests/unit/services/`)

#### 1. Embedding Service Tests (`test_embedding.py`)

```python
# Test cases for backend/services/embedding/sarashina.py

class TestSarashinaEmbeddingModel:
    """Test Sarashina embedding model"""

    @pytest.fixture
    def model(self):
        """Create model instance for testing"""
        return SarashinaEmbeddingModel(
            model_path="/app/models/sarashina",
            device="cpu"  # Use CPU for tests
        )

    def test_model_initialization(self, model):
        """Test model initializes correctly"""
        assert model.model_name == "sbintuitions/sarashina-embedding-v1-1b"
        assert model.dimension == 1792
        assert model.max_length == 512

    def test_embed_single_text(self, model):
        """Test embedding single text"""
        text = "これはテストです"
        result = model.embed(text)
        assert len(result.vector) == 1792
        assert result.token_count > 0

    def test_embed_batch_texts(self, model):
        """Test embedding multiple texts"""
        texts = ["テキスト1", "テキスト2", "テキスト3"]
        results = model.embed_batch(texts)
        assert len(results) == 3
        assert all(len(r.vector) == 1792 for r in results)

    def test_text_truncation(self, model):
        """Test long texts are truncated"""
        long_text = "あ" * 1000
        result = model.embed(long_text)
        assert result.token_count <= model.max_length

    def test_empty_text_handling(self, model):
        """Test empty text is handled"""
        with pytest.raises(ValidationException):
            model.embed("")


class TestEmbeddingService:
    """Test embedding service"""

    @pytest.mark.asyncio
    async def test_embed_text_caching():
        """Test embedding results are cached"""
        service = await get_embedding_service()
        text = "テスト"

        result1 = await service.embed_text(text)
        result2 = await service.embed_text(text)

        assert result1.embedding.text_hash == result2.embedding.text_hash

    @pytest.mark.asyncio
    async def test_embed_chunks():
        """Test embedding document chunks"""
        service = await get_embedding_service()
        chunks = {
            "chunk1": "テキスト1",
            "chunk2": "テキスト2"
        }

        result = await service.embed_chunks(chunks, "doc1")
        assert result.total_chunks == 2
        assert result.dimension == 1792
```

#### 2. OCR Service Tests (`test_ocr.py`)

```python
# Test cases for backend/services/ocr/

class TestOCRService:
    """Test OCR service"""

    @pytest.fixture
    def sample_pdf(self):
        """Load sample PDF for testing"""
        with open("tests/fixtures/sample.pdf", "rb") as f:
            return f.read()

    def test_get_available_engines(self):
        """Test getting available OCR engines"""
        engines = OCRService.get_available_engines()
        assert "yomitoku" in engines

    def test_engine_selection():
        """Test selecting OCR engine"""
        service = OCRService(engine="yomitoku")
        assert service.engine == "yomitoku"

    @pytest.mark.asyncio
    async def test_process_pdf(self, sample_pdf):
        """Test PDF processing"""
        result = await OCRService.process_pdf(sample_pdf)
        assert result.total_pages > 0
        assert result.markdown
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_confidence_threshold(self, sample_pdf):
        """Test confidence threshold filtering"""
        options = OCROptions(confidence_threshold=0.99)
        result = await OCRService.process_pdf(sample_pdf, options=options)
        # Should trigger fallback or fail if below threshold


class TestYomiTokuEngine:
    """Test YomiToku OCR engine"""

    def test_engine_initialization():
        """Test YomiToku engine initializes"""
        engine = YomiTokuEngine()
        assert engine.name == "yomitoku"

    def test_supports_table_extraction():
        """Test table extraction support"""
        assert YomiTokuEngine.SUPPORTS_TABLES is True
```

#### 3. Reranker Service Tests (`test_reranker.py`)

```python
# Test cases for backend/services/reranker/llama_nv.py

class TestLlamaNVReranker:
    """Test Llama NV reranker"""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance"""
        return LlamaNVReranker(device="cpu")

    def test_initialization(self, reranker):
        """Test reranker initializes"""
        assert reranker.model_name == "nvidia/Llama-3.2-NV-RerankQA-1B-v2"

    def test_rerank_simple(self, reranker):
        """Test simple reranking"""
        query = "テスト"
        texts = ["関連するテキスト", "無関係なテキスト"]
        results = reranker.rerank_simple(query, texts, top_k=2)
        assert len(results) <= 2
        assert all("score" in r for r in results)

    def test_rerank_empty_texts(self, reranker):
        """Test reranking with empty texts"""
        with pytest.raises(ValidationException):
            reranker.rerank_simple("query", [])


class TestRerankerService:
    """Test reranker service"""

    @pytest.mark.asyncio
    async def test_service_initialization():
        """Test service initializes correctly"""
        service = await get_reranking_service()
        health = await service.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_rerank_results():
        """Test reranking search results"""
        service = await get_reranking_service()
        query = "テスト"
        results = [
            SearchResult(text="関連テキスト1", score=0.8),
            SearchResult(text="関連テキスト2", score=0.7)
        ]
        reranked = await service.rerank(query, results)
        assert len(reranked) <= len(results)
```

#### 4. LLM Service Tests (`test_llm.py`)

```python
# Test cases for backend/services/llm/

class TestOllamaLLM:
    """Test Ollama LLM client"""

    def test_initialization():
        """Test Ollama client initializes"""
        llm = OllamaLLM(model="qwen3:4b")
        assert llm.model == "qwen3:4b"

    @pytest.mark.asyncio
    async def test_generate():
        """Test text generation"""
        llm = OllamaLLM(model="qwen3:4b")
        response = await llm.generate("テスト")
        assert response.text
        assert response.tokens > 0

    @pytest.mark.asyncio
    async def test_generate_with_context():
        """Test generation with context"""
        llm = OllamaLLM(model="qwen3:4b")
        context = ["コンテキスト1", "コンテキスト2"]
        response = await llm.generate("テスト", context=context)
        assert response.text


class TestLLMService:
    """Test LLM service"""

    @pytest.mark.asyncio
    async def test_health_check():
        """Test Ollama health check"""
        service = await get_llm_service()
        health = await service.health_check()
        assert health["status"] in ["healthy", "degraded"]
```

### Database Unit Tests (`tests/unit/db/`)

#### 1. Repository Tests (`test_repositories.py`)

```python
# Test cases for backend/db/repositories/

class TestDocumentRepository:
    """Test document repository"""

    @pytest.mark.asyncio
    async def test_create_document(self, db_session):
        """Test creating a document"""
        repo = DocumentRepository(db_session)
        doc = await repo.create(
            filename="test.pdf",
            user_id="user123"
        )
        assert doc.id
        assert doc.filename == "test.pdf"

    @pytest.mark.asyncio
    async def test_get_document(self, db_session):
        """Test getting a document"""
        repo = DocumentRepository(db_session)
        doc = await repo.get("doc123")
        assert doc is not None

    @pytest.mark.asyncio
    async def test_list_documents(self, db_session):
        """Test listing user's documents"""
        repo = DocumentRepository(db_session)
        docs = await repo.list_by_user("user123", limit=10)
        assert len(docs) <= 10


class TestChunkRepository:
    """Test chunk repository"""

    @pytest.mark.asyncio
    async def test_create_chunk(self, db_session):
        """Test creating a chunk"""
        repo = ChunkRepository(db_session)
        chunk = await repo.create(
            document_id="doc123",
            text="テストチャンク"
        )
        assert chunk.id

    @pytest.mark.asyncio
    async def test_get_chunks_by_document(self, db_session):
        """Test getting chunks by document"""
        repo = ChunkRepository(db_session)
        chunks = await repo.get_by_document("doc123")
        assert isinstance(chunks, list)
```

### API Unit Tests (`tests/unit/api/`)

#### 1. Route Tests (`test_routes.py`)

```python
# Test cases for backend/api/v1/

class TestHealthRoutes:
    """Test health check routes"""

    def test_health_endpoint(client):
        """Test GET /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAuthRoutes:
    """Test authentication routes"""

    def test_register_success(client, test_user):
        """Test user registration"""
        response = client.post("/api/v1/auth/register", json=test_user)
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data

    def test_login_success(client, test_user):
        """Test user login"""
        response = client.post("/api/v1/auth/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })
        assert response.status_code == 200


class TestDocumentRoutes:
    """Test document routes"""

    @pytest.mark.asyncio
    async def test_upload_document(client, auth_headers, sample_pdf):
        """Test document upload"""
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files={"file": ("test.pdf", sample_pdf, "application/pdf")}
        )
        assert response.status_code == 201

    def test_list_documents(client, auth_headers):
        """Test listing documents"""
        response = client.get("/api/v1/documents", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
```

---

## Integration Tests

### Service Integration (`tests/integration/services/`)

#### 1. OCR + Embedding Pipeline (`test_ocr_embedding_pipeline.py`)

```python
class TestOCREmbeddingPipeline:
    """Test OCR to embedding pipeline"""

    @pytest.mark.asyncio
    async def test_pdf_to_embeddings():
        """Test converting PDF to embeddings"""
        # Load PDF
        pdf_bytes = load_test_pdf()

        # OCR
        ocr_result = await ocr_pdf(pdf_bytes)

        # Chunk
        chunks = chunk_text(ocr_result.markdown)

        # Embed
        service = await get_embedding_service()
        embeddings = await service.embed_chunks(chunks, "doc1")

        assert embeddings.total_chunks > 0
        assert embeddings.dimension == 1792
```

#### 2. Full RAG Pipeline (`test_rag_pipeline.py`)

```python
class TestRAGPipeline:
    """Test full RAG pipeline"""

    @pytest.mark.asyncio
    async def test_query_pipeline():
        """Test query from start to finish"""
        query = "テスト"

        # Embed query
        service = await get_embedding_service()
        query_emb = await service.embed_text(query)

        # Search
        results = await search_service.search(
            query_emb.vector,
            top_k=20
        )

        # Rerank
        reranked = await reranker_service.rerank(query, results)

        # Generate
        response = await llm_service.generate(
            query,
            [r.text for r in reranked]
        )

        assert response.text
        assert response.sources
```

### Database Integration (`tests/integration/db/`)

#### 1. Milvus Integration (`test_milvus.py`)

```python
class TestMilvusIntegration:
    """Test Milvus vector database"""

    @pytest.mark.asyncio
    async def test_insert_and_search():
        """Test inserting vectors and searching"""
        client = get_milvus_client()

        # Insert
        vectors = [[0.1] * 1792, [0.2] * 1792]
        await client.insert("document_chunks", vectors)

        # Search
        results = await client.search([0.15] * 1792, limit=10)
        assert len(results) > 0
```

#### 2. PostgreSQL Integration (`test_postgres.py`)

```python
class TestPostgresIntegration:
    """Test PostgreSQL database"""

    @pytest.mark.asyncio
    async def test_crud_operations():
        """Test CRUD operations"""
        async with get_db_session() as session:
            # Create
            doc = Document(filename="test.pdf")
            session.add(doc)
            await session.commit()

            # Read
            result = await session.get(Document, doc.id)
            assert result.filename == "test.pdf"

            # Update
            result.status = "completed"
            await session.commit()

            # Delete
            await session.delete(result)
            await session.commit()
```

---

## E2E Tests

### API E2E Tests (`tests/e2e/api/`)

#### 1. Document Upload Flow (`test_document_upload_flow.py`)

```python
class TestDocumentUploadFlow:
    """Test complete document upload flow"""

    @pytest.mark.asyncio
    async def test_upload_to_completion():
        """Test upload -> OCR -> embedding -> complete"""
        # 1. Login
        token = await login_user()

        # 2. Upload PDF
        doc_id = await upload_pdf(token, "test.pdf")

        # 3. Wait for processing
        await wait_for_status(doc_id, "completed", timeout=60)

        # 4. Verify chunks created
        chunks = await get_chunks(token, doc_id)
        assert len(chunks) > 0

        # 5. Verify embeddings created
        assert all(c.embedding_id for c in chunks)
```

#### 2. Query Flow (`test_query_flow.py`)

```python
class TestQueryFlow:
    """Test complete query flow"""

    @pytest.mark.asyncio
    async def test_query_with_uploaded_document():
        """Test query against uploaded document"""
        # 1. Upload document
        token = await login_user()
        doc_id = await upload_and_wait(token, "test.pdf")

        # 2. Query
        response = await query_rag(
            token,
            "ドキュメントの内容について教えて"
        )

        # 3. Verify response
        assert response.answer
        assert response.sources
        assert any(s.document_id == doc_id for s in response.sources)
```

---

## Performance Tests

### Benchmarks (`tests/performance/`)

#### 1. Embedding Performance (`test_embedding_performance.py`)

```python
class TestEmbeddingPerformance:
    """Test embedding performance"""

    @pytest.mark.asyncio
    async def test_single_embedding_latency():
        """Test single text embedding latency"""
        service = await get_embedding_service()
        text = "テスト" * 100

        start = time.time()
        await service.embed_text(text)
        latency = (time.time() - start) * 1000

        assert latency < 200  # <200ms target

    @pytest.mark.asyncio
    async def test_batch_throughput():
        """Test batch embedding throughput"""
        service = await get_embedding_service()
        texts = ["テスト" * 100 for _ in range(100)]

        start = time.time()
        await service.embed_texts(texts)
        throughput = len(texts) / (time.time() - start)

        assert throughput > 20  # >20 texts/second
```

#### 2. Query Performance (`test_query_performance.py`)

```python
class TestQueryPerformance:
    """Test query performance"""

    @pytest.mark.asyncio
    async def test_query_latency_p95():
        """Test 95th percentile query latency"""
        latencies = []
        for _ in range(100):
            start = time.time()
            await query_rag("テスト")
            latencies.append((time.time() - start) * 1000)

        p95 = np.percentile(latencies, 95)
        assert p95 < 2000  # <2s target
```

---

## Test Data & Fixtures

### Fixtures (`tests/fixtures/`)

#### 1. Conftest.py

```python
# tests/fixtures/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from backend.main import app
from backend.db.session import get_db_session


@pytest.fixture
async def client():
    """Test HTTP client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session():
    """Test database session"""
    async with get_db_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def auth_headers(client, test_user):
    """Authentication headers"""
    response = client.post("/api/v1/auth/login", json={
        "email": test_user["email"],
        "password": test_user["password"]
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_user():
    """Test user data"""
    return {
        "email": "test@example.com",
        "password": "test_password",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_pdf():
    """Sample PDF bytes"""
    with open("tests/fixtures/sample.pdf", "rb") as f:
        return f.read()


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM response"""
    async def mock_generate(*args, **kwargs):
        return LLMResponse(
            text="Mocked response",
            tokens=10,
            model="mock"
        )
    monkeypatch.setattr("OllamaLLM.generate", mock_generate)


@pytest.fixture
async def clean_db(db_session):
    """Clean database before each test"""
    yield
    # Cleanup
    await db_session.execute("DELETE FROM chunks")
    await db_session.execute("DELETE FROM documents")
    await db_session.execute("DELETE FROM users")
```

#### 2. Test Data Files

```
tests/fixtures/
├── sample.pdf              # Sample Japanese PDF
├── sample_multi_page.pdf   # Multi-page PDF
├── sample_with_table.pdf   # PDF with tables
├── sample_scanned.pdf      # Scanned PDF (images)
└── test_vectors.npy        # Pre-computed embeddings
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
      milvus:
        image: milvusdb/milvus:latest
      redis:
        image: redis:latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/unit -v --cov=backend --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Test Execution

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/unit/core/test_config.py -v

# Run specific test
pytest tests/unit/core/test_config.py::TestSettings::test_default_settings -v

# Run integration tests
pytest tests/integration -v

# Run E2E tests
pytest tests/e2e -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Test Markers

```python
# pytest.ini
[tool:pytest]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow running tests",
    "gpu: Tests requiring GPU",
    "external: Tests requiring external services"
]
```

### Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| **Core** | 95% | TBD |
| **Services** | 85% | TBD |
| **API** | 80% | TBD |
| **Database** | 75% | TBD |
| **Overall** | 80% | TBD |

---

## Test Schedule

| Test Type | When to Run | Duration |
|-----------|-------------|----------|
| **Unit** | Every commit | <5s |
| **Integration** | Every PR | <30s |
| **E2E** | Pre-merge | <2min |
| **Performance** | Nightly | <10min |
| **Full Suite** | Pre-release | <15min |

---

## Next Steps

1. **Phase 1**: Set up test framework (pytest, fixtures, conftest.py)
2. **Phase 2**: Write unit tests for core components
3. **Phase 3**: Write unit tests for services
4. **Phase 4**: Write integration tests
5. **Phase 5**: Write E2E tests
6. **Phase 6**: Add performance benchmarks
7. **Phase 7**: Configure CI/CD integration

---

**Status**: 📋 Planning
**Next Action**: Set up pytest configuration and create first unit test
**Owner**: Development Team
**Review Date**: 2026-01-15
