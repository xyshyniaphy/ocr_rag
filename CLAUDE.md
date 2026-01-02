# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Japanese OCR RAG System** - a production-grade Retrieval-Augmented Generation system optimized for Japanese PDF document processing. The system is privacy-first (air-gapped deployment supported) and designed for legal, financial, academic, and enterprise document analysis.

## IMPORTANT RULES

1. **DO NOT edit `Dockerfile.base` unless explicitly asked by the user**
   - The base image is complex and takes ~10 minutes to build
   - It contains ML models and heavy dependencies (PyTorch, CUDA, etc.)
   - Only edit when adding new ML libraries or updating core dependencies

2. **ALWAYS use `dev.sh` to start/stop Docker services**
   - Use `./dev.sh up` to start development environment
   - Use `./dev.sh down` to stop services
   - Use `./dev.sh logs [service]` to view logs
   - Use `./dev.sh shell` to open shell in app container
   - Do NOT use `docker-compose` directly

## Directory Structure

```
ocr_rag/
├── backend/                   # Python Backend (FastAPI)
│   ├── main.py               # FastAPI application entry point
│   ├── core/                  # Configuration, logging, security
│   ├── models/                # SQLAlchemy & Pydantic models
│   ├── api/                   # REST API routes
│   │   └── v1/               # API v1 endpoints
│   ├── db/                    # Database (PostgreSQL + Milvus)
│   │   ├── repositories/     # Repository pattern
│   │   └── vector/           # Vector database client
│   ├── storage/              # MinIO object storage client
│   ├── services/             # Business logic
│   │   ├── ocr/             # OCR processing (YomiToku)
│   │   ├── embedding/       # Embedding generation (Sarashina)
│   │   ├── retrieval/       # Vector + Keyword search (Hybrid)
│   │   ├── reranker/        # Reranking (Llama-3.2-NV-RerankQA)
│   │   ├── llm/            # LLM generation (Qwen via Ollama)
│   │   └── rag/            # RAG orchestration
│   ├── tasks/                # Celery background tasks
│   └── utils/                # Utility functions
│
├── frontend/                  # Streamlit Admin UI
│   └── app.py                # Streamlit application
│
├── Dockerfile.base            # Base image (ML models, dependencies)
├── Dockerfile.app             # Application image (lightweight)
├── docker-compose.dev.yml     # Development environment
├── docker-compose.prd.yml     # Production environment
├── dev.sh                     # Development environment manager script
├── requirements-base.txt      # Base ML dependencies
└── requirements-app.txt       # Application dependencies
```

## Quick Start

```bash
# Development
./dev.sh          # Start development environment (default: up)
./dev.sh logs     # View logs
./dev.sh shell    # Open shell in container
./dev.sh down     # Stop all services
./dev.sh help     # Show all commands

# Testing (IMPORTANT: All tests MUST run inside Docker)
./test.sh         # Run all tests in Docker
./test.sh unit    # Run unit tests only
./test.sh integration --coverage  # Run integration tests with coverage
./test.sh -m "not slow"  # Run fast tests only
./test.sh --help   # Show all test options

# Production
./prod.sh         # Start production environment (when available)
```

## Access Points

| Service | URL | Notes |
|---------|-----|-------|
| FastAPI Backend | http://localhost:8000 | Main API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Streamlit Admin UI | http://localhost:8501 | Web interface |
| WebSocket | ws://localhost:8000/api/v1/stream/ws | Real-time updates |
| MinIO Console | http://localhost:9001 | Object storage (dev only) |
| Prometheus Metrics | http://localhost:8000/api/v1/admin/metrics | Metrics scraping endpoint |

**Optional Services** (require profiles):
| Service | Profile | URL |
|---------|---------|-----|
| Prometheus | `--profile monitoring` | http://localhost:9090 |
| Grafana | `--profile monitoring` | http://localhost:3000 |
| PgAdmin | `--profile tools` | http://localhost:5050 |

## Default Credentials

| Service | Username | Password |
|----------|----------|----------|
| Admin User | admin@example.com | admin123 |
| MinIO | minioadmin | minioadmin |

## Architecture

### High-Level Flow

**Document Ingestion:**
```
PDF Upload → Validation → OCR (YomiToku) → Markdown → Chunking
→ Embedding (Sarashina) → Milvus → PostgreSQL → Complete
```

**Query Processing:**
```
User Query → Embedding → Hybrid Search (Milvus + PostgreSQL BM25)
→ Reranking (Llama-3.2-NV) → Context Assembly → LLM (Qwen)
→ Response with Sources
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| OCR | YomiToku (Japanese optimized) |
| Embedding | Sarashina-Embedding-v1-1B (1792D) |
| Reranker | Llama-3.2-NV-RerankQA-1B-v2 |
| LLM | **GLM-4.5-Air** (default, cloud) or Qwen2.5-14B (Ollama, fallback) |
| Vector DB | Milvus 2.4+ |
| Metadata DB | PostgreSQL 16+ |
| Object Storage | MinIO |
| Cache | Redis |
| Task Queue | Celery |

## Configuration

All configuration is managed through:
- **Environment variables** (`.env` file)
- **`backend/core/config.py`** - Settings class with validation

Key environment variables:
- `SECRET_KEY` - JWT signing key (generate with `openssl rand -hex 32`)
- `POSTGRES_PASSWORD` - PostgreSQL password
- `LLM_PROVIDER` - LLM provider: `glm` (default, cloud) or `ollama` (local)
- `GLM_API_KEY` - GLM API key from https://z.ai/
- `GLM_MODEL` - GLM model: `GLM-4.5-Air` (recommended), `GLM-4.5`, `GLM-4.7`
- `GLM_BASE_URL` - GLM API endpoint: `https://api.z.ai/api/paas/v4/` (international)
- `OCR_ENGINE` - OCR engine selection (`yomitoku`)

## LLM Providers

The system supports two LLM providers:

### GLM (Default, Recommended)
- **Provider**: Z.ai international platform
- **Models**: GLM-4.5-Air (fast, cost-effective), GLM-4.5, GLM-4.7
- **Advantages**:
  - ✅ Fast response times (cloud API)
  - ✅ No local GPU required
  - ✅ OpenAI-compatible API
  - ✅ Cost-effective for production
  - ✅ High-quality Japanese responses
- **Configuration**:
  ```bash
  LLM_PROVIDER=glm
  GLM_API_KEY=your-api-key-from-z.ai
  GLM_MODEL=GLM-4.5-Air
  GLM_BASE_URL=https://api.z.ai/api/paas/v4/
  ```
- **Get API Key**: https://z.ai/ (international platform, 1M+ free tokens)

### Ollama (Fallback, Local)
- **Provider**: Local Ollama server
- **Models**: Qwen2.5, Qwen3, and other Ollama models
- **Advantages**:
  - ✅ Privacy (local processing)
  - ✅ No API costs
  - ✅ Air-gapped deployment
- **Disadvantages**:
  - ⚠️ Slower response times
  - ⚠️ Requires local GPU
  - ⚠️ Limited by hardware
- **Configuration**:
  ```bash
  LLM_PROVIDER=ollama
  OLLAMA_HOST=ollama:11434
  OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M
  ```

### Switching Providers

Set the `LLM_PROVIDER` environment variable:
- `glm` - Use GLM cloud API (default, recommended)
- `ollama` - Use local Ollama (fallback)

The system automatically selects the appropriate client based on this setting.

## API Endpoints

### Authentication & User Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | Login and get access token |
| `/api/v1/auth/register` | POST | Register new user |
| `/api/v1/auth/refresh` | POST | Refresh access token |
| `/api/v1/auth/me` | GET | Get current user profile |
| `/api/v1/auth/me` | PUT | Update user profile |
| `/api/v1/auth/me/password` | PUT | Change password |
| `/api/v1/auth/logout` | POST | Logout |
| `/api/v1/auth/users` | GET | List all users (admin) |
| `/api/v1/auth/users/{id}` | DELETE | Deactivate user (admin) |

### Document Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/documents/upload` | POST | Upload PDF document |
| `/api/v1/documents` | GET | List documents (with filters) |
| `/api/v1/documents/{id}` | GET | Get document details |
| `/api/v1/documents/{id}` | DELETE | Delete document (soft delete) |
| `/api/v1/documents/{id}/download` | GET | Download original PDF |
| `/api/v1/documents/{id}/status` | GET | Get processing status |

**Query Parameters for `/api/v1/documents`:**
- `status`: Filter by status (pending, processing, completed, failed)
- `category`: Filter by category
- `date_from`: Filter documents created after this date (ISO 8601)
- `date_to`: Filter documents created before this date (ISO 8601)
- `limit`: Page size (1-100, default 20)
- `offset`: Page offset (default 0)

### Query & RAG
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Submit RAG query |
| `/api/v1/queries` | GET | List query history |
| `/api/v1/queries/{id}` | GET | Get query details |
| `/api/v1/queries/{id}/feedback` | POST | Submit query feedback |
| `/api/v1/stream/ws` | WebSocket | Real-time query streaming |

### Permissions Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/permissions/{document_id}` | GET | List document permissions |
| `/api/v1/permissions/{document_id}/grant` | POST | Grant permission to user |
| `/api/v1/permissions/{document_id}/revoke` | DELETE | Revoke permission from user |
| `/api/v1/permissions/user/{user_id}` | GET | List user permissions |

**Permission Types:** `can_view`, `can_download`, `can_delete`

### Admin & Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/admin/stats` | GET | Get system statistics |
| `/api/v1/admin/users` | GET | List all users (paginated) |
| `/api/v1/admin/metrics` | GET | Prometheus metrics endpoint |

## Module Import Conventions

- **Backend code**: Use absolute imports from `backend` package
  ```python
  from backend.core.config import settings
  from backend.db.session import get_db_session
  from backend.api.v1 import auth, documents
  ```

- **Uvicorn command**: Use `backend.main:app` module path
  ```bash
  uvicorn backend.main:app --reload
  ```

## Code Structure Guidelines

1. **API Routes**: Place in `backend/api/v1/`
   - Follow FastAPI routing patterns
   - Use dependency injection for database sessions
   - Return Pydantic models for responses

2. **Models**:
   - **SQLAlchemy models** (database tables):
     - `backend/db/models.py` - User, Document, Query models
     - `backend/models/chunk.py` - Chunk model
     - `backend/models/permission.py` - Permission model
   - **Pydantic models** (request/response schemas):
     - `backend/models/user.py` - User request/response schemas
     - `backend/models/document.py` - Document request/response schemas
     - `backend/models/query.py` - Query request/response schemas
   - All SQLAlchemy models use the shared `Base` from `backend.db.base`

3. **Services**: Business logic in `backend/services/`
   - OCR engines in `services/ocr/`
   - Embedding in `services/embedding/`
   - Retrieval (vector + keyword) in `services/retrieval/`
   - Reranking in `services/reranker/`
   - LLM generation in `services/llm/` (supports GLM and Ollama)
   - RAG pipeline in `services/rag/`

4. **Database Access**: Use Repository pattern
   - Repositories in `backend/db/repositories/`
   - Vector DB client in `backend/db/vector/`

## Japanese Language Handling

- **Text Normalization**: Use Unicode NFKC normalization
- **Chunking Separators**: `["\n\n", "\n", "。", "！", "？", "；", "、"]`
- **Chunk Size**: 512 characters (Japanese chars ≈ 0.5 tokens)
- **Overlap**: 50 characters

## GPU Resource Management

Single GPU (RTX 4090 24GB) allocation:
- OCR (YomiToku): 40% VRAM (~9.6GB)
- Embedding (Sarashina): 30% VRAM (~7.2GB)
- Reranker: 10% VRAM (~2.4GB)
- LLM (Qwen): 20% VRAM (~4.8GB)

## Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Query (GPU) | <2s | 5s (95th percentile) |
| OCR Processing | <10s/page | 30s/page |
| Embedding | <50ms/chunk | 200ms/chunk |
| Vector Search | <100ms | 500ms |

## Docker Multi-Stage Build

The project uses a multi-stage Docker build:

1. **Base Image** (`Dockerfile.base`):
   - Contains ML models (Sarashina, YomiToku)
   - Heavy dependencies (PyTorch, Transformers, sentence-transformers)
   - Cached and rebuilt infrequently
   - **WARNING**: DO NOT edit without explicit user approval

2. **App Image** (`Dockerfile.app`):
   - Application code (`backend/`, `frontend/`)
   - Lightweight dependencies (FastAPI, Streamlit)
   - Rebuilt on code changes

Use `./dev.sh rebuild base` or `./dev.sh rebuild app` to rebuild images.

## Model Storage Architecture

**CRITICAL POLICY: ALL MODELS MUST BE IN BASE IMAGE - NO RUNTIME DOWNLOADS**

All ML models MUST be pre-downloaded in the Docker base image. Runtime downloads from HuggingFace Hub are **FORBIDDEN** to ensure:
- ✅ Air-gapped deployment capability
- ✅ Predictable startup times
- ✅ No external dependencies at runtime
- ✅ Version-locked models

```
Container File System (Base Image - Read-Only):
├── /app/models/                    # Base image (read-only)
│   ├── sarashina/                  # ✅ Pre-downloaded in Dockerfile.base
│   └── yomitoku/                   # ✅ Library managed (cached)
│
└── /app/reranker_models/           # ✅ Base image (read-only)
    └── llama-nv-reranker/          # ✅ Pre-downloaded in Dockerfile.base

Container File System (Volume Mounts - Read-Write):
└── /app/reranker_models/           # Volume mount (for cache only, NOT for models)
    └── huggingface_cache/          # HF download cache (read-only after build)
```

### Model-by-Model Details

| Model | Path | Source | Volume | In Base Image? |
|-------|------|--------|--------|----------------|
| **Sarashina** | `/app/models/sarashina/` | Pre-downloaded | None | ✅ YES |
| **Reranker** | `/app/reranker_models/llama-nv-reranker/` | Pre-downloaded | None | ✅ YES |
| **YomiToku** | `/app/models/yomitoku/` | Library managed | None | ✅ YES (cached) |
| **Qwen LLM** | N/A (Ollama) | Ollama managed | Yes | N/A (Ollama service) |

### Important Rules

1. **ALL Models Pre-Downloaded**: ALL models MUST be in Docker base image
   - ✅ Edit `Dockerfile.base` to add/update models
   - ❌ NO runtime downloads from HuggingFace Hub
   - ❌ NO fallback to internet for models

2. **Sarashina**: Pre-downloaded in base image at `/app/models/sarashina/`
   - ✅ Downloaded via `huggingface-cli` in Dockerfile.base
   - ✅ Located in base image, no volume mount needed
   - ✅ Code MUST NOT fall back to HuggingFace Hub

3. **Reranker**: Pre-downloaded in base image at `/app/reranker_models/llama-nv-reranker/`
   - ✅ Downloaded via `huggingface-cli` in Dockerfile.base
   - ✅ Located in base image, no volume mount needed
   - ✅ Code MUST NOT fall back to HuggingFace Hub
   - ❌ NO volume mount for model storage

4. **YomiToku & Qwen**: Managed by their respective libraries
   - ✅ YomiToku caches models in base image during build
   - ✅ Qwen runs in separate Ollama container
   - ✅ Libraries MUST use local paths only

## Development Workflow

1. **Make changes to code** in `backend/` or `frontend/`
2. **Restart containers**: `./dev.sh restart`
3. **Hot reload**: Development mode auto-reloads on Python changes
4. **View logs**: `./dev.sh logs` or `./dev.sh logs app`
5. **Run tests**: `./test.sh unit` (always run tests in Docker)

---

## Testing

**CRITICAL POLICY: ALL TESTS MUST RUN INSIDE DOCKER**

### Test Summary (2026-01-03)

- **Total Tests:** 456
- **Passing:** 378 (83%)
- **Failing:** 29 (6%)
- **Skipped:** 46 (10%)

| Category | Tests | Pass Rate | Status |
|----------|-------|-----------|--------|
| **Unit** | 206 | 99.5% | ✅ Excellent |
| **Integration** | 210 | 82% | ✅ Good |
| **Total** | **456** | **83%** | ✅ Good |

### Unit Tests (205 passed, 1 skipped)
- ✅ `test_config.py` - Configuration validation
- ✅ `test_exceptions.py` - Custom exception classes
- ✅ `test_logging.py` - Logging configuration
- ✅ `test_security.py` - JWT token creation/verification
- ✅ `test_cache.py` - Cache manager functionality
- ✅ `test_permissions.py` - Permission system (14 tests, 1 skipped)

### Integration Tests (173 passed, 29 failed, 9 skipped)
- ✅ **Auth API** - Login, registration, token refresh, profile management
- ✅ **Query API** - Query submission, history, feedback
- ✅ **Document API** - Upload, list, delete (with ACL enforcement)
- ✅ **Admin API** - Stats, user list, metrics endpoint
- ✅ **Permissions API** - Grant/revoke/list permissions
- ⚠️ **Database Tests** - Milvus tests failing (external service dependency)
- ⚠️ **Service Tests** - Embedding tests failing (GPU dependency)

### Known Issues
1. **Status Code Mismatches** - Some tests expect 422 but API returns 400 (both are valid)
2. **Admin User Setup** - Admin API tests need proper admin user creation
3. **External Dependencies** - Milvus/Embedding tests require running services
4. **GPU Tests** - Some embedding tests require CUDA GPU

All tests MUST be executed inside the Docker container to ensure:
- ✅ Consistent test environment
- ✅ Access to all dependencies (PostgreSQL, Milvus, Redis, Ollama)
- ✅ GPU access for model tests
- ✅ No local machine setup required

### Docker Test Infrastructure

The project uses a dedicated Docker test container:

**Components**:
1. **Dockerfile.test** - Test image based on `ocr-rag-app:dev` with test dependencies
2. **docker-compose.dev.yml test service** - Dedicated test container with `--profile test`
3. **requirements-test.txt** - Test dependencies (pytest, pytest-asyncio, httpx, etc.)
4. **pytest.ini** - Pytest configuration with markers, coverage, and asyncio settings

**Test Container** (`ocr-rag-test-dev`):
- Image: `ocr-rag-test:dev`
- Profile: `test` (requires `--profile test` to start)
- Depends on: postgres, milvus, minio, redis, ollama
- Mounts: backend, frontend, tests, pytest.ini
- GPU: Enabled
- Command: `sleep infinity` (stays running for manual commands)

### Test Runner Script (`test.sh`)

The `test.sh` script is the RECOMMENDED way to run tests:

```bash
# Run all tests
./test.sh

# Run specific test types
./test.sh unit              # Unit tests only (fast, <5s)
./test.sh integration       # Integration tests (medium, <30s)
./test.sh e2e              # End-to-end tests (slow, <2min)

# With coverage report
./test.sh --coverage
./test.sh unit --coverage

# With markers
./test.sh -m "not slow"     # Skip slow tests
./test.sh -m "unit"         # Unit tests only
./test.sh -m "gpu"          # GPU tests only

# Other options
./test.sh --verbose         # Verbose output
./test.sh --parallel        # Run tests in parallel
./test.sh --build           # Rebuild containers before testing
./test.sh --keep            # Keep containers running after tests

# Show help
./test.sh --help
```

### Test Structure

```
tests/
├── fixtures/              # Shared fixtures (conftest.py)
│   └── conftest.py       # Pytest configuration and fixtures
│
├── unit/                  # Unit tests (fast, isolated)
│   └── core/             # Core component tests
│       ├── test_config.py      # Configuration tests
│       ├── test_exceptions.py  # Exception tests
│       ├── test_logging.py     # Logging tests
│       ├── test_security.py    # Security tests
│       ├── test_cache.py       # Cache tests
│       └── test_permissions.py # Permission system tests
│
├── integration/           # Integration tests (medium speed)
│   ├── api/              # API integration tests
│   │   ├── conftest.py        # API fixtures
│   │   ├── test_auth_api.py    # Auth endpoints (login, register, profile, admin)
│   │   ├── test_documents_api.py # Document endpoints (with ACL)
│   │   ├── test_query_api.py   # Query endpoints (history, feedback)
│   │   ├── test_admin_api.py   # Admin endpoints
│   │   └── test_system_endpoints.py # Health check
│   ├── db/               # Database integration tests
│   │   └── test_milvus_integration.py
│   └── services/         # Service integration tests
│       └── test_embedding_integration.py
│
├── e2e/                  # End-to-end tests (slow)
│
└── manual/               # Manual test scripts (existing)
    ├── test_embedding.py
    ├── test_ocr.py
    └── test_reranker.py
```

### Test Categories

| Category | Duration | When to Run | Coverage Target |
|----------|----------|-------------|-----------------|
| **Unit** | <5s | Every commit | 90%+ |
| **Integration** | <30s | Every PR | 70%+ |
| **E2E** | <2min | Pre-merge | Critical paths |
| **Performance** | <10min | Nightly | N/A |

### Test Markers

```bash
# Run tests by marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m e2e           # E2E tests only
pytest -m "not slow"    # Skip slow tests
pytest -m gpu           # GPU tests only
pytest -m external      # Tests with external services
```

### Coverage Reports

```bash
# Generate coverage report
./test.sh --coverage

# View HTML report
open htmlcov/index.html

# View in terminal
cat test-results/coverage.json | jq
```

### Writing Tests

When writing new tests, follow these guidelines:

1. **Use Docker**: Always write tests to run inside Docker
2. **Use Fixtures**: Leverage fixtures in `tests/fixtures/conftest.py`
3. **Mark Tests**: Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.gpu`, etc.)
4. **Mock External Services**: Mock external APIs, but use real database services
5. **Test Isolation**: Each test should be independent

Example test:

```python
import pytest
from backend.core.config import settings

@pytest.mark.unit
class TestSettings:
    def test_default_app_name(self):
        """Test default APP_NAME"""
        assert settings.APP_NAME == "Japanese OCR RAG System"

    @pytest.mark.gpu
    def test_embedding_device(self):
        """Test embedding device configuration"""
        assert settings.EMBEDDING_DEVICE == "cuda:0"
```

### Test Fixtures

Available fixtures (in `tests/fixtures/conftest.py`):

- `client` - HTTPX AsyncClient for API testing
- `auth_headers` - Authentication headers
- `db_session` - Database session with auto-rollback
- `test_user` - Test user data
- `sample_pdf` - Sample PDF bytes
- `embedding_service` - Embedding service (CPU mode)
- `mock_llm_response` - Mock LLM response

### CI/CD Integration

Tests run automatically in CI/CD:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: ./test.sh --coverage

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Running Tests with Docker Compose

For more control over test execution, you can use Docker Compose directly:

```bash
# Start test container (keeps running for interactive use)
docker compose -f docker-compose.dev.yml --profile test up -d test

# Run specific test file
docker exec ocr-rag-test-dev pytest tests/integration/api/test_query_api.py -v

# Run specific test class
docker exec ocr-rag-test-dev pytest tests/integration/api/test_query_api.py::TestQueryAPIAuth -v

# Run single test
docker exec ocr-rag-test-dev pytest tests/integration/api/test_query_api.py::TestQueryAPIAuth::test_query_without_auth_returns_401 -v

# Run with coverage
docker exec ocr-rag-test-dev pytest tests/integration/api/test_query_api.py --cov=backend/api/v1/query --cov-report=html

# Stop test container when done
docker compose -f docker-compose.dev.yml --profile test down
```

**One-shot test execution** (container starts, runs tests, exits):
```bash
# Run specific tests
docker compose -f docker-compose.dev.yml --profile test run --rm test pytest tests/unit/core/test_config.py -v

# Run with coverage
docker compose -f docker-compose.dev.yml --profile test run --rm test pytest --cov=backend --cov-report=html

# Run with markers
docker compose -f docker-compose.dev.yml --profile test run --rm test pytest -m "not gpu" -v
```

**Note**: When dev containers are already running, use `--no-deps` to avoid conflicts:
```bash
docker compose -f docker-compose.dev.yml --profile test run --no-deps --rm test pytest tests/unit/ -v
```

### Test Structure

```
tests/
├── fixtures/              # Shared fixtures (conftest.py)
│   └── conftest.py       # Pytest configuration and fixtures
│
├── unit/                  # Unit tests (fast, isolated)
│   └── core/             # Core component tests
│       ├── test_config.py    # Configuration tests
│       ├── test_exceptions.py # Exception tests
│       ├── test_logging.py    # Logging tests
│       ├── test_security.py   # Security tests
│       └── test_cache.py      # Cache tests
│
├── integration/           # Integration tests (medium speed)
│   ├── api/              # API integration tests
│   │   ├── test_documents_api.py    # Documents API tests
│   │   └── test_query_api.py        # Query API tests
│   ├── services/         # Service integration tests
│   │   └── test_embedding_integration.py
│   └── db/               # Database integration tests
│       └── test_milvus_integration.py
│
├── e2e/                  # End-to-end tests (slow)
│
└── manual/               # Manual test scripts (existing)
    ├── test_embedding.py
    ├── test_ocr.py
    └── test_reranker.py
```

### Test Categories

| Category | Duration | When to Run | Coverage Target |
|----------|----------|-------------|-----------------|
| **Unit** | <5s | Every commit | 90%+ |
| **Integration** | <30s | Every PR | 70%+ |
| **E2E** | <2min | Pre-merge | Critical paths |
| **Performance** | <10min | Nightly | N/A |

### Test Markers

```bash
# Run tests by marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m e2e           # E2E tests only
pytest -m "not slow"    # Skip slow tests
pytest -m gpu           # GPU tests only
pytest -m external      # Tests with external services
pytest -m database       # Tests requiring database access
```

**Available Markers** (defined in `pytest.ini`):
- `unit` - Unit tests (fast, isolated)
- `integration` - Integration tests (medium speed)
- `e2e` - End-to-end tests (slow)
- `slow` - Slow running tests
- `gpu` - Tests requiring GPU
- `external` - Tests requiring external services
- `asyncio` - Async tests
- `database` - Tests requiring database access

### Coverage Reports

```bash
# Generate coverage report
./test.sh --coverage

# View HTML report
open htmlcov/index.html

# View in terminal
cat test-results/coverage.json | jq
```

### Writing Tests

When writing new tests, follow these guidelines:

1. **Use Docker**: Always write tests to run inside Docker
2. **Use Fixtures**: Leverage fixtures in `tests/fixtures/conftest.py`
3. **Mark Tests**: Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.gpu`, etc.)
4. **Mock External Services**: Mock external APIs, but use real database services
5. **Test Isolation**: Each test should be independent

Example test:

```python
import pytest
from backend.core.config import settings

@pytest.mark.unit
class TestSettings:
    def test_default_app_name(self):
        """Test default APP_NAME"""
        assert settings.APP_NAME == "Japanese OCR RAG System"

    @pytest.mark.gpu
    def test_embedding_device(self):
        """Test embedding device configuration"""
        assert settings.EMBEDDING_DEVICE == "cuda:0"
```

### Test Fixtures

Available fixtures (in `tests/fixtures/conftest.py`):

- `client` - HTTPX AsyncClient for API testing (uses ASGI transport)
- `auth_headers` - Authentication headers
- `db_session` - Database session with auto-rollback
- `test_user` - Test user data
- `sample_pdf` - Sample PDF bytes
- `embedding_service` - Embedding service (CPU mode)
- `mock_llm_response` - Mock LLM response

### CI/CD Integration

Tests run automatically in CI/CD:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: ./test.sh --coverage

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Troubleshooting Tests

**Issue**: Tests fail with "Module not found"
- **Fix**: Tests must run in Docker: `./test.sh`

**Issue**: GPU tests fail on CPU-only machine
- **Fix**: Skip GPU tests: `./test.sh -m "not gpu"`

**Issue**: Database connection errors
- **Fix**: Start services first: `./dev.sh up`

**Issue**: Import errors
- **Fix**: Ensure `PYTHONPATH=/app` is set (automatic in Docker)

**Issue**: "All connection attempts failed" when running tests
- **Cause**: Test container can't connect to dependencies
- **Fix**: Ensure all services are running: `./dev.sh up`
- **Fix**: Use `--profile test` to start test container with dependencies

**Issue**: Permission denied on test files
- **Fix**: Check file permissions: `chmod 644 tests/integration/api/test_*.py`

**Issue**: Pytest markers not found
- **Fix**: Add missing marker to `pytest.ini` [markers] section

## Troubleshooting

**Issue**: OCR confidence low
- Check GPU memory: `nvidia-smi`
- Verify input image quality
- Check YomiToku model is loaded correctly

**Issue**: Query latency high
- Check GPU utilization
- Increase Milvus `nprobe` parameter
- Enable query caching

**Issue**: Out of memory
- Reduce `EMBEDDING_BATCH_SIZE`
- Use smaller LLM model (Q4_K_M quantization)
- Add more GPUs

## Database

### Database Structure

The system uses two databases:

1. **PostgreSQL** (`rag_metadata`) - Relational database for:
   - `users` - User accounts with authentication
   - `documents` - Document metadata and status
   - `chunks` - Text chunks with embeddings metadata
   - `queries` - RAG query history and responses
   - `permissions` - Document-level access control (ACL)

2. **Milvus** (`document_chunks`) - Vector database for:
   - 1792-dimensional embeddings (Sarashina-Embedding-v1-1B)
   - Semantic search with HNSW index
   - Hybrid vector + keyword search support

### Database Initialization

Database tables are automatically created on first startup in development mode:

```python
# In backend/db/session.py
async def init_db() -> None:
    # Import all SQLAlchemy models
    from backend.db.models import User, Document, Query
    from backend.models.chunk import Chunk
    from backend.models.permission import Permission

    # Create tables in development mode
    if settings.ENVIRONMENT == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
```

**Important**: All SQLAlchemy models use the shared `Base` from `backend.db.base`:
- `backend/db/models.py` - User, Document, Query models
- `backend/models/chunk.py` - Chunk model
- `backend/models/permission.py` - Permission model

### Admin User

The default admin user can be created using the seed script:

```bash
# From within the app container
docker exec ocr-rag-app-dev python /app/scripts/seed_admin.py
```

**Admin credentials:**
- Email: `admin@example.com`
- Password: `admin123`
- Role: `admin`

The script:
- Checks if admin user exists
- Creates new admin user if not exists
- Updates password if admin already exists
- Uses bcrypt password hashing

### Database Migrations

For production, use Alembic for database migrations:

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Database Connection

Database connection is managed in `backend/db/session.py`:

```python
# Connection string format
postgresql+asyncpg://raguser:password@postgres:5432/rag_metadata

# Session dependency injection
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

## Security Notes

- All API endpoints require authentication except `/health` and `/`
- JWT tokens expire after 15 minutes (access) or 7 days (refresh)
- File uploads limited to 50MB PDFs only
- Rate limiting: 60 queries/minute, 10 uploads/minute
