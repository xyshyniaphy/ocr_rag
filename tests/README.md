# Tests Directory

This directory contains all tests for the **Japanese OCR RAG System**.

## Directory Structure

```
tests/
├── TEST_PLAN.md                 # Comprehensive test plan document
├── README.md                    # This file
├── pytest.ini                   # Pytest configuration
├── requirements-test.txt        # Test dependencies
│
├── fixtures/                    # Shared fixtures and configuration
│   └── conftest.py              # Main pytest fixtures
│
├── unit/                        # Unit tests (fast, isolated)
│   └── core/                    # Core component tests
│       ├── test_config.py       # Configuration tests
│       ├── test_exceptions.py   # Exception tests
│       ├── test_logging.py      # Logging tests (TODO)
│       ├── test_security.py     # Security tests (TODO)
│       └── test_cache.py        # Cache tests (TODO)
│
├── integration/                 # Integration tests (medium speed)
│   ├── services/                # Service integration tests
│   └── db/                      # Database integration tests
│
├── e2e/                         # End-to-end tests (slow)
│   └── api/                     # API flow tests
│
├── performance/                 # Performance benchmarks
│   ├── test_embedding_performance.py
│   └── test_query_performance.py
│
└── manual/                      # Manual test scripts
    ├── test_embedding.py
    ├── test_ocr.py
    ├── test_reranker.py
    ├── test_retrieval.py
    ├── test_llm.py
    └── test_rag.py
```

## Quick Start

### Installation

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or using uv (faster)
uv pip install -r requirements-test.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/unit/core/test_config.py -v

# Run specific test
pytest tests/unit/core/test_config.py::TestSettingsDefaults::test_app_name_default -v

# Run with markers
pytest -m "unit"              # Unit tests only
pytest -m "integration"        # Integration tests only
pytest -m "not slow"           # Skip slow tests
pytest -m "gpu"                # GPU tests only

# Run in parallel (faster)
pytest -n auto

# Generate HTML report
pytest --html=report.html --self-contained-html

# Show print statements
pytest -s
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components.

- **Duration**: < 5 seconds total
- **Isolation**: No external dependencies
- **Coverage Target**: 90%+

```bash
pytest tests/unit -v --tb=short
```

### Integration Tests (`tests/integration/`)

Tests for service integration and database interactions.

- **Duration**: < 30 seconds total
- **Dependencies**: Docker services (PostgreSQL, Milvus, Redis)
- **Coverage Target**: 70%+

```bash
# Start services first
docker-compose up -d postgres milvus redis

# Run integration tests
pytest tests/integration -v
```

### E2E Tests (`tests/e2e/`)

Full pipeline tests from API to database.

- **Duration**: < 2 minutes total
- **Dependencies**: Full Docker stack
- **Coverage**: Critical paths only

```bash
# Start full stack
./dev.sh up

# Run E2E tests
pytest tests/e2e -v
```

### Performance Tests (`tests/performance/`)

Benchmarks for critical operations.

```bash
pytest tests/performance -v --benchmark-only
```

## Test Fixtures

### Available Fixtures

| Fixture | Description | Scope |
|---------|-------------|-------|
| `client` | HTTPX AsyncClient for API testing | function |
| `auth_headers` | Authentication headers for API requests | function |
| `db_session` | Database session (auto-rollback) | function |
| `clean_db` | Clean database before/after test | function |
| `test_user` | Test user data | function |
| `sample_pdf` | Sample PDF bytes | function |
| `sample_embedding` | Sample embedding vector | function |
| `mock_llm_response` | Mock LLM response | function |
| `embedding_service` | Embedding service (CPU) | function |
| `ocr_service` | OCR service | function |
| `skip_if_no_gpu` | Skip test if no GPU | function |
| `benchmark_timer` | Performance timer | function |

### Example Usage

```python
import pytest

class TestMyFeature:
    @pytest.mark.asyncio
    async def test_with_client(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_with_auth(self, client, auth_headers):
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_with_db(self, db_session):
        # Use db_session - auto rollback after test
        pass

    def test_skip_if_no_gpu(self, skip_if_no_gpu):
        # Test will be skipped if no GPU available
        pass
```

## Test Markers

```bash
# Run specific test types
pytest -m unit              # Unit tests only
pytest -m integration        # Integration tests only
pytest -m e2e               # E2E tests only
pytest -m "not slow"        # Skip slow tests
pytest -m gpu               # GPU tests only
pytest -m external          # Tests with external services

# Combine markers
pytest -m "unit and not gpu"
pytest -m "integration or e2e"
```

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=backend --cov-report=term-missing

# HTML report
pytest --cov=backend --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=backend --cov-report=xml
```

### Coverage Targets

| Component | Target |
|-----------|--------|
| Core | 95% |
| Services | 85% |
| API | 80% |
| Database | 75% |
| **Overall** | **80%** |

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
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
        run: pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Writing New Tests

### Unit Test Template

```python
#!/usr/bin/env python3
"""
Unit Tests for [Component]
Tests for backend/[path]/[module].py
"""

import pytest
from backend.[path].[module] import [Class]


class Test[ClassName]:
    """Test [Class] functionality"""

    def test_[specific_behavior](self):
        """Test [specific behavior]"""
        # Arrange
        input_data = "..."

        # Act
        result = [ClassName].[method](input_data)

        # Assert
        assert result == expected_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Integration Test Template

```python
#!/usr/bin/env python3
"""
Integration Tests for [Service]
"""

import pytest


class Test[Service]Integration:
    """Test [Service] integration"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_[flow](self, client, auth_headers):
        """Test complete [flow]"""
        # Arrange
        payload = {...}

        # Act
        response = await client.post(
            "/api/v1/endpoint",
            json=payload,
            headers=auth_headers
        )

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["key"] == expected_value
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "Module not found"

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install app dependencies
pip install -r requirements-app.txt
```

**Issue**: GPU tests fail on CPU-only machine

```bash
# Skip GPU tests
pytest -m "not gpu"
```

**Issue**: Database connection errors

```bash
# Start required services
./dev.sh up

# Or use docker-compose
docker-compose up -d postgres milvus redis
```

**Issue**: Import errors

```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/ocr_rag

# Or use pytest --pythonpath
pytest --pythonpath=/path/to/ocr_rag
```

## Test Execution Timeline

| Test Suite | Duration | When to Run |
|------------|----------|-------------|
| Unit | <5s | Every commit |
| Integration | <30s | Every PR |
| E2E | <2min | Pre-merge |
| Performance | <10min | Nightly |
| Full | <15min | Pre-release |

## Further Reading

- **[TEST_PLAN.md](TEST_PLAN.md)** - Comprehensive test plan
- **[Pytest Documentation](https://docs.pytest.org/)**
- **[pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)**
- **[HTTPX Testing](https://www.python-httpx.org/#advanced-testing-clients)**

## Status

| Component | Tests | Coverage | Status |
|-----------|-------|----------|--------|
| Core | 2 | TBD | 🟡 Setup |
| Services | 0 | 0% | 🔴 Not Started |
| API | 0 | 0% | 🔴 Not Started |
| Database | 0 | 0% | 🔴 Not Started |
| Integration | 0 | 0% | 🔴 Not Started |
| E2E | 0 | 0% | 🔴 Not Started |

**Overall Progress**: 🟡 Initial setup complete, implementation in progress

---

**Last Updated**: 2026-01-02
