# Project Structure Design

**Version:** 1.0
**Date:** 2026-01-01

---

## 1. Directory Structure

```
ocr_rag/
├── README.md
├── CLAUDE.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
│
├── docs/                          # Documentation
│   ├── design/                    # Design documents
│   │   ├── 01-system-architecture.md
│   │   ├── 02-api-specification.md
│   │   ├── 03-database-schema.md
│   │   └── 04-project-structure.md
│   ├── api/                       # API documentation (OpenAPI/Swagger)
│   └── guides/                    # User guides
│
├── app/                           # Main application code
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   │
│   ├── api/                       # API route handlers
│   │   ├── __init__.py
│   │   ├── deps.py                # Dependency injection
│   │   │
│   │   ├── v1/                    # API v1 routes
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # Authentication endpoints
│   │   │   ├── documents.py       # Document management
│   │   │   ├── query.py           # Query endpoints
│   │   │   ├── admin.py           # Admin endpoints
│   │   │   └── stream.py          # WebSocket streaming
│   │   │
│   │   └── middleware/            # Custom middleware
│   │       ├── __init__.py
│   │       ├── auth.py            # JWT authentication
│   │       ├── rate_limit.py      # Rate limiting
│   │       └── error_handler.py   # Error handling
│   │
│   ├── core/                      # Core application logic
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── security.py            # Security utilities (JWT, password hashing)
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── logging.py             # Logging configuration
│   │
│   ├── models/                    # Pydantic models (schemas)
│   │   ├── __init__.py
│   │   ├── auth.py                # Auth-related models
│   │   ├── document.py            # Document models
│   │   ├── query.py               # Query models
│   │   ├── user.py                # User models
│   │   └── common.py              # Shared models
│   │
│   ├── services/                  # Business logic services
│   │   ├── __init__.py
│   │   │
│   │   ├── auth_service.py        # Authentication & authorization
│   │   ├── user_service.py        # User management
│   │   ├── document_service.py    # Document CRUD operations
│   │   │
│   │   ├── ocr/                   # OCR processing
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base OCR interface
│   │   │   ├── yomitoku.py        # YomiToku OCR engine
│   │   │   ├── paddleocr.py       # PaddleOCR-VL fallback
│   │   │   ├── preprocessor.py    # Image preprocessing
│   │   │   ├── postprocessor.py   # Text postprocessing
│   │   │   └── quality.py         # Quality assessment
│   │   │
│   │   ├── embedding/             # Embedding generation
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base embedding interface
│   │   │   ├── sarashina.py       # Sarashina embedding model
│   │   │   └── batch_processor.py # Batch processing logic
│   │   │
│   │   ├── retrieval/             # Retrieval & search
│   │   │   ├── __init__.py
│   │   │   ├── vector_search.py   # Milvus vector search
│   │   │   ├── keyword_search.py  # PostgreSQL BM25 search
│   │   │   ├── hybrid_search.py   # Hybrid vector + keyword
│   │   │   └── reranker.py        # Reranking service
│   │   │
│   │   ├── llm/                   # LLM generation
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base LLM interface
│   │   │   ├── qwen.py            # Qwen2.5 LLM (Ollama)
│   │   │   ├── prompt_templates.py # Japanese prompt templates
│   │   │   └── stream_handler.py  # Streaming response handling
│   │   │
│   │   ├── rag/                   # RAG orchestration
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py        # Main RAG pipeline
│   │   │   ├── query_processor.py # Query understanding
│   │   │   ├── context_assembler.py # Context assembly
│   │   │   └── response_formatter.py # Response formatting
│   │   │
│   │   └── processing/            # Background processing
│   │       ├── __init__.py
│   │       ├── document_processor.py # Document ingestion pipeline
│   │       ├── chunking.py        # Text chunking
│   │       └── task_queue.py      # Celery task queue
│   │
│   ├── db/                        # Database layer
│   │   ├── __init__.py
│   │   ├── session.py             # Database session management
│   │   ├── base.py                # Base ORM model
│   │   │
│   │   ├── repositories/          # Repository pattern (data access)
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base repository
│   │   │   ├── user_repo.py       # User repository
│   │   │   ├── document_repo.py   # Document repository
│   │   │   ├── chunk_repo.py      # Chunk repository
│   │   │   └── query_repo.py      # Query repository
│   │   │
│   │   └── vector/                # Vector database (Milvus)
│   │       ├── __init__.py
│   │       ├── client.py          # Milvus client
│   │       ├── collections.py     # Collection management
│   │       └── search.py          # Search operations
│   │
│   ├── storage/                   # Object storage (MinIO)
│   │   ├── __init__.py
│   │   ├── client.py              # MinIO client
│   │   ├── buckets.py             # Bucket management
│   │   └── paths.py               # Path utilities
│   │
│   ├── tasks/                     # Background tasks (Celery)
│   │   ├── __init__.py
│   │   ├── celery_app.py          # Celery application
│   │   ├── document_tasks.py      # Document processing tasks
│   │   └── monitoring_tasks.py    # Monitoring tasks
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── text.py                # Text processing utilities
│       ├── japanese.py            # Japanese text normalization
│       ├── file.py                # File handling utilities
│       ├── crypto.py              # Cryptography (hashing, encryption)
│       └── validators.py          # Input validators
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── fixtures/                  # Test fixtures
│   │   ├── __init__.py
│   │   ├── users.py
│   │   ├── documents.py
│   │   └── queries.py
│   │
│   ├── unit/                      # Unit tests
│   │   ├── test_services/
│   │   │   ├── test_auth_service.py
│   │   │   ├── test_ocr_service.py
│   │   │   ├── test_embedding_service.py
│   │   │   └── test_rag_service.py
│   │   ├── test_repositories/
│   │   └── test_utils/
│   │
│   ├── integration/               # Integration tests
│   │   ├── test_api/
│   │   ├── test_db/
│   │   └── test_tasks/
│   │
│   └── e2e/                       # End-to-end tests
│       ├── test_document_upload.py
│       └── test_query_flow.py
│
├── scripts/                       # Utility scripts
│   ├── init_db.py                 # Database initialization
│   ├── seed_data.py               # Seed test data
│   ├── migrate_milvus.py          # Milvus migration
│   └── benchmark.py               # Performance benchmarking
│
├── frontend/                      # Streamlit admin UI
│   ├── __init__.py
│   ├── app.py                     # Streamlit entry point
│   ├── pages/                     # Streamlit pages
│   │   ├── __init__.py
│   │   ├── 1_Dashboard.py
│   │   ├── 2_Documents.py
│   │   ├── 3_Queries.py
│   │   ├── 4_Admin.py
│   │   └── 5_Settings.py
│   └── components/                # UI components
│       ├── __init__.py
│       ├── document_uploader.py
│       ├── query_interface.py
│       └── stats_display.py
│
├── config/                        # Configuration files
│   ├── nginx.conf                 # NGINX configuration
│   ├── supervisor.conf            # Supervisor configuration
│   └── prometheus/                # Prometheus monitoring
│       └── prometheus.yml
│
├── deployment/                    # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile.app
│   │   ├── Dockerfile.ollama
│   │   └── docker-compose.prod.yml
│   └── kubernetes/                # Kubernetes manifests
│       ├── namespace.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       └── ingress.yaml
│
└── .github/                       # GitHub configuration
    └── workflows/
        └── ci-cd.yml              # CI/CD pipeline
```

---

## 2. Module Organization

### 2.1 Application Entry Point

**`app/main.py`** - FastAPI application setup:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1 import auth, documents, query, admin, stream
from app.db.session import init_db, close_db
from app.db.vector.client import init_milvus, close_milvus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging()
    await init_db()
    await init_milvus()

    yield

    # Shutdown
    await close_db()
    await close_milvus()


app = FastAPI(
    title="Japanese OCR RAG System",
    description="Production-grade RAG system for Japanese PDF documents",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])
app.include_router(stream.router, prefix="/api/v1/stream", tags=["WebSocket"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}
```

### 2.2 Configuration Module

**`app/core/config.py`** - Configuration management:

```python
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "Japanese OCR RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_HASH_ALGORITHM: str = "bcrypt"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # Database (PostgreSQL)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "rag_metadata"

    # Vector Database (Milvus)
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_chunks"

    # Object Storage (MinIO)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False

    # Cache (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Task Queue (Celery)
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # OCR Configuration
    OCR_ENGINE: str = "yomitoku"  # yomitoku, paddleocr
    YOMITOKU_MODEL_PATH: str = "/models/yomitoku"
    PADDLEOCR_MODEL_PATH: str = "/models/paddleocr"
    OCR_CONFIDENCE_THRESHOLD: float = 0.85
    OCR_FALLBACK_THRESHOLD: float = 0.80

    # Embedding Configuration
    EMBEDDING_MODEL: str = "sbintuitions/sarashina-embedding-v1-1b"
    EMBEDDING_MODEL_PATH: str = "/models/sarashina-embedding-v1-1b"
    EMBEDDING_DEVICE: str = "cuda:0"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIMENSION: int = 768

    # Reranker Configuration
    RERANKER_MODEL: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    RERANKER_DEVICE: str = "cuda:0"
    RERANKER_TOP_K_INPUT: int = 20
    RERANKER_TOP_K_OUTPUT: int = 5
    RERANKER_THRESHOLD: float = 0.65

    # LLM Configuration (Ollama)
    OLLAMA_HOST: str = "localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:14b-instruct-q4_K_M"
    OLLAMA_NUM_CTX: int = 32768
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_P: float = 0.9

    # Chunking Configuration
    CHUNK_SIZE: int = 512  # Characters
    CHUNK_OVERLAP: int = 50
    CHUNK_MAX_TABLE_SIZE: int = 2048

    # Search Configuration
    SEARCH_TOP_K: int = 20
    SEARCH_NPROBE: int = 128
    HYBRID_SEARCH_VECTOR_WEIGHT: float = 0.7
    HYBRID_SEARCH_KEYWORD_WEIGHT: float = 0.3

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 10

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_CONTENT_TYPES: List[str] = ["application/pdf"]

    # Monitoring
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090
    ENABLE_TRACING: bool = True
    JAEGER_HOST: Optional[str] = None
    JAEGER_PORT: int = 6831

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

### 2.3 Service Layer Pattern

**Base Service Interface:**

```python
# app/services/ocr/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from pathlib import Path

class OCRResult:
    """OCR result data class"""
    def __init__(
        self,
        text: str,
        confidence: float,
        pages: int,
        metadata: Dict[str, Any]
    ):
        self.text = text
        self.confidence = confidence
        self.pages = pages
        self.metadata = metadata


class BaseOCREngine(ABC):
    """Base OCR engine interface"""

    @abstractmethod
    async def process(
        self,
        pdf_path: Path,
        language: str = "ja"
    ) -> OCRResult:
        """Process PDF and extract text"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Get OCR confidence score"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if OCR engine is healthy"""
        pass
```

**YomiToku Implementation:**

```python
# app/services/ocr/yomitoku.py
from .base import BaseOCREngine, OCRResult
from pathlib import Path
import asyncio

class YomiTokuOCREngine(BaseOCREngine):
    """YomiToku OCR engine implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path", "/models/yomitoku")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YomiToku model"""
        # Lazy import
        from yomitoku import YomiTokuOCR
        self.ocr_engine = YomiTokuOCR(self.model_path)

    async def process(
        self,
        pdf_path: Path,
        language: str = "ja"
    ) -> OCRResult:
        """Process PDF with YomiToku"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_sync,
            pdf_path,
            language
        )
        return result

    def _process_sync(
        self,
        pdf_path: Path,
        language: str
    ) -> OCRResult:
        """Synchronous processing"""
        # Process PDF
        output = self.ocr_engine.process_pdf(
            pdf_path,
            language=language,
            output_format="markdown",
            preserve_tables=True
        )

        return OCRResult(
            text=output["markdown"],
            confidence=output["average_confidence"],
            pages=output["page_count"],
            metadata={
                "engine": "yomitoku",
                "page_confidences": output["page_confidences"],
                "tables_detected": output["tables_count"]
            }
        )

    def get_confidence(self) -> float:
        return 0.0  # Extracted from result

    async def health_check(self) -> bool:
        try:
            # Test processing
            return True
        except Exception:
            return False
```

### 2.4 Repository Pattern

**Base Repository:**

```python
# app/db/repositories/base.py
from typing import Generic, TypeVar, Type, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

ModelType = TypeVar("ModelType")

class BaseRepository(Generic[ModelType]):
    """Base repository with CRUD operations"""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get(self, id: str) -> Optional[ModelType]:
        """Get by ID"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get all with pagination"""
        result = await self.session.execute(
            select(self.model)
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def create(self, obj: ModelType) -> ModelType:
        """Create new record"""
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def update(self, id: str, **kwargs) -> Optional[ModelType]:
        """Update record"""
        await self.session.execute(
            update(self.model)
            .where(self.model.id == id)
            .values(**kwargs)
        )
        return await self.get(id)

    async def delete(self, id: str) -> bool:
        """Delete record"""
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0
```

**Document Repository:**

```python
# app/db/repositories/document_repo.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime

from app.db.base import Base
from app.models.document import Document
from .base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Document repository"""

    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)

    async def find_by_owner(
        self,
        owner_id: str,
        offset: int = 0,
        limit: int = 20
    ) -> List[Document]:
        """Find documents by owner"""
        result = await self.session.execute(
            select(self.model)
            .where(self.model.owner_id == owner_id)
            .where(self.model.deleted_at.is_(None))
            .order_by(self.model.uploaded_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def find_by_hash(self, file_hash: str) -> Optional[Document]:
        """Find document by file hash (deduplication)"""
        result = await self.session.execute(
            select(self.model)
            .where(self.model.file_hash == file_hash)
            .where(self.model.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Update document processing status"""
        return await self.update(
            document_id,
            status=status,
            error_message=error_message,
            updated_at=datetime.utcnow()
        )

    async def count_by_owner(self, owner_id: str) -> int:
        """Count documents by owner"""
        result = await self.session.execute(
            select(func.count(self.model.id))
            .where(self.model.owner_id == owner_id)
            .where(self.model.deleted_at.is_(None))
        )
        return result.scalar()
```

---

## 3. Import Conventions

### 3.1 Absolute Imports

Use absolute imports from the `app` package:

```python
# ✅ Good
from app.services.auth_service import AuthService
from app.models.user import UserCreate
from app.db.repositories.user_repo import UserRepository

# ❌ Avoid relative imports
from ..services.auth_service import AuthService
from .models.user import UserCreate
```

### 3.2 Import Grouping

Organize imports in this order:

```python
# 1. Standard library imports
import os
import asyncio
from pathlib import Path
from typing import Optional, List

# 2. Third-party imports
from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

# 3. Application imports
from app.core.config import settings
from app.models.user import User
from app.db.repositories.user_repo import UserRepository
```

---

## 4. Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| **Modules** | `snake_case` | `auth_service.py` |
| **Classes** | `PascalCase` | `AuthService`, `UserRepository` |
| **Functions** | `snake_case` | `create_user()`, `get_document()` |
| **Variables** | `snake_case` | `user_id`, `document_count` |
| **Constants** | `UPPER_SNAKE_CASE` | `MAX_UPLOAD_SIZE`, `DEFAULT_PAGE_SIZE` |
| **Private** | `_leading_underscore` | `_internal_function()` |

---

## 5. Dependencies Management

### 5.1 requirements.txt

```
# Web Framework
fastapi==0.108.0
uvicorn[standard]==0.25.0
python-multipart==0.0.6

# Database
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
alembic==1.13.1
psycopg2-binary==2.9.9

# Vector Database
pymilvus==2.4.0

# Object Storage
minio==7.2.0

# Cache
redis==5.0.1
hiredis==2.3.2

# Task Queue
celery[redis]==5.3.6
flower==2.0.1

# AI/ML
transformers==4.36.0
torch==2.2.0+cu121
sentencepiece==0.1.99
protobuf==4.25.1

# OCR
yomitoku==1.2.0
paddlepaddle-gpu==2.6.0
paddleocr==2.7.0.3
opencv-python-headless==4.9.0.80

# LLM
ollama==0.1.6
openai==1.10.0  # For API compatibility

# Utilities
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
pyyaml==6.0.1

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-instrumentation-fastapi==0.43b0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
httpx==0.26.0

# Development
black==24.1.1
ruff==0.1.14
mypy==1.8.0
```

### 5.2 pyproject.toml

```toml
[tool.poetry]
name = "ocr-rag"
version = "1.0.0"
description = "Japanese OCR RAG System"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
# ... dependencies from requirements.txt ...

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.4"
black = "^24.1.1"
ruff = "^0.1.14"
mypy = "^1.8.0"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## 6. Environment Variables

### 6.1 .env.example

```bash
# Application
APP_NAME=Japanese OCR RAG System
DEBUG=false
ENVIRONMENT=development

# Security (generate with: openssl rand -hex 32)
SECRET_KEY=your-secret-key-here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=raguser
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_metadata

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Ollama
OLLAMA_HOST=ollama:11434
OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M
```

---

## 7. Configuration Files

### 7.1 docker-compose.yml

```yaml
version: '3.8'

services:
  # Application
  app:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.app
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - POSTGRES_HOST=postgres
      - MILVUS_HOST=milvus
      - MINIO_ENDPOINT=minio:9000
      - REDIS_HOST=redis
      - OLLAMA_HOST=ollama:11434
    volumes:
      - ./models:/models:ro
      - ./data:/app/data
    depends_on:
      - postgres
      - milvus
      - minio
      - redis
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: rag_metadata
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Milvus
  milvus:
    image: milvusdb/milvus:v2.4.0-gpu
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # MinIO
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Ollama
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # etcd (for Milvus)
  etcd:
    image: quay.io/coreos/etcd:v3.5.9
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
    volumes:
      - etcd_data:/etcd

volumes:
  postgres_data:
  milvus_data:
  minio_data:
  redis_data:
  ollama_models:
  etcd_data:
```

---

**END OF PROJECT STRUCTURE DESIGN**
