# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Japanese OCR RAG System** - a production-grade Retrieval-Augmented Generation system optimized for Japanese PDF document processing. The system is privacy-first (air-gapped deployment supported) and designed for legal, financial, academic, and enterprise document analysis.

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
│   │   ├── ocr/             # OCR processing (YomiToku, PaddleOCR)
│   │   ├── embedding/       # Embedding generation (Sarashina)
│   │   ├── retrieval/       # Vector + Keyword search
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
├── requirements-base.txt      # Base ML dependencies
└── requirements-app.txt       # Application dependencies
```

## Quick Start

```bash
# Development
make dev          # Start development environment
make logs         # View logs
make shell        # Open shell in container

# Production
make prod         # Start production environment
```

## Access Points

| Service | URL |
|---------|-----|
| FastAPI Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Admin UI | http://localhost:8501 |
| WebSocket | ws://localhost:8000/api/v1/stream/ws |
| MinIO Console | http://localhost:9001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Flower (Celery) | http://localhost:9100 |

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
| OCR (Primary) | YomiToku |
| OCR (Fallback) | PaddleOCR-VL |
| Embedding | Sarashina-Embedding-v1-1B (768D) |
| Reranker | Llama-3.2-NV-RerankQA-1B-v2 |
| LLM | Qwen2.5-14B (Ollama) |
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
- `OLLAMA_HOST` - Ollama LLM server address
- `OCR_ENGINE` - OCR engine selection (`yomitoku` or `paddleocr`)

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
   - SQLAlchemy models in `backend/models/` (database tables)
   - Pydantic models in `backend/models/` (request/response schemas)

3. **Services**: Business logic in `backend/services/`
   - OCR engines in `services/ocr/`
   - Embedding in `services/embedding/`
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
   - Contains ML models (Sarashina, YomiToku, PaddleOCR)
   - Heavy dependencies (PyTorch, Transformers)
   - Cached and rebuilt infrequently

2. **App Image** (`Dockerfile.app`):
   - Application code (`backend/`, `frontend/`)
   - Lightweight dependencies (FastAPI, Streamlit)
   - Rebuilt on code changes

## Development Workflow

1. **Make changes to code** in `backend/` or `frontend/`
2. **Restart containers**: `make down && make dev`
3. **Hot reload**: Development mode auto-reloads on Python changes
4. **View logs**: `make logs` or `docker-compose logs -f app`

## Troubleshooting

**Issue**: OCR confidence low
- Check GPU memory: `nvidia-smi`
- Try PaddleOCR fallback: Set `OCR_ENGINE=paddleocr`

**Issue**: Query latency high
- Check GPU utilization
- Increase Milvus `nprobe` parameter
- Enable query caching

**Issue**: Out of memory
- Reduce `EMBEDDING_BATCH_SIZE`
- Use smaller LLM model (Q4_K_M quantization)
- Add more GPUs

## Security Notes

- All API endpoints require authentication except `/health` and `/`
- JWT tokens expire after 15 minutes (access) or 7 days (refresh)
- File uploads limited to 50MB PDFs only
- Rate limiting: 60 queries/minute, 10 uploads/minute
