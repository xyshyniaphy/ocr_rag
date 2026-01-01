# Japanese OCR RAG System

A production-grade Retrieval-Augmented Generation (RAG) system optimized for Japanese PDF document processing. Built with FastAPI, Streamlit, and state-of-the-art OCR and embedding models.

## Features

- **Japanese OCR**: YomiToku (primary) and PaddleOCR-VL (fallback) for accurate text extraction
- **Semantic Search**: Sarashina-Embedding-v1-1B for Japanese text embeddings
- **RAG Pipeline**: Hybrid vector + keyword search with LLM-powered responses
- **Privacy-First**: Air-gapped deployment support, no external API dependencies
- **GPU Accelerated**: Optimized for single GPU (RTX 4090 24GB)
- **Production Ready**: Docker-based deployment with monitoring and observability

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ocr_rag

# Start development environment
./dev.sh up

# View logs
./dev.sh logs

# Stop services
./dev.sh down
```

## Access Points

| Service | URL |
|---------|-----|
| FastAPI Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Admin UI | http://localhost:8501 |
| MinIO Console | http://localhost:9001 |
| Flower (Celery) | http://localhost:9100 |

## Default Credentials

| Service | Username | Password |
|----------|----------|----------|
| Admin User | admin@example.com | admin123 |
| MinIO | minioadmin | minioadmin |

## Database Initialization

The system automatically creates database tables on first startup in development mode. The following tables are created:

- `users` - User accounts with authentication
- `documents` - Document metadata and status
- `chunks` - Text chunks with embeddings metadata
- `queries` - RAG query history and responses
- `permissions` - Document-level access control

### Creating Admin User

After starting the services, create the admin user:

```bash
docker exec ocr-rag-app-dev python /app/scripts/seed_admin.py
```

## Architecture

**Document Ingestion:**
```
PDF Upload → Validation → OCR (YomiToku) → Markdown → Chunking
→ Embedding (Sarashina) → Milvus → PostgreSQL → Complete
```

**Query Processing:**
```
User Query → Embedding → Hybrid Search (Milvus + PostgreSQL)
→ Reranking (Llama-3.2-NV) → Context Assembly → LLM (Qwen)
→ Response with Sources
```

### Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Backend Framework | FastAPI | High-performance async API |
| Frontend Framework | Streamlit | Admin UI and monitoring |
| OCR (Primary) | YomiToku | Japanese optimized OCR |
| OCR (Fallback) | PaddleOCR-VL | Multilingual OCR fallback |
| Embedding | Sarashina-Embedding-v1-1B | 1792D Japanese embeddings |
| Reranker | Llama-3.2-NV-RerankQA-1B-v2 | Cross-encoder for relevance |
| LLM | Qwen2.5-14B (Ollama) | Generation via Ollama |
| Vector DB | Milvus 2.4+ | HNSW index, COSINE similarity |
| Metadata DB | PostgreSQL 16+ | Document metadata and chunks |
| Object Storage | MinIO | PDF file storage |
| Cache | Redis | Query and embedding cache |
| Task Queue | Celery | Async document processing |

## Development

### Project Structure

```
ocr_rag/
├── backend/                   # Python Backend (FastAPI)
│   ├── main.py               # Application entry point
│   ├── core/                  # Configuration, logging, security
│   ├── models/                # SQLAlchemy & Pydantic models
│   ├── api/v1/               # REST API endpoints
│   ├── db/                    # Database (PostgreSQL + Milvus)
│   │   ├── repositories/     # Repository pattern
│   │   └── vector/           # Vector database client
│   ├── storage/              # MinIO client
│   ├── services/             # Business logic
│   │   ├── ocr/             # OCR processing (YomiToku)
│   │   ├── embedding/       # Embedding (Sarashina)
│   │   ├── retrieval/       # Vector + Keyword search
│   │   ├── reranker/        # Reranking (Llama-3.2-NV)
│   │   ├── llm/            # LLM generation (Qwen)
│   │   └── rag/            # RAG orchestration
│   ├── tasks/                # Celery tasks
│   └── utils/                # Utilities
├── frontend/                  # Streamlit Admin UI
├── scripts/                   # Utility scripts
│   └── download_reranker_model.py  # Pre-download reranker
├── Dockerfile.base            # Base ML image
├── Dockerfile.app             # Application image
├── docker-compose.dev.yml     # Development config
└── dev.sh                     # Dev environment manager
```

### Environment Variables

Key environment variables (see `.env` for full list):

```bash
SECRET_KEY=your-secret-key-here
POSTGRES_PASSWORD=your-password
OLLAMA_HOST=http://ollama:11434
OCR_ENGINE=yomitoku
```

## Production Deployment

```bash
# Build production images
docker compose -f docker-compose.prd.yml build

# Start production services
docker compose -f docker-compose.prd.yml up -d
```

## Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Query (GPU) | <2s | 5s (95th percentile) |
| OCR Processing | <10s/page | 30s/page |
| Embedding | <50ms/chunk | 200ms/chunk |
| Vector Search | <100ms | 500ms |
| Reranking | <200ms (20 docs) | 500ms |

## Troubleshooting

**Issue**: OCR confidence low
- Check GPU memory: `nvidia-smi`
- Try PaddleOCR fallback: Set `OCR_ENGINE=paddleocr`

**Issue**: Query latency high
- Check GPU utilization
- Increase Milvus `nprobe` parameter
- Enable query caching

**Issue**: Reranker model download slow
- Model downloads on first use (~2GB)
- Pre-download with: `docker exec ocr-rag-app-dev python /app/scripts/download_reranker_model.py`
- Subsequent runs use cached model from volume

**Issue**: Out of memory
- Reduce `EMBEDDING_BATCH_SIZE`
- Reduce `RERANKER_BATCH_SIZE`
- Use smaller LLM model (Q4_K_M quantization)
- Add more GPUs

## Model Storage

Models are stored in separate locations:

| Model | Location | Storage Type |
|-------|----------|--------------|
| Sarashina Embedding | `/app/models/sarashina/` | Base image (pre-built) |
| YomiToku OCR | Library managed | Runtime cache |
| Llama-3.2-NV Reranker | `/app/reranker_models/` | Docker volume (cached) |
| Qwen LLM | `/root/.ollama` | Docker volume (Ollama) |

**Note**: The reranker model is downloaded on first use and cached in a Docker volume. Pre-download with the provided script for faster startup.

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]
