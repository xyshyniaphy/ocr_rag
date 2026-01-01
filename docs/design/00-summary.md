# Design Documentation Summary

**Project:** Japanese OCR RAG System
**Version:** 1.0
**Date:** 2026-01-01

---

## Document Overview

This directory contains the complete system design documentation for the Japanese OCR RAG System - a production-grade Retrieval-Augmented Generation system optimized for Japanese PDF document processing.

### Design Documents

| Document | Description |
|----------|-------------|
| **01-system-architecture.md** | High-level system architecture, component specifications, data flows, scalability design, security architecture, and deployment patterns |
| **02-api-specification.md** | Complete REST API and WebSocket specifications with request/response formats, error handling, rate limiting, and authentication flows |
| **03-database-schema.md** | Milvus vector database schema and PostgreSQL metadata database schema with tables, indexes, functions, and triggers |
| **04-project-structure.md** | Directory structure, module organization, import conventions, naming conventions, and configuration management |

---

## Quick Reference

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway (NGINX)                          │
├─────────────────────────────────────────────────────────────────┤
│                    Application (FastAPI)                        │
│                    LangChain RAG Orchestration                  │
├─────────────────────────────────────────────────────────────────┤
│     OCR (YomiToku)    │   Embedding (Sarashina)   │   LLM     │
│                       │   Reranker (Llama-3.2)   │  (Qwen)    │
├─────────────────────────────────────────────────────────────────┤
│   Milvus (Vector DB)  │   PostgreSQL (Metadata)  │  MinIO     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technology Stack

| Component | Technology |
|-----------|------------|
| **API Gateway** | NGINX |
| **Backend** | FastAPI |
| **Admin UI** | Streamlit |
| **OCR** | YomiToku (primary), PaddleOCR-VL (fallback) |
| **Embedding** | Sarashina-Embedding-v1-1B (768D) |
| **Reranker** | Llama-3.2-NV-RerankQA-1B-v2 |
| **LLM** | Qwen2.5-14B (Ollama) |
| **Vector DB** | Milvus 2.4+ (IVF_FLAT index) |
| **Metadata DB** | PostgreSQL 16+ |
| **Object Storage** | MinIO |
| **Cache** | Redis |
| **Task Queue** | Celery |

### Data Flow Summary

**Document Ingestion:**
```
PDF Upload → Validation → OCR (YomiToku) → Markdown → Chunking
→ Embedding (Sarashina) → Milvus → PostgreSQL Metadata → Complete
```

**Query Processing:**
```
User Query → Embedding → Hybrid Search (Vector + BM25)
→ Reranking (Top 20→5) → Context Assembly → LLM (Qwen)
→ Response with Sources → Cache Result
```

### API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | User authentication |
| `/api/v1/auth/refresh` | POST | Refresh access token |
| `/api/v1/documents/upload` | POST | Upload PDF document |
| `/api/v1/documents/{id}` | GET | Get document details |
| `/api/v1/documents/{id}/status` | GET | Get processing status |
| `/api/v1/documents` | GET | List documents with filters |
| `/api/v1/documents/{id}` | DELETE | Delete document |
| `/api/v1/query` | POST | Submit RAG query |
| `/api/v1/stream` | WebSocket | Streaming query interface |
| `/api/v1/health` | GET | System health check |
| `/api/v1/stats` | GET | System statistics |

### Database Schema Summary

**PostgreSQL Tables:**
- `users` - User accounts and authentication
- `documents` - Document metadata and status
- `chunks` - Text chunk metadata
- `queries` - Query history and feedback
- `permissions` - Document-level ACLs
- `audit_log` - Security audit trail

**Milvus Collections:**
- `document_chunks` - 768D embeddings with metadata

### Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Query (GPU) | <2s | 5s (95th percentile) |
| OCR Processing | <10s/page | 30s/page |
| Embedding | <50ms/chunk | 200ms/chunk |
| Vector Search | <100ms | 500ms |

---

## Implementation Guidance

### Development Environment Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/xyshyniaphy/ocr_rag.git
   cd ocr_rag
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start services:**
   ```bash
   docker-compose up -d
   ```

6. **Run application:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Key Implementation Considerations

1. **Japanese Text Processing:**
   - Use Unicode NFKC normalization
   - Apply Japanese-specific separators for chunking
   - Handle vertical text (縦書き) correctly

2. **OCR Fallback Strategy:**
   - Primary: YomiToku (confidence >= 0.85)
   - Fallback: PaddleOCR-VL (confidence < 0.80 or multi-language)

3. **Hybrid Search:**
   - 70% semantic (vector search with Milvus)
   - 30% keyword (BM25 with PostgreSQL)
   - Rerank top 20 → top 5 results

4. **LLM Prompt Engineering:**
   - Japanese prompt template with citation requirements
   - Strict constraints against hallucination
   - Source attribution mandatory

5. **Error Handling:**
   - Comprehensive error logging
   - User-friendly error messages
   - Graceful degradation when services are unavailable

### Testing Strategy

**Unit Tests:**
- Service layer (OCR, embedding, retrieval, LLM)
- Repository layer (database operations)
- Utility functions

**Integration Tests:**
- API endpoints
- Database operations
- Background tasks

**End-to-End Tests:**
- Document upload flow
- Query processing flow
- Error scenarios

### Monitoring & Observability

**Metrics:**
- Request latency (histogram)
- GPU utilization (gauge)
- Cache hit rate (gauge)
- Error rate (counter)

**Logging:**
- Structured JSON logging
- Log levels: DEBUG, INFO, WARNING, ERROR
- Elasticsearch + Kibana for aggregation

**Tracing:**
- OpenTelemetry distributed tracing
- Jaeger for trace visualization
- Sample rate: 10%

---

## Security Checklist

- [ ] TLS 1.3 enabled for all communications
- [ ] JWT tokens with appropriate expiry
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CSRF protection (SameSite cookies)
- [ ] File upload validation
- [ ] PII detection and redaction
- [ ] Audit logging enabled
- [ ] Role-based access control (RBAC)
- [ ] Document-level permissions (ACLs)

---

## Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Milvus collections created
- [ ] MinIO buckets created
- [ ] Models downloaded/configured
- [ ] SSL certificates obtained
- [ ] Firewall rules configured

### Production Deployment

- [ ] Docker images built and pushed
- [ ] Kubernetes manifests applied
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Alert rules configured
- [ ] Backup strategy in place
- [ ] Disaster recovery tested

---

## Troubleshooting

### Common Issues

**Issue:** OCR confidence low
- **Solution:** Check image quality, increase DPI, try PaddleOCR fallback

**Issue:** Query latency high
- **Solution:** Check GPU utilization, increase nprobe, enable caching

**Issue:** LLM hallucination
- **Solution:** Verify source context, adjust prompt template, lower temperature

**Issue:** Out of memory (GPU)
- **Solution:** Reduce batch sizes, use model quantization, add more GPUs

---

## Next Steps

1. **Phase 1 - Foundation:**
   - Set up project structure
   - Configure databases (PostgreSQL, Milvus, MinIO)
   - Implement base repository pattern
   - Set up authentication

2. **Phase 2 - Core Services:**
   - Implement OCR service (YomiToku + PaddleOCR)
   - Implement embedding service (Sarashina)
   - Implement chunking service
   - Set up task queue (Celery)

3. **Phase 3 - RAG Pipeline:**
   - Implement retrieval service (vector + keyword)
   - Implement reranker service
   - Implement LLM service (Qwen via Ollama)
   - Assemble RAG pipeline

4. **Phase 4 - API & UI:**
   - Implement FastAPI endpoints
   - Implement WebSocket streaming
   - Build Streamlit admin UI
   - Set up API gateway (NGINX)

5. **Phase 5 - Testing & Deployment:**
   - Write comprehensive tests
   - Set up monitoring (Prometheus, Grafana)
   - Configure CI/CD pipeline
   - Deploy to production

---

**For detailed specifications, please refer to the individual design documents listed above.**
