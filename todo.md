# Japanese OCR RAG System - Implementation Tasks

## Overview

This document tracks the remaining implementation tasks for the Japanese OCR RAG System. The core infrastructure (database, API, authentication) is complete, but the RAG pipeline services are not yet implemented.

## Priority Levels

- 🔴 **P0 - Critical**: Core functionality required for basic operation
- 🟡 **P1 - High**: Important features for production use
- 🟢 **P2 - Medium**: Nice-to-have enhancements
- ⚪ **P3 - Low**: Optional/future features

---

## 🔴 P0 - Core RAG Pipeline

### 1. OCR Service (`backend/services/ocr/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-02)
**Description:** Implement OCR processing for Japanese PDFs

- [x] **YomiToku Integration** (Primary)
  - [x] Load YomiToku model from base image
  - [x] Process PDF pages with YomiToku
  - [x] Extract text with bounding boxes
  - [x] Return confidence scores
  - [x] Handle multi-column layouts


- [x] **Output Format**
  - [x] Convert OCR output to Markdown
  - [x] Preserve document structure (headers, tables, lists)
  - [x] Export to MinIO object storage

**Files:** `backend/services/ocr/__init__.py`, `backend/services/ocr/yomitoku.py`,

---

### 2. Text Chunking Service (`backend/services/processing/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-02)
**Description:** Split documents into searchable chunks

- [x] **Chunking Strategy**
  - [x] Japanese-aware chunking (sentence boundaries)
  - [x] Configurable chunk size (default: 512 chars)
  - [x] Configurable overlap (default: 50 chars)
  - [x] Preserve metadata (page numbers, sections)

- [x] **Chunk Types**
  - [x] Text chunks (paragraphs)
  - [ ] Table detection and extraction (future)
  - [ ] Header/section boundary detection (future)

**Separators:** `["\n\n", "\n", "。", "！", "？", "；", "、"]`

**Files:** `backend/services/processing/__init__.py`, `backend/services/processing/chunker.py`, `backend/services/processing/chunking/`

---

### 3. Embedding Service (`backend/services/embedding/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-02)
**Description:** Generate embeddings for text chunks

- [x] **Model Loading**
  - [x] Load Sarashina-Embedding-v1-1B (1792D)
  - [x] GPU memory management
  - [x] Batch processing configuration

- [x] **Embedding Generation**
  - [x] Process chunks in batches (configurable size)
  - [x] Handle GPU OOM errors gracefully
  - [x] Store embeddings in Milvus

- [x] **Quality Metrics**
  - [x] Track embedding dimensions
  - [x] Log processing time per chunk

**Files:** `backend/services/embedding/__init__.py`, `backend/services/embedding/service.py`, `backend/services/embedding/models.py`

---

### 4. Retrieval Service (`backend/services/retrieval/`)

**Status:** Stub only
**Description:** Search vector database for relevant chunks

- [ ] **Vector Search**
  - [ ] Connect to Milvus collection
  - [ ] Query by embedding similarity
  - [ ] Configurable top_k (1-20)
  - [ ] Return document metadata

- [ ] **Hybrid Search** (Optional)
  - [ ] Combine vector + keyword search (BM25)
  - [ ] Score fusion strategies

**Files:** `backend/services/retrieval/__init__.py`, `backend/services/retrieval/vector_search.py`

---

### 5. Reranking Service (`backend/services/retrieval/`)

**Status:** Mock only
**Description:** Re-rank search results for relevance

- [ ] **Model Loading**
  - [ ] Load Llama-3.2-NV-RerankQA-1B-v2
  - [ ] GPU memory allocation

- [ ] **Reranking Logic**
  - [ ] Take query + top_k results
  - [ ] Return re-ranked results with scores

**Files:** `backend/services/retrieval/reranker.py`

---

### 6. LLM Service (`backend/services/llm/`)

**Status:** Stub only
**Description:** Generate answers using Qwen LLM

- [ ] **Ollama Integration**
  - [ ] Connect to Ollama API
  - [ ] Model: `qwen2.5:14b-instruct-q4_K_M`
  - [ ] Handle streaming responses

- [ ] **Prompt Engineering**
  - [ ] System prompt for RAG
  - [ ] Context assembly (include sources)
  - [ ] Japanese language handling

- [ ] **Response Generation**
  - [ ] Generate answer with sources
  - [ ] Format citations
  - [ ] Handle edge cases (no results, low confidence)

**Files:** `backend/services/llm/__init__.py`, `backend/services/llm/ollama.py`

---

### 7. RAG Orchestration (`backend/services/rag/`)

**Status:** Stub only
**Description:** Orchestrate the full RAG pipeline

- [ ] **Pipeline Implementation**
  ```python
  def rag_pipeline(query: str, top_k: int = 5) -> RAGResponse:
      1. Embed query
      2. Vector search (Milvus)
      3. Rerank results
      4. Assemble context
      5. Generate answer (LLM)
      6. Return with sources
  ```

- [ ] **Error Handling**
  - [ ] Graceful degradation
  - [ ] Fallback strategies

- [ ] **Performance Tracking**
  - [ ] Stage-level timing
  - [ ] Confidence scores

**Files:** `backend/services/rag/__init__.py`, `backend/services/rag/pipeline.py`

---

## 🟡 P1 - High Priority Features

### 8. Background Task Processing

**Status:** ✅ **IMPLEMENTED** (2026-01-02)

- [x] **Celery Tasks** (`backend/tasks/document_tasks.py`)
  - [x] `process_document()` - Full OCR → Chunking → Embedding → Milvus
  - [ ] `reprocess_document()` - Re-run failed documents (stub)
  - [ ] `cleanup_old_documents()` - Delete old files (stub)

- [x] **Task Status Updates**
  - [x] Update document status in DB
  - [ ] Send WebSocket notifications (future)
  - [x] Handle errors and retries

---

### 9. Document Processing Workflow

**Status:** ✅ **IMPLEMENTED** (2026-01-02)

- [x] **Upload Handler** → Triggers Celery task automatically
- [x] **Status Tracking** → Document status updates in DB (pending → processing → completed/failed)
- [x] **Error Handling** → Proper error messages to logs and DB
- [ ] **Real-time WebSocket updates** (future)

**Test Result:** Document processing successfully completed with 95.2% OCR confidence, 2 pages, 3 chunks

---

### 10. WebSocket Query Streaming

**Status:** Endpoint exists, returns mock data

- [ ] **Integrate RAG Pipeline**
  - [ ] Replace mock `handle_query()` with real pipeline
  - [ ] Stream tokens as they're generated
  - [ ] Send sources when ready
  - [ ] Send completion signal

---

## 🟢 P2 - Medium Priority Features

### 11. Permission System (ACL)

**Status:** Database model exists, not enforced

- [ ] **Middleware**
  - [ ] Check document permissions on API calls
  - [ ] Enforce owner_id in document upload
  - [ ] Admin bypass

- [ ] **UI Features**
  - [ ] Permission management UI
  - [ ] Share documents with other users

---

### 12. User Management Enhancements

- [ ] **Profile Management**
  - [ ] Update email/name
  - [ ] Change password
  - [ ] User preferences

- [ ] **Admin Features**
  - [ ] Create/edit/delete users
  - [ ] Activity logs
  - [ ] Usage statistics

---

### 13. Advanced Query Features

- [ ] **Query History**
  - [ ] Save past queries
  - [ ] Re-run queries
  - [ ] Export results

- [ ] **Document Filtering**
  - [ ] Filter by tags/categories
  - [ ] Date range filters
  - [ ] Saved searches

---

### 14. Monitoring & Observability

- [ ] **Prometheus Metrics**
  - [ ] Query latency histogram
  - [ ] OCR processing time
  - [ ] Error rates

- [ ] **Logging**
  - [ ] Structured JSON logs (partially done)
  - [ ] Request tracing
  - [ ] Performance profiling

---

## ⚪ P3 - Low Priority / Optional

### 15. Streamlit UI Enhancements

- [ ] **Session Persistence** (Known limitation)
  - [ ] Use localStorage for tokens
  - [ ] Auto-reconnect on refresh

- [ ] **UI Polish**
  - [ ] Better error messages
  - [ ] Loading animations
  - [ ] Toast notifications

---

### 16. Advanced Features

- [ ] **Multi-modal Query**
  - [ ] Query by image
  - [ ] Table extraction

- [ ] **Document Comparison**
  - [ ] Compare versions
  - [ ] Diff viewer

- [ ] **Export Features**
  - [ ] Export query results
  - [ ] Download documents with annotations

---

## Database Migrations

- [ ] **Alembic Setup**
  - [ ] Initialize Alembic
  - [ ] Create initial migration
  - [ ] Migration rollback testing

---

## Testing

- [ ] **Unit Tests**
  - [ ] OCR service tests
  - [ ] Embedding service tests
  - [ ] RAG pipeline tests

- [ ] **Integration Tests**
  - [ ] API endpoint tests
  - [ ] Database tests
  - [ ] WebSocket tests

- [ ] **E2E Tests**
  - [ ] Upload → Process → Query workflow

---

## Documentation

- [ ] **API Documentation**
  - [ ] OpenAPI spec completion
  - [ ] Example requests/responses

- [ ] **Developer Guide**
  - [ ] Service architecture
  - [ ] Adding new OCR models
  - [ ] Customizing chunking

---

## Implementation Order Recommendation

### Phase 1: Core Pipeline (Must Have)
1. ✅ ~~OCR Service (YomiToku)~~ **COMPLETED**
2. ✅ ~~Chunking Service~~ **COMPLETED**
3. ✅ ~~Embedding Service (Sarashina)~~ **COMPLETED**
4. [ ] Vector Search (Milvus)
5. [ ] LLM Service (GLM/Qwen)
6. [ ] RAG Orchestration

### Phase 2: Production Readiness
7. ✅ ~~Background Task Processing (Celery)~~ **COMPLETED**
8. ✅ ~~Document Processing Workflow~~ **COMPLETED**
9. [ ] WebSocket Streaming
10. [ ] Error Handling & Logging (partial)

### Phase 3: Enhancements
11. [ ] Permission System
12. [ ] Advanced Query Features
13. [ ] Monitoring & Metrics
14. [ ] Testing & Documentation

---

## Current API Endpoints Status

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/v1/auth/login` | POST | ✅ Implemented |
| `/api/v1/auth/register` | POST | ✅ Implemented |
| `/api/v1/auth/refresh` | POST | ✅ Implemented |
| `/api/v1/auth/me` | GET | ✅ Implemented |
| `/api/v1/auth/logout` | POST | ✅ Implemented |
| `/api/v1/documents/upload` | POST | ✅ **Full pipeline implemented** |
| `/api/v1/documents` | GET | ✅ Implemented |
| `/api/v1/documents/{id}` | GET | ✅ Implemented |
| `/api/v1/documents/{id}` | DELETE | ✅ Implemented |
| `/api/v1/query` | POST | ❌ Mock response |
| `/api/v1/documents/search` | GET | ✅ Implemented |
| `/api/v1/admin/stats` | GET | ✅ Implemented |
| `/api/v1/admin/users` | GET | ⚠️ Basic only |
| `/api/v1/stream/ws` | WebSocket | ❌ Mock response |

---

## Notes

- **ML Models** are included in the Docker base image (`Dockerfile.base`)
- **Ollama** container has `qwen2.5:14b` model available
- **Milvus** vector database is running and configured
- **PostgreSQL** database schema is created

---

## Quick Start Implementation

To implement the RAG pipeline:

```bash
# 1. Start with OCR service
# Edit: backend/services/ocr/yomitoku.py

# 2. Then embedding
# Edit: backend/services/embedding/sarashina.py

# 3. Then retrieval
# Edit: backend/services/retrieval/vector_search.py

# 4. Then LLM
# Edit: backend/services/llm/ollama.py

# 5. Finally orchestrate
# Edit: backend/services/rag/pipeline.py

# 6. Wire up in API
# Edit: backend/api/v1/query.py
```

---

**Generated:** 2026-01-01
**Last Updated:** 2026-01-02

## Recent Updates (2026-01-02)

### Completed Implementation:
1. ✅ **OCR Service** - YomiToku integration fully working with 95%+ confidence
2. ✅ **Text Chunking** - Japanese-aware chunking with configurable parameters
3. ✅ **Embedding Service** - Sarashina 1792D embeddings with GPU support
4. ✅ **Background Processing** - Celery worker handling document processing pipeline
5. ✅ **Document Upload Pipeline** - End-to-end: Upload → OCR → Chunk → Embed → Milvus → PostgreSQL

### Remaining P0 Tasks:
- **Retrieval Service** - Vector search from Milvus
- **Reranking Service** - Llama-3.2-NV-RerankQA integration
- **LLM Service** - GLM-4.5-Air or Qwen integration
- **RAG Orchestration** - Full pipeline integration

### Next Steps:
1. Implement vector search in Milvus
2. Add reranking for better relevance
3. Integrate LLM service (GLM/Ollama)
4. Wire up full RAG pipeline to query endpoint
