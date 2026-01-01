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

**Status:** Stub only
**Description:** Implement OCR processing for Japanese PDFs

- [ ] **YomiToku Integration** (Primary)
  - [ ] Load YomiToku model from base image
  - [ ] Process PDF pages with YomiToku
  - [ ] Extract text with bounding boxes
  - [ ] Return confidence scores
  - [ ] Handle multi-column layouts

- [ ] **PaddleOCR Integration** (Fallback)
  - [ ] Load PaddleOCR-VL model
  - [ ] Fallback logic when YomiToku fails
  - [ ] Confidence threshold validation

- [ ] **Output Format**
  - [ ] Convert OCR output to Markdown
  - [ ] Preserve document structure (headers, tables, lists)
  - [ ] Export to MinIO object storage

**Files:** `backend/services/ocr/__init__.py`, `backend/services/ocr/yomitoku.py`, `backend/services/ocr/paddleocr.py`

---

### 2. Text Chunking Service (`backend/services/processing/`)

**Status:** Stub only
**Description:** Split documents into searchable chunks

- [ ] **Chunking Strategy**
  - [ ] Japanese-aware chunking (sentence boundaries)
  - [ ] Configurable chunk size (default: 512 chars)
  - [ ] Configurable overlap (default: 50 chars)
  - [ ] Preserve metadata (page numbers, sections)

- [ ] **Chunk Types**
  - [ ] Text chunks (paragraphs)
  - [ ] Table detection and extraction
  - [ ] Header/section boundary detection

**Separators:** `["\n\n", "\n", "。", "！", "？", "；", "、"]`

**Files:** `backend/services/processing/__init__.py`, `backend/services/processing/chunker.py`

---

### 3. Embedding Service (`backend/services/embedding/`)

**Status:** Stub only
**Description:** Generate embeddings for text chunks

- [ ] **Model Loading**
  - [ ] Load Sarashina-Embedding-v1-1B (768D)
  - [ ] GPU memory management
  - [ ] Batch processing configuration

- [ ] **Embedding Generation**
  - [ ] Process chunks in batches (configurable size)
  - [ ] Handle GPU OOM errors gracefully
  - [ ] Store embeddings in Milvus

- [ ] **Quality Metrics**
  - [ ] Track embedding dimensions
  - [ ] Log processing time per chunk

**Files:** `backend/services/embedding/__init__.py`, `backend/services/embedding/sarashina.py`

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

**Status:** Task definitions exist but not implemented

- [ ] **Celery Tasks** (`backend/tasks/document_tasks.py`)
  - [ ] `process_document()` - Full OCR → Chunking → Embedding
  - [ ] `reprocess_document()` - Re-run failed documents
  - [ ] `cleanup_old_documents()` - Delete old files

- [ ] **Task Status Updates**
  - [ ] Update document status in DB
  - [ ] Send WebSocket notifications
  - [ ] Handle errors and retries

---

### 9. Document Processing Workflow

**Status:** Upload works, processing doesn't

- [ ] **Upload Handler** → Update to trigger Celery task
- [ ] **Status Tracking** → Real-time updates via WebSocket
- [ ] **Error Handling** → Proper error messages to UI

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
1. OCR Service (YomiToku)
2. Chunking Service
3. Embedding Service (Sarashina)
4. Vector Search (Milvus)
5. LLM Service (Qwen/Ollama)
6. RAG Orchestration

### Phase 2: Production Readiness
7. Background Task Processing (Celery)
8. Document Processing Workflow
9. WebSocket Streaming
10. Error Handling & Logging

### Phase 3: Enhancements
11. Permission System
12. Advanced Query Features
13. Monitoring & Metrics
14. Testing & Documentation

---

## Current API Endpoints Status

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/v1/auth/login` | POST | ✅ Implemented |
| `/api/v1/auth/register` | POST | ✅ Implemented |
| `/api/v1/auth/refresh` | POST | ✅ Implemented |
| `/api/v1/auth/me` | GET | ✅ Implemented |
| `/api/v1/auth/logout` | POST | ✅ Implemented |
| `/api/v1/documents/upload` | POST | ⚠️ Upload only |
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
**Last Updated:** During database initialization
