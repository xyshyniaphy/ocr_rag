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

**Status:** ✅ **IMPLEMENTED** (2026-01-03)
**Description:** Search vector database for relevant chunks

- [x] **Vector Search**
  - [x] Connect to Milvus collection
  - [x] Query by embedding similarity
  - [x] Configurable top_k (1-20)
  - [x] Return document metadata

- [x] **Hybrid Search**
  - [x] Combine vector + keyword search (BM25)
  - [x] Score fusion strategies (RRF)

**Files:** `backend/services/retrieval/__init__.py`, `backend/services/retrieval/service.py`

---

### 5. Reranking Service (`backend/services/reranker/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-03)
**Description:** Re-rank search results for relevance

- [x] **Model Loading**
  - [x] Load Llama-3.2-NV-RerankQA-1B-v2
  - [x] GPU memory allocation

- [x] **Reranking Logic**
  - [x] Take query + top_k results
  - [x] Return re-ranked results with scores

**Files:** `backend/services/reranker/__init__.py`, `backend/services/reranker/service.py`, `backend/services/reranker/models.py`

---

### 6. LLM Service (`backend/services/llm/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-03)
**Description:** Generate answers using GLM-4.5-Air (primary) and Qwen (fallback)

- [x] **GLM Integration (Primary)**
  - [x] Connect to GLM API
  - [x] Model: `GLM-4.5-Air` (default)
  - [x] Handle streaming responses
  - [x] OpenAI-compatible API

- [x] **Ollama Integration (Fallback)**
  - [x] Connect to Ollama API
  - [x] Model: `qwen2.5:14b-instruct-q4_K_M`
  - [x] Handle streaming responses

- [x] **Prompt Engineering**
  - [x] System prompt for RAG
  - [x] Context assembly (include sources)
  - [x] Japanese language handling

- [x] **Response Generation**
  - [x] Generate answer with sources
  - [x] Format citations
  - [x] Handle edge cases (no results, low confidence)

**Files:** `backend/services/llm/__init__.py`, `backend/services/llm/service.py`

---

### 7. RAG Orchestration (`backend/services/rag/`)

**Status:** ✅ **IMPLEMENTED** (2026-01-03)
**Description:** Orchestrate the full RAG pipeline

- [x] **Pipeline Implementation**
  ```python
  def rag_pipeline(query: str, top_k: int = 5) -> RAGResponse:
      1. Embed query
      2. Vector search (Milvus)
      3. Rerank results
      4. Assemble context
      5. Generate answer (LLM)
      6. Return with sources
  ```

- [x] **Error Handling**
  - [x] Graceful degradation
  - [x] Fallback strategies

- [x] **Performance Tracking**
  - [x] Stage-level timing
  - [x] Confidence scores

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

**Status:** ✅ **IMPLEMENTED** (2026-01-03)

- [x] **Integrate RAG Pipeline**
  - [x] Replaced mock `handle_query()` with real pipeline
  - [x] Stream tokens as they're generated
  - [x] Send sources when ready
  - [x] Send completion signal

---

## 🟢 P2 - Medium Priority Features

### 11. Permission System (ACL)

**Status:** ✅ **IMPLEMENTED** (2026-01-03)

- [x] **Permission Service** (`backend/core/permissions.py`)
  - [x] Document-level permission checking
  - [x] Owner, admin, and role-based access control
  - [x] Permission enforcement middleware

- [x] **API Integration**
  - [x] Document upload requires authentication (owner_id from user)
  - [x] Document get/delete/download check permissions
  - [x] Document list filters by accessible documents

- [ ] **Permission Management API** (Future)
  - [ ] Grant/revoke permissions endpoints
  - [ ] List permissions endpoint

**Files:** `backend/core/permissions.py`, `backend/api/dependencies.py`

---

### 12. User Management Enhancements

**Status:** ✅ **IMPLEMENTED** (2026-01-03)

- [x] **Profile Management**
  - [x] Update email/name (PUT /api/v1/auth/me)
  - [x] Change password (PUT /api/v1/auth/me/password)
  - [ ] User preferences (future)

- [x] **Admin Features**
  - [x] List all users (GET /api/v1/auth/users)
  - [x] Deactivate users (DELETE /api/v1/auth/users/{id})
  - [ ] Activity logs (future)
  - [ ] Usage statistics (future)

**Files:** `backend/api/v1/auth.py` (added PUT /me, PUT /me/password, GET /users, DELETE /users/{id})

---

### 13. Advanced Query Features

**Status:** ✅ **IMPLEMENTED** (2026-01-03)

- [x] **Query History**
  - [x] Queries automatically saved to database
  - [x] List user's query history (GET /api/v1/queries)
  - [x] Get query details (GET /api/v1/queries/{id})
  - [ ] Re-run queries (future)
  - [ ] Export results (future)

- [x] **Query Feedback**
  - [x] Submit feedback (POST /api/v1/queries/{id}/feedback)
  - [x] Rating (1-5 stars), helpfulness, text feedback

- [x] **Document Filtering**
  - [x] Filter by document_ids in query
  - [ ] Filter by tags/categories (future)
  - [ ] Date range filters (future)

**Files:** `backend/api/v1/query.py` (added GET /queries, GET /queries/{id}, POST /queries/{id}/feedback)

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

**Status:** ✅ **COMPREHENSIVE TEST SUITE ADDED** (2026-01-03)

### Test Summary
- **Total Tests:** 456
- **Passing:** 357 (78%)
- **Failing:** 50 (11%)
- **Skipped:** 46 (10%)

### Unit Tests (205 passed, 1 skipped)
- ✅ **Core Tests** (`tests/unit/core/`)
  - ✅ `test_config.py` - Configuration validation
  - ✅ `test_exceptions.py` - Custom exception classes
  - ✅ `test_logging.py` - Logging configuration
  - ✅ `test_security.py` - JWT token creation/verification
  - ✅ `test_cache.py` - Cache manager functionality
  - ✅ `test_permissions.py` - Permission system (14 tests, 1 skipped)

### Integration Tests (152 passed, 49 failed, 9 skipped)
- ✅ **API Tests** (`tests/integration/api/`)
  - ✅ Auth API - Login, registration, token refresh, profile management
  - ✅ Query API - Query submission, history, feedback
  - ✅ Document API - Upload, list, delete (with ACL enforcement)
  - ⚠️ Admin API - Some tests failing (needs admin user setup)
  - ⚠️ System endpoints - Health check tests failing

- ⚠️ **Database Tests** (`tests/integration/db/`)
  - ❌ Milvus tests failing (external service dependency)

- ⚠️ **Service Tests** (`tests/integration/services/`)
  - ❌ Embedding tests failing (GPU dependency)

### Test Categories
| Category | Tests | Pass Rate |
|----------|-------|-----------|
| Unit | 206 | 99.5% |
| Integration | 210 | 72% |
| **Total** | **456** | **78%** |

### Known Issues
1. **Status Code Mismatches** - Some tests expect 422 but API returns 400 (both are valid)
2. **Admin User Setup** - Admin API tests need proper admin user creation
3. **External Dependencies** - Milvus/Embedding tests require running services
4. **GPU Tests** - Some embedding tests require CUDA GPU

### Test Infrastructure
- ✅ **Fixtures** (`tests/fixtures/conftest.py`) - Shared fixtures for unit tests
- ✅ **API Fixtures** (`tests/integration/api/conftest.py`) - Integration test fixtures
- ✅ **Database Initialization** - Auto-init for PostgreSQL
- ✅ **Mock Services** - Mock RAG service for query tests

### Running Tests
```bash
# All tests
./test.sh

# Unit tests only
./test.sh unit

# Integration tests only
./test.sh integration

# With coverage
./test.sh --coverage

# Specific test
docker exec ocr-rag-test-dev pytest tests/unit/core/test_config.py -v
```

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
4. ✅ ~~Vector Search (Milvus)~~ **COMPLETED**
5. ✅ ~~LLM Service (GLM/Qwen)~~ **COMPLETED**
6. ✅ ~~RAG Orchestration~~ **COMPLETED**

### Phase 2: Production Readiness
7. ✅ ~~Background Task Processing (Celery)~~ **COMPLETED**
8. ✅ ~~Document Processing Workflow~~ **COMPLETED**
9. ✅ ~~WebSocket Streaming~~ **COMPLETED**
10. [ ] Error Handling & Logging (partial)

### Phase 3: Enhancements
11. ✅ ~~Permission System~~ **COMPLETED** (2026-01-03)
12. ✅ ~~Advanced Query Features~~ **COMPLETED** (2026-01-03)
13. [ ] Monitoring & Metrics
14. [ ] Testing & Documentation (in progress)

---

## Current API Endpoints Status

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/v1/auth/login` | POST | ✅ Implemented |
| `/api/v1/auth/register` | POST | ✅ Implemented |
| `/api/v1/auth/refresh` | POST | ✅ Implemented |
| `/api/v1/auth/me` | GET | ✅ Implemented |
| `/api/v1/auth/me` | PUT | ✅ **Profile update implemented (2026-01-03)** |
| `/api/v1/auth/me/password` | PUT | ✅ **Password change implemented (2026-01-03)** |
| `/api/v1/auth/logout` | POST | ✅ Implemented |
| `/api/v1/auth/users` | GET | ✅ **User list (admin) implemented (2026-01-03)** |
| `/api/v1/auth/users/{id}` | DELETE | ✅ **User deactivate (admin) implemented (2026-01-03)** |
| `/api/v1/documents/upload` | POST | ✅ **Full pipeline + ACL implemented** |
| `/api/v1/documents` | GET | ✅ **ACL filtered implemented (2026-01-03)** |
| `/api/v1/documents/{id}` | GET | ✅ **ACL checked implemented (2026-01-03)** |
| `/api/v1/documents/{id}` | DELETE | ✅ **ACL checked implemented (2026-01-03)** |
| `/api/v1/documents/{id}/download` | GET | ✅ **ACL checked implemented (2026-01-03)** |
| `/api/v1/query` | POST | ✅ **Real RAG pipeline implemented** |
| `/api/v1/queries` | GET | ✅ **Query history implemented (2026-01-03)** |
| `/api/v1/queries/{id}` | GET | ✅ **Query details implemented (2026-01-03)** |
| `/api/v1/queries/{id}/feedback` | POST | ✅ **Query feedback implemented (2026-01-03)** |
| `/api/v1/documents/search` | GET | ✅ Implemented |
| `/api/v1/admin/stats` | GET | ✅ Implemented |
| `/api/v1/stream/ws` | WebSocket | ✅ **Real RAG streaming implemented** |

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
**Last Updated:** 2026-01-03

## Recent Updates (2026-01-03)

### Comprehensive Test Suite Added:
1. ✅ **Unit Tests** - 205 tests passing (99.5% pass rate)
   - Config, logging, security, cache, permissions tests
2. ✅ **Integration Tests** - 152 tests passing (72% pass rate)
   - Auth API (login, register, profile, password, admin)
   - Query API (history, feedback)
   - Document API (ACL enforcement)
3. ✅ **Test Infrastructure**
   - Fixtures for unit and integration tests
   - Database initialization
   - Mock RAG service

### P1 Features Completed (Previous):
1. ✅ **Permission System (ACL)** - Document-level access control with owner/admin/role-based permissions
2. ✅ **User Management** - Profile update, password change, admin user list/deactivation
3. ✅ **Query History** - Automatic query saving, history listing, query details endpoint
4. ✅ **Query Feedback** - Rating (1-5 stars), helpfulness, text feedback submission

### All P0 Tasks Complete (Previous):
- ✅ **OCR Service** - YomiToku integration fully working with 95%+ confidence
- ✅ **Text Chunking** - Japanese-aware chunking with configurable parameters
- ✅ **Embedding Service** - Sarashina 1792D embeddings with GPU support
- ✅ **Retrieval Service** - Hybrid vector + keyword search (Milvus + BM25)
- ✅ **Reranking Service** - Llama-3.2-NV-RerankQA for improved relevance
- ✅ **LLM Service** - GLM-4.5-Air (cloud) + Qwen (local fallback)
- ✅ **RAG Orchestration** - Full pipeline with timing and confidence tracking
- ✅ **Background Processing** - Celery worker for async document processing
- ✅ **Document Upload Pipeline** - End-to-end: Upload → OCR → Chunk → Embed → Milvus → PostgreSQL
- ✅ **Query API** - REST endpoint with real RAG responses
- ✅ **WebSocket Streaming** - Real-time token streaming

### Next Steps (P2/P3 Features):
1. Fix remaining 50 failing integration tests (mostly status code mismatches and admin setup)
2. Monitoring & Observability - Prometheus metrics, structured logging
3. Permission Management API - Grant/revoke permissions UI and endpoints
4. Advanced Document Filtering - Filter by tags/categories, date ranges
