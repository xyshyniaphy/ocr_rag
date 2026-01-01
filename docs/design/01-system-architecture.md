# System Architecture Design

**Version:** 1.0
**Date:** 2026-01-01
**Status:** Design Specification

---

## 1. Architecture Overview

### 1.1 System Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL INTERFACES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   Browser    │  │  Mobile App  │  │  External    │                 │
│  │   (Web UI)   │  │   (Future)   │  │    Systems   │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│         │                 │                 │                          │
└─────────┼─────────────────┼─────────────────┼──────────────────────────┘
          │                 │                 │
          │    HTTPS/WSS    │                 │
          └─────────────────┴─────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY LAYER                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  NGINX / HAProxy                                                  │  │
│  │  - TLS Termination                                                │  │
│  │  - Rate Limiting                                                  │  │
│  │  - Request Routing                                                │  │
│  │  - WebSocket Upgrade                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                │
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────────────────────┐ │
│  │   FastAPI Backend   │  │        Streamlit Admin UI               │ │
│  │                     │  │                                         │ │
│  │  - REST API         │  │  - Document Upload Management           │ │
│  │  - WebSocket Server │  │  - System Monitoring Dashboard         │ │
│  │  - Auth/JWT         │  │  - User/Role Management                │ │
│  │  - Request Queue    │  │  - Configuration Editor                 │ │
│  └──────────┬──────────┘  └─────────────────────────────────────────┘ │
│             │                                                              │
│  ┌──────────┴──────────────────────────────────────────────────────┐     │
│  │              LangChain RAG Orchestration Layer                  │     │
│  │                                                                   │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │     │
│  │  │   Document   │  │    Query     │  │    Prompt    │          │     │
│  │  │   Processor  │  │    Router    │  │   Templates  │          │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                    SERVICE LAYER (Microservices)                        │
│                                                                         │
│  ┌───────────────────────┐  ┌─────────────────────────────────────┐   │
│  │    OCR Service        │  │       Embedding Service             │   │
│  │                        │  │                                     │   │
│  │  - YomiToku Engine     │  │  - Sarashina Model (1.2B params)   │   │
│  │  - PaddleOCR-VL Backup │  │  - Batch Processing (64 chunks)    │   │
│  │  - Image Preprocessing │  │  - GPU Memory Management           │   │
│  │  - Confidence Scoring  │  │  - ONNX Runtime Optimization      │   │
│  └───────────────────────┘  └─────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────┐  ┌─────────────────────────────────────┐   │
│  │    Reranker Service   │  │         LLM Service                 │   │
│  │                        │  │                                     │   │
│  │  - Llama-3.2-NV        │  │  - Qwen2.5-14B (Ollama)            │   │
│  │  - Top-20→Top-5        │  │  - Streaming Responses             │   │
│  │  - Cross-Encoder       │  │  - Context Management (32K tokens) │   │
│  │  - NVIDIA NIM API      │  │  - Stop Sequence Handling          │   │
│  └───────────────────────┘  └─────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA ACCESS LAYER                                  │
│                                                                         │
│  ┌───────────────────────┐  ┌─────────────────────────────────────┐   │
│  │   Vector Repository   │  │      Metadata Repository            │   │
│  │                        │  │                                     │   │
│  │  - Milvus Client       │  │  - PostgreSQL Client               │   │
│  │  - Collection Manager  │  │  - ORM (SQLAlchemy or similar)     │   │
│  │  - Search Manager      │  │  - Transaction Management          │   │
│  │  - Index Manager       │  │  - Connection Pool                 │   │
│  └───────────────────────┘  └─────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Object Storage Repository                      │   │
│  │                                                                   │   │
│  │  - MinIO Client                                                   │   │
│  │  - Bucket Management (raw-pdfs, ocr-outputs, thumbnails)         │   │
│  │  - Presigned URL Generator                                        │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                               │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Milvus    │  │  PostgreSQL  │  │    MinIO     │              │
│  │  2.4+ Stand  │  │    16+       │  │    Latest    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Redis     │  │   Ollama     │  │   NVIDIA     │              │
│  │   (Cache)    │  │   LLM Server │  │    CUDA      │              │
│  │              │  │              │  │   Runtime    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer Responsibilities

| Layer | Responsibility | Key Technologies |
|-------|---------------|------------------|
| **API Gateway** | External interface, security, routing | NGINX/HAProxy |
| **Application** | Business logic, orchestration, UI | FastAPI, Streamlit, LangChain |
| **Service** | Specialized AI/ML processing | YomiToku, PaddleOCR, Sarashina, Qwen |
| **Data Access** | Database operations abstraction | Milvus Client, PostgreSQL ORM, MinIO Client |
| **Infrastructure** | Data persistence, caching, compute | Milvus, PostgreSQL, MinIO, Redis, Ollama |

---

## 2. Component Specifications

### 2.1 API Gateway Component

**Purpose:** Single entry point for all external requests

**Responsibilities:**
- TLS 1.3 termination
- Request rate limiting (100 req/min per user, configurable)
- WebSocket upgrade handling
- Request routing to backend services
- Static asset serving (web UI)
- Health check endpoints

**Configuration:**
```nginx
# nginx.conf
upstream fastapi_backend {
    least_conn;
    server rag-app:8000 max_fails=3 fail_timeout=30s;
}

upstream streamlit_admin {
    server rag-app:8501;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.3;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;

    # REST API
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /v1/stream {
        proxy_pass http://fastapi_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Admin UI
    location /admin/ {
        proxy_pass http://streamlit_admin;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2.2 Authentication & Authorization Component

**Purpose:** Identity and access management

**Authentication Flow:**
```
Client → POST /api/v1/auth/login
       ← JWT Access Token (15 min expiry)
       ← JWT Refresh Token (7 day expiry)

Client → Request with Authorization: Bearer <token>
       → API Gateway validates JWT
       → Pass user context to backend
```

**RBAC Roles:**

| Role | Permissions |
|------|-------------|
| **Admin** | Full access: upload, delete, configure, user management |
| **Power User** | Upload, query, view all documents, manage own documents |
| **User** | Query only, view own uploaded documents |
| **Viewer** | Read-only query access, no upload |

**JWT Payload Structure:**
```json
{
  "sub": "user-uuid",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "power_user",
  "permissions": ["upload:document", "query:all", "view:own"],
  "iat": 1704096000,
  "exp": 1704096900,
  "jti": "unique-token-id"
}
```

### 2.3 Document Processing Pipeline Component

**Purpose:** Orchestrate PDF ingestion from upload to indexed embeddings

**Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  1. UPLOAD STAGE
     ┌─────────────────────────────────────────────────────────────────┐
     │  - File validation (PDF format, <50MB)                          │
     │  - Virus scan (optional, ClamAV)                                │
     │  - Duplicate detection (SHA256 hash check)                      │
     │  - Store in MinIO (raw-pdfs bucket)                             │
     │  - Create metadata record in PostgreSQL                         │
     │  - Return document_id with status="pending"                     │
     └─────────────────────────────────────────────────────────────────┘

  2. OCR STAGE (Async Task Queue)
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Retrieve PDF from MinIO                                      │
     │  - Page-by-page image extraction (300 DPI)                      │
     │  - Preprocessing: deskew, denoise, contrast enhance             │
     │  - YomiToku OCR (primary)                                       │
     │    ├─ Success confidence >=0.85: continue                        │
     │    └─ Confidence <0.85: retry with higher resolution            │
     │  - PaddleOCR-VL fallback (if YomiToku fails)                    │
     │  - Post-processing: normalize Japanese text, validate tables    │
     │  - Output: Markdown + JSON (with confidence scores)             │
     │  - Store in MinIO (ocr-outputs bucket)                          │
     └─────────────────────────────────────────────────────────────────┘

  3. CHUNKING STAGE
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Parse Markdown output                                       │
     │  - Detect table blocks (keep intact, max 2048 chars)           │
     │  - RecursiveCharacterTextSplitter with Japanese separators     │
     │  - Add metadata: page_number, chunk_index, section_header      │
     │  - Insert chunks into PostgreSQL (chunks table)                │
     └─────────────────────────────────────────────────────────────────┘

  4. EMBEDDING STAGE
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Batch chunks (64 per batch)                                 │
     │  - Sarashina-Embedding-v1-1B inference (GPU)                   │
     │  - L2-normalize vectors                                        │
     │  - Insert into Milvus (batch insert)                           │
     │  - Update PostgreSQL with embedding_id                         │
     │  - Update document status: "processing" → "completed"          │
     └─────────────────────────────────────────────────────────────────┘

  5. THUMBNAIL STAGE (Parallel)
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Extract first page as PNG                                   │
     │  - Generate thumbnail (200x300px)                              │
     │  - Store in MinIO (thumbnails bucket)                          │
     │  - Update PostgreSQL with thumbnail path                       │
     └─────────────────────────────────────────────────────────────────┘
```

**Error Handling:**
- Each stage has retry policy (3 attempts with exponential backoff)
- Failed stages update document status: "failed" with error details
- Dead letter queue for manual review
- Circuit breaker for external service failures

### 2.4 Query Processing Pipeline Component

**Purpose:** Process user queries with retrieval-augmented generation

**Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUERY PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  1. QUERY UNDERSTANDING
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Receive user query (Japanese text)                          │
     │  - Validate query length (1-500 characters)                     │
     │  - Detect query intent (keyword search vs semantic search)     │
     │  - Extract filters (document_ids, date range, tags)            │
     │  - Normalize query text (NFKC, half-width → full-width)        │
     └─────────────────────────────────────────────────────────────────┘

  2. HYBRID SEARCH
     ┌─────────────────────────────────────────────────────────────────┐
     │  Vector Search (70% weight):                                    │
     │  - Generate query embedding (Sarashina, L2-normalized)          │
     │  - Milvus IVF_FLAT search (nprobe=128, top_k=20)               │
     │  - Return: chunk_id + score                                     │
     │                                                                   │
     │  Keyword Search (30% weight):                                   │
     │  - PostgreSQL full-text search (Japanese mecab tokenizer)      │
     │  - BM25 ranking (top_k=20)                                      │
     │  - Return: chunk_id + score                                     │
     │                                                                   │
     │  Score Fusion:                                                   │
     │  - Combine scores: 0.7*vector_score + 0.3*keyword_score         │
     │  - Deduplicate chunks                                           │
     │  - Return top 20 candidates                                     │
     └─────────────────────────────────────────────────────────────────┘

  3. RERANKING
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Input: query + top 20 candidates                             │
     │  - Llama-3.2-NV-RerankQA-1B-v2 inference                        │
     │  - Cross-encoder scoring (query-passage relevance)             │
     │  - Filter by threshold (score >= 0.65)                         │
     │  - Return top 5 chunks with rerank scores                      │
     └─────────────────────────────────────────────────────────────────┘

  4. CONTEXT ASSEMBLY
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Retrieve full chunk text from PostgreSQL                     │
     │  - Sort by relevance score                                      │
     │  - Add source metadata (document_id, page_number, title)       │
     │  - Format as context:                                           │
     │    [文書1: 決算報告書, 第12頁]                                   │
     │    営業利益は150億円となり、前年比20%増加しました。               │
     │    [文書2: 四半期報告, 第5頁]                                    │
     │    主要な要因は...                                               │
     │  - Estimate context tokens (ensure <24K for 8K response)       │
     └─────────────────────────────────────────────────────────────────┘

  5. LLM GENERATION
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Construct prompt with:                                       │
     │    - System message (Japanese AI assistant persona)             │
     │    - Context (from stage 4)                                     │
     │    - User query                                                 │
     │    - Constraints (cite sources, no hallucination)               │
     │  - Call Qwen2.5-14B via Ollama API                              │
     │  - Stream response tokens                                       │
     │  - Monitor for stop sequences: [END], 参考文献:, Sources:       │
     │  - Extract cited sources from response                         │
     └─────────────────────────────────────────────────────────────────┘

  6. RESPONSE FORMATTING
     ┌─────────────────────────────────────────────────────────────────┐
     │  - Assemble final JSON:                                         │
     │    {                                                            │
     │      "query_id": "uuid",                                        │
     │      "answer": "日本語の回答テキスト...",                         │
     │      "sources": [                                               │
     │        {"document_id": "xxx", "page": 12, "snippet": "..."}    │
     │      ],                                                         │
     │      "confidence": 0.89,                                        │
     │      "processing_time_ms": 1850                                 │
     │    }                                                            │
     │  - Log query to PostgreSQL (queries table)                     │
     │  - Return to client (HTTP 200 or stream)                       │
     └─────────────────────────────────────────────────────────────────┘
```

### 2.5 Caching Strategy Component

**Purpose:** Reduce redundant computation and improve latency

**Cache Hierarchy:**

| Cache Layer | Technology | TTL | Purpose |
|-------------|------------|-----|---------|
| **L1 - Query Cache** | Redis | 1 hour | Cache query+answer pairs |
| **L2 - Embedding Cache** | Redis | 30 days | Cache document embeddings |
| **L3 - Vector Cache** | Milvus | Persistent | Vector database built-in cache |
| **L4 - OCR Cache** | MinIO | 90 days | OCR output Markdown files |

**Cache Invalidation:**
- Query cache: invalidated on document upload/delete
- Embedding cache: invalidated on document re-processing
- Vector cache: automatic on Milvus insert/delete
- OCR cache: manual re-process trigger

---

## 3. Data Flow Diagrams

### 3.1 Document Upload Flow

```
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌──────────┐
│ Browser │────▶│ NGINX   │────▶│ FastAPI     │────▶│ MinIO    │
│ Client  │     │ Gateway │     │ Upload      │     │ Storage  │
└─────────┘     └─────────┘     └─────────────┘     └──────────┘
                      │                  │
                      │                  ▼
                      │           ┌─────────────┐
                      │           │ PostgreSQL  │
                      │           │ Metadata    │
                      │           └─────────────┘
                      │
                      ▼
                ┌─────────┐
                │ Return  │
                │ doc_id  │
                └─────────┘

Async Background Task (Celery/Redis Queue):

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ OCR Service │────▶│ Chunking    │────▶│ Embedding   │
│ (YomiToku)  │     │ Service     │     │ Service     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ MinIO       │     │ PostgreSQL  │     │ Milvus      │
│ OCR outputs │     │ chunks tbl  │     │ vectors     │
└─────────────┘     └─────────────┘     └─────────────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │ Update      │
                                      │ doc status  │
                                      │ completed   │
                                      └─────────────┘
```

### 3.2 Query Processing Flow

```
┌─────────┐     ┌─────────┐     ┌─────────────┐
│ Browser │────▶│ NGINX   │────▶│ FastAPI     │
│ Client  │     │ Gateway │     │ Query       │
└─────────┘     └─────────┘     │ Handler     │
                                └─────────────┘
                                       │
                    Check Redis Cache ─┼─ Hit: Return cached
                                       │ Miss
                                       ▼
                                ┌─────────────┐
                                │ Query       │
                                │ Embedding   │
                                │ (Sarashina) │
                                └─────────────┘
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │ Milvus   │  │ Postgres │  │ Reranker │
                   │ Vector   │  │ BM25     │  │ Service  │
                   │ Search   │  │ Search   │  │          │
                   └──────────┘  └──────────┘  └──────────┘
                        │              │              │
                        └──────────────┼──────────────┘
                                       ▼
                                ┌─────────────┐
                                │ Context     │
                                │ Assembly    │
                                └─────────────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │ LLM         │
                                │ (Qwen2.5)   │
                                └─────────────┘
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │ Stream   │  │ Store    │  │ Cache    │
                   │ Response │  │ Query    │  │ Result   │
                   └──────────┘  └──────────┘  └──────────┘
```

---

## 4. Scalability Design

### 4.1 Horizontal Scaling Strategy

**Stateless Components** (can scale horizontally):
- API Gateway (multiple NGINX instances behind load balancer)
- FastAPI Backend (multiple instances with shared session store)
- OCR Service (task queue workers)
- Embedding Service (task queue workers)
- Reranker Service (task queue workers)

**Stateful Components** (require clustering/sharding):
- Milvus (distributed cluster with 3+ nodes)
- PostgreSQL (primary-replica replication)
- MinIO (distributed mode with erasure coding)
- Redis (Cluster mode with sharding)

### 4.2 Vertical Scaling Strategy

**GPU Allocation:**
```
Single GPU (RTX 4090 24GB):
├── OCR Service: 40% VRAM (~9.6GB) - YomiToku
├── Embedding Service: 30% VRAM (~7.2GB) - Sarashina
├── Reranker Service: 10% VRAM (~2.4GB) - Llama-3.2-NV
└── LLM Service: 20% VRAM (~4.8GB) - Qwen2.5-14B (Q4_K_M)
```

**Multi-GPU Configuration:**
- GPU 0: OCR (YomiToku) + Embedding (Sarashina)
- GPU 1: LLM (Qwen2.5) + Reranker

### 4.3 Load Balancing

**Algorithm:** Least Connections (for long-running OCR tasks)

**Health Checks:**
- HTTP GET /health every 5 seconds
- Mark unhealthy after 3 consecutive failures
- Drain connections before removing from pool

---

## 5. Security Architecture

### 5.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Unauthorized access | JWT auth, RBAC, rate limiting |
| Data breach | AES-256 encryption at rest, TLS 1.3 in transit |
| SQL injection | Parameterized queries, ORM usage |
| XSS | Input sanitization, Content Security Policy |
| CSRF | SameSite cookies, CSRF tokens |
| DoS | Rate limiting, request throttling, circuit breakers |
| PII leakage | PII detection, redaction option, audit logging |

### 5.2 Security Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SECURITY LAYERS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Layer 1: Network Security                                              │
│  - TLS 1.3 (HTTPS/WSS)                                                  │
│  - DDoS protection (Cloudflare or on-prem)                              │
│  - Network segmentation (DMZ for API gateway)                           │
│                                                                         │
│  Layer 2: Authentication & Authorization                                │
│  - JWT tokens (short-lived access, long-lived refresh)                  │
│  - RBAC (Admin/Power User/User/Viewer)                                  │
│  - Document-level ACLs                                                  │
│                                                                         │
│  Layer 3: Application Security                                          │
│  - Input validation (file type, size, content)                         │
│  - Output encoding (prevent XSS)                                       │
│  - SQL injection prevention (parameterized queries)                    │
│  - CSRF protection (SameSite cookies)                                  │
│                                                                         │
│  Layer 4: Data Security                                                 │
│  - AES-256-GCM encryption at rest                                       │
│  - Encrypted backups                                                   │
│  - Key management (HashiCorp Vault or AWS KMS)                         │
│  - PII detection & redaction                                           │
│                                                                         │
│  Layer 5: Audit & Monitoring                                            │
│  - Comprehensive audit logging                                         │
│  - Real-time alerting (suspicious activity)                            │
│  - Security event correlation (SIEM)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Selection Rationale

| Component | Technology | Justification |
|-----------|------------|---------------|
| **API Gateway** | NGINX | Battle-tested, high performance, excellent WebSocket support |
| **Backend** | FastAPI | Modern async Python, automatic OpenAPI docs, type safety |
| **Admin UI** | Streamlit | Rapid development, Python-native, good for internal tools |
| **Orchestration** | LangChain | RAG standard, rich integrations, active community |
| **OCR** | YomiToku | Japanese-specialized, >95% accuracy on tables, vertical text support |
| **Embedding** | Sarashina-1B | SOTA Japanese embeddings (JMTEB), decoder-based (better quality) |
| **Reranker** | Llama-3.2-NV | NVIDIA NIM integration, optimized cross-encoder |
| **LLM** | Qwen2.5-14B | Strong Japanese performance, efficient quantization (Q4_K_M) |
| **Vector DB** | Milvus 2.4 | Open-source, GPU support, hybrid search, proven scalability |
| **Metadata DB** | PostgreSQL 16 | ACID compliance, JSONB support, mature ecosystem |
| **Object Storage** | MinIO | S3-compatible, self-hosted, high performance |
| **Cache** | Redis | In-memory, versatile data structures, clustering support |
| **Task Queue** | Celery + Redis | Proven reliability, monitoring tools, retry mechanisms |
| **Container** | Docker Compose | Simple deployment, GPU pass-through support |
| **Orchestration** | Kubernetes (optional) | Multi-node scaling, rolling updates, self-healing |

---

## 7. Deployment Architecture

### 7.1 Single-Server Deployment (Development/Small Production)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Single Server (Docker)                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       Docker Network                             │  │
│  │                                                                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────────┐   │  │
│  │  │ NGINX   │  │ Redis   │  │      RAG App Container         │   │  │
│  │  │ :443    │  │ :6379   │  │                                 │   │  │
│  │  └────┬────┘  └────┬────┘  │  ┌─────────────────────────┐     │   │  │
│  │       │            │        │  │   FastAPI :8000        │     │   │  │
│  │       │            │        │  │   Streamlit :8501      │     │   │  │
│  │       └────────────┼────────┘  │   Celery Worker        │     │   │  │
│  │                    │           │   - OCR (YomiToku)     │     │   │  │
│  │                    │           │   - Embedding (Sarashi) │     │   │  │
│  │                    │           │   - Reranker            │     │   │  │
│  │                    │           └─────────────────────────┘     │   │  │
│  │                    │                                         │   │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │  │
│  │  │              GPU Access (nvidia-docker)                  │ │   │  │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │ │   │  │
│  │  │  │ Milvus  │  │ Postgres│  │  Ollama │  │  MinIO  │     │ │   │  │
│  │  │  │ :19530  │  │ :5432   │  │ :11434  │  │ :9000   │     │ │   │  │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │ │   │  │
│  │  └──────────────────────────────────────────────────────────┘ │   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  GPU: RTX 4090 (24GB VRAM) - shared via nvidia-docker runtime           │
│  RAM: 64GB                                                              │
│  Storage: 2TB NVMe SSD                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Multi-Node Deployment (Enterprise)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Load Balancer                                 │
│                        (NGINX/HAProxy)                                  │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────┴────────┐          ┌───────────┴──────────┐          ┌────────┴─────────┐
│  Node 1: API   │          │  Node 2: Workers    │          │  Node 3: LLM     │
│  + Gateway     │          │  (OCR + Embedding)  │          │  + Reranker      │
│                │          │                     │          │                  │
│  NGINX ×2      │          │  Celery Workers ×4  │          │  Ollama          │
│  FastAPI ×2    │          │  - GPU: 2×RTX4090   │          │  - GPU: A100     │
│  Streamlit ×1  │          │  - YomiToku         │          │  - Qwen2.5-14B   │
│                │          │  - Sarashina        │          │  - Llama-3.2-NV  │
└────────────────┘          └─────────────────────┘          └──────────────────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────┴────────┐          ┌───────────┴──────────┐          ┌────────┴─────────┐
│  Node 4: Data  │          │  Node 5: Data       │          │  Node 6: Data    │
│  Milvus #1     │          │  Milvus #2          │          │  Milvus #3       │
│  (Coordinator) │          │  (Worker)           │          │  (Worker)        │
│                │          │                     │          │                  │
│  PostgreSQL    │          │  MinIO              │          │  Redis Cluster   │
│  (Primary)     │          │  (Distributed)      │          │  (Shard 3)       │
│                │          │                     │          │                  │
└────────────────┘          └─────────────────────┘          └──────────────────┘
```

---

## 8. Monitoring & Observability

### 8.1 Metrics Collection

**Application Metrics** (Prometheus):
```python
# Key metrics to track
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter('http_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'Request latency')

# RAG-specific metrics
query_count = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'RAG query latency', ['stage'])
ocr_duration = Histogram('ocr_duration_seconds', 'OCR processing time', ['engine'])
embedding_duration = Histogram('embedding_duration_seconds', 'Embedding generation time')

# Resource metrics
gpu_utilization = Gauge('nvidia_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('nvidia_gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
vector_db_size = Gauge('milvus_collection_size', 'Vector DB collection size', ['collection'])
```

### 8.2 Distributed Tracing

**OpenTelemetry Integration:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(agent_host_name="jaeger", agent_port=6831)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))

tracer = trace.get_tracer(__name__)

# Example usage
with tracer.start_as_current_span("process_query"):
    with tracer.start_as_current_span("vector_search"):
        results = milvus.search(...)
    with tracer.start_as_current_span("llm_generation"):
        response = ollama.generate(...)
```

### 8.3 Alerting Rules

**Critical Alerts** (PagerDuty/SMS):
- GPU memory >95% for 5 minutes
- Milvus down (all nodes)
- PostgreSQL primary down
- Error rate >10% for 5 minutes
- Query latency >10s (95th percentile)

**Warning Alerts** (Email/Slack):
- GPU memory >80% for 15 minutes
- Disk space <20%
- Queue depth >100 tasks
- Cache hit rate <50%

---

## 9. Disaster Recovery & Business Continuity

### 9.1 Backup Strategy

| Data | Backup Frequency | Retention | Storage Location |
|------|------------------|-----------|------------------|
| PostgreSQL | Daily full + continuous WAL | 30 days daily, 1 year monthly | Off-site / Cloud |
| Milvus | Weekly full + daily incremental | 30 days | Off-site / Cloud |
| MinIO | Continuous versioning | 30 versions | Off-site replication |
| Models (Git LFS) | Per commit | Forever | GitHub / GitLab |

### 9.2 Recovery Objectives

- **RTO** (Recovery Time Objective): 4 hours
- **RPO** (Recovery Point Objective): 1 hour

### 9.3 Failover Procedure

1. **Database Failover** (PostgreSQL):
   - Promote replica to primary
   - Update connection strings
   - Rebuild old primary

2. **Vector DB Failover** (Milvus):
   - Remaining nodes continue serving (quorum-based)
   - Replace failed node
   - Rebalance data

3. **Application Failover**:
   - Load balancer redirects traffic to healthy nodes
   - Auto-scaling replaces failed instances
   - Task queue re-distributes jobs

---

**END OF SYSTEM ARCHITECTURE DESIGN**
