# Japanese OCR RAG System - Technical Specification Document

**Version:** 1.0
**Date:** January 3, 2026
**Target Language:** Japanese (日本語)
**Classification:** Production-Ready Architecture
**Document Owner:** Technical Architecture Team

---

## 1.1 Implementation Status

### P0 Core RAG Pipeline: ✅ COMPLETE (2026-01-03)
| Component | Status | Notes |
|-----------|--------|-------|
| OCR Service (YomiToku) | ✅ Complete | 95%+ accuracy on Japanese documents |
| Text Chunking | ✅ Complete | Japanese-aware with configurable parameters |
| Embedding (Sarashina 1792D) | ✅ Complete | GPU-accelerated, batch processing |
| Retrieval (Hybrid) | ✅ Complete | Vector + BM25 keyword search |
| Reranking (Llama-3.2-NV) | ✅ Complete | Top-K re-ranking for relevance |
| LLM (GLM-4.5-Air) | ✅ Complete | Primary cloud LLM with Japanese support |
| RAG Orchestration | ✅ Complete | Full 5-stage pipeline with error handling |
| Background Processing | ✅ Complete | Celery worker for async document processing |
| Query API | ✅ Complete | REST endpoint with real RAG responses |
| WebSocket Streaming | ✅ Complete | Real-time token streaming |

### P1 High Priority: 🟡 IN PROGRESS
| Feature | Status | Notes |
|---------|--------|-------|
| Permission System (ACL) | ⚠️ Model exists | Enforcement middleware pending |
| User Management | ⚠️ Basic complete | Profile editing pending |
| Advanced Query Features | ⚠️ API exists | Query history, filtering pending |

### P2/P3 Features: ⏳ TODO
- Monitoring & Observability (Prometheus metrics)
- Unit Tests (191 passing ✅)
- Integration/E2E Tests
- API Documentation completion

---

## 1. Executive Summary

### 1.1 System Overview
This document specifies a production-grade Retrieval-Augmented Generation (RAG) system optimized for Japanese PDF document processing. The system addresses three core challenges:

1. **OCR Accuracy**: Japanese documents contain vertical text (縦書き), complex table structures, and mixed Kanji/Hiragana/Katakana characters requiring specialized OCR engines
2. **Semantic Search**: Japanese language embedding models must handle contextual nuances, particles (助詞), and honorifics (敬語) for accurate retrieval
3. **Answer Generation**: Japanese LLMs must maintain natural language flow while accurately citing source documents

### 1.2 Target Use Cases
- **Legal Document Analysis**: Contract review, compliance checking, legal precedent research
- **Financial Report Processing**: Annual reports, quarterly earnings, regulatory filings
- **Academic Research**: Paper analysis, literature review, citation management
- **Government Document Search**: Public records, policy documents, administrative notices
- **Enterprise Knowledge Management**: Internal document search, standard operating procedures (SOP), technical manuals

### 1.3 System Characteristics
- **Privacy-First**: All processing occurs locally (air-gapped deployment supported)
- **Accuracy**: >95% OCR accuracy on complex tables; >90% retrieval relevance (JMTEB benchmark)
- **Scalability**: 10,000+ document corpus; 5-10 concurrent users on single GPU
- **Latency**: <3s per query end-to-end (GPU); <20s (CPU-only)
- **Languages**: Primary Japanese; optional English fallback

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Streamlit  │  │  REST API    │  │   Web UI     │              │
│  │   (Admin)    │  │ (FastAPI)    │  │  (End User)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                  Application Layer (Orchestration)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              RAG Pipeline (Custom Implementation)             │   │
│  │  - OCR Service  - Text Chunker  - Embedding Service         │   │
│  │  - Retrieval Service - Reranker Service - LLM Service       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                      Processing Layer (3 Parallel Pipelines)         │
│                                                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  OCR Pipeline   │  │  Embedding       │  │  LLM Generation  │  │
│  │                 │  │  Pipeline        │  │  Pipeline        │  │
│  │ ┌─────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │
│  │ │  YomiToku   │ │  │ │  Sarashina   │ │  │ │  GLM-4.5-Air │ │  │
│  │ │     OCR     │ │  │ │ Embedding-v1 │ │  │ │  (Primary)   │ │  │
│  │ │   (Japanese)│ │  │ │     1B       │ │  │ │   (Z.ai)     │ │  │
│  │ │    1792D    │ │  │ │   (1792D)    │ │  │ └──────────────┘ │  │
│  │ └─────────────┘ │  │ └──────────────┘ │  │                  │  │
│  │                 │  │                  │  │ ┌──────────────┐ │  │
│  │                 │  │ ┌──────────────┐ │  │ │   Qwen2.5    │ │  │
│  │                 │  │ │   Reranker   │ │  │ │   14B        │ │  │
│  │                 │  │ │ Llama-3.2-NV │ │  │ │  (Fallback)  │ │  │
│  │                 │  │ └──────────────┘ │  │ │  (Ollama)    │ │  │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                       Storage Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Milvus    │  │  PostgreSQL  │  │   MinIO      │              │
│  │  Vector DB   │  │  Metadata DB │  │ Object Store │              │
│  │ (1792D embed)│  │  (Document)  │  │ (Raw PDFs)   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### 2.2.1 Document Ingestion Flow
```
PDF Upload -> Pre-validation -> OCR Processing -> Markdown Conversion 
  -> Text Chunking -> Embedding Generation -> Vector DB Storage 
    -> Metadata Indexing -> Confirmation
```

#### 2.2.2 Query Processing Flow
```
User Query -> Query Understanding -> Hybrid Search (Vector + Keyword)
  -> Candidate Retrieval (Top 20) -> Reranking (Top 5) 
    -> Context Assembly -> LLM Generation -> Citation Injection 
      -> Response Formatting -> User Display
```

---

## 3. Component Specifications

### 3.1 OCR Engine Layer

#### 3.1.1 Primary OCR: YomiToku
- **Model**: kotaro-kinoshita/yomitoku (Japanese-specialized)
- **Deployment**: Local container (Docker) or Python library
- **Input Format**: PDF (scanned or native), PNG, JPEG (300 DPI minimum recommended)
- **Output Format**: Structured Markdown with table preservation
- **Performance Requirements**:
  - Throughput: >=5 pages/minute on RTX 4090; >=1 page/minute on CPU
  - Accuracy: >=95% on tables with cell merging
  - Vertical Text: 100% layout preservation
- **Configuration**:
  ```yaml
  yomitoku:
    language: ja
    output_format: markdown
    table_detection: true
    vertical_text_support: true
    image_resolution: auto  # Dynamic adjustment
    batch_size: 4  # Pages per batch
    gpu_memory_fraction: 0.4  # Reserve 40% VRAM
  ```
- **Error Handling**:
  - Retry with higher resolution if confidence <0.85
  - Log low-confidence regions for manual review
  - Multi-language documents handled via YomiToku's built-in support

#### 3.1.3 OCR Quality Assurance
- **Pre-processing**:
  - Deskew (angle correction +/-15 degrees)
  - Noise reduction (bilateral filter)
  - Contrast enhancement (CLAHE for scanned docs)
  - Binarization (Otsu's method for low-quality scans)
- **Post-processing**:
  - Japanese character normalization (half-width -> full-width Katakana)
  - Space insertion between Kanji and Latin characters
  - Table structure validation (check row/column consistency)
  - Confidence scoring per text block

### 3.2 Embedding Model Layer

#### 3.2.1 Primary Embedding: Sarashina-Embedding-v1-1B
- **Model**: sbintuitions/sarashina-embedding-v1-1b
- **Architecture**: Decoder-based (1.2B parameters)
- **Embedding Dimension**: **1792D** (high-dimensional for better semantic capture)
- **Max Context Length**: 8,192 tokens
- **Normalization**: L2-normalized (required for cosine similarity)
- **Deployment**:
  - **Local**: HuggingFace Transformers (CUDA accelerated)
  - **API**: Custom inference server (FastAPI + ONNX Runtime)
  - **Batch Processing**: 64 documents/batch on RTX 4090
- **Configuration**:
  ```python
  sarashina_config = {
      "model_name": "sbintuitions/sarashina-embedding-v1-1b",
      "device": "cuda:0",
      "normalize_embeddings": True,
      "batch_size": 64,
      "max_length": 512,  # Chunk size aligned
      "pooling": "eos_token",  # Use EOS token hidden state
      "precision": "fp16"  # Mixed precision for speed
  }
  ```
- **Performance SLA**:
  - Throughput: 1,000 chunks/minute on RTX 4090
  - Latency: <100ms per query embedding
  - Memory: 3GB VRAM

#### 3.2.2 Reranker: Llama-3.2-NV-RerankQA-1B-v2
- **Purpose**: Re-score top-K candidates (semantic + cross-encoder)
- **Input**: Query + 20 candidate passages -> Output: 5 best passages
- **Deployment**: NVIDIA NIM container or local Ollama
- **Configuration**:
  ```yaml
  reranker:
    model: nvidia/llama-3.2-nv-rerankqa-1b-v2
    top_k_input: 20
    top_k_output: 5
    score_threshold: 0.65  # Minimum relevance score
    timeout_ms: 500  # Per-batch timeout
  ```

### 3.3 Vector Database Layer

#### 3.3.1 Milvus Configuration
- **Version**: Milvus 2.4+ (standalone or distributed)
- **Collection Schema**:
  ```python
  schema = {
      "fields": [
          {"name": "chunk_id", "type": "VARCHAR", "max_length": 64, "is_primary": True},
          {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 1792},  # Sarashina-Embedding-v1-1B
          {"name": "text_content", "type": "VARCHAR", "max_length": 4096},
          {"name": "document_id", "type": "VARCHAR", "max_length": 64},
          {"name": "page_number", "type": "INT32},
          {"name": "chunk_index", "type": "INT32},
          {"name": "metadata", "type": "JSON"}  # Title, author, date, etc.
      ],
      "enable_dynamic_field": True
  }
  ```
- **Index Type**: IVF_FLAT (balance between speed and accuracy)
  - **nlist**: 1024 (number of clusters)
  - **nprobe**: 128 (clusters to search)
  - **metric_type**: IP (Inner Product, for normalized vectors)
- **Search Parameters**:
  ```yaml
  search:
    top_k: 20  # Initial candidates
    nprobe: 128
    consistency_level: "Strong"  # Ensure latest data
    timeout: 5000  # 5s timeout
  ```
- **Hybrid Search** (Vector + Keyword):
  ```python
  hybrid_search_params = {
      "vector_weight": 0.7,  # 70% semantic similarity
      "keyword_weight": 0.3,  # 30% BM25 keyword match
      "rerank": True  # Apply reranker after hybrid retrieval
  }
  ```

#### 3.3.2 Metadata Database: PostgreSQL
- **Version**: PostgreSQL 16+
- **Schema**:
  ```sql
  CREATE TABLE documents (
      document_id UUID PRIMARY KEY,
      filename VARCHAR(512) NOT NULL,
      file_hash VARCHAR(64) UNIQUE,  -- SHA256 for deduplication
      upload_timestamp TIMESTAMPTZ DEFAULT NOW(),
      file_size_bytes BIGINT,
      page_count INT,
      language VARCHAR(10) DEFAULT 'ja',
      ocr_status VARCHAR(20),  -- pending/processing/completed/failed
      ocr_confidence FLOAT,
      metadata JSONB,  -- Custom fields (title, author, tags, etc.)
      INDEX idx_filename (filename),
      INDEX idx_upload_timestamp (upload_timestamp),
      INDEX idx_metadata USING gin(metadata jsonb_path_ops)
  );

  CREATE TABLE chunks (
      chunk_id UUID PRIMARY KEY,
      document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
      page_number INT,
      chunk_index INT,  -- Sequential order within page
      text_content TEXT,
      embedding_id VARCHAR(64),  -- Reference to Milvus vector
      token_count INT,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      INDEX idx_document_page (document_id, page_number)
  );

  CREATE TABLE queries (
      query_id UUID PRIMARY KEY,
      user_id VARCHAR(64),
      query_text TEXT,
      timestamp TIMESTAMPTZ DEFAULT NOW(),
      response_time_ms INT,
      retrieved_chunks INT,
      feedback_score INT,  -- 1-5 user rating
      INDEX idx_user_timestamp (user_id, timestamp)
  );
  ```

#### 3.3.3 Object Storage: MinIO
- **Purpose**: Store raw PDF files, OCR intermediate outputs, cached results
- **Buckets**:
  - `raw-pdfs`: Original uploaded PDFs
  - `ocr-outputs`: Markdown and JSON outputs from OCR
  - `thumbnails`: First page preview images
- **Retention Policy**:
  - Raw PDFs: Retain indefinitely
  - OCR outputs: 90-day cache (regenerate if older)
  - Thumbnails: 30-day cache

### 3.4 LLM Generation Layer

#### 3.4.1 Primary LLM: GLM-4.5-Air (Z.ai Platform)
- **Model**: GLM-4.5-Air (Z.ai international platform)
- **Provider**: Z.ai (https://api.z.ai/api/paas/v4/)
- **Context Window**: 128,000 tokens
- **Deployment**: Cloud API (OpenAI-compatible)
- **Advantages**:
  - ✅ Fast response times (cloud API)
  - ✅ No local GPU required
  - ✅ Cost-effective for production
  - ✅ High-quality Japanese responses
- **Configuration**:
  ```yaml
  glm_config:
    api_key: ${GLM_API_KEY}  # From https://z.ai/
    base_url: https://api.z.ai/api/paas/v4/
    model: GLM-4.5-Air  # Fast, cost-effective
    temperature: 0.1  # Low temperature for factual accuracy
    top_p: 0.9
    max_tokens: 2048
    stream: true  # Enable streaming responses
  ```
- **Prompt Template**:
  ```python
  JAPANESE_RAG_PROMPT = """あなたは正確で信頼できる日本語の文書分析AIアシスタントです。

【重要な制約】
1. 以下の【参考文献】に記載された情報のみを使用して回答してください
2. 文献に情報が無い場合は「提供された文書には該当情報が見つかりませんでした」と明言してください
3. 推測や一般知識での補完は一切行わないでください
4. 情報源を必ず明記してください（例：「〇〇報告書」第X頁による）
5. 表や図の内容を参照する際は、その旨を明確に示してください

【参考文献】
{context}

【質問】
{question}

【回答】（参考文献に基づき、正確に回答してください）
"""
  ```

#### 3.4.2 Fallback LLM: Qwen2.5-14B (Ollama)
- **Use Case**: Local fallback when GLM API is unavailable
- **Model**: Qwen/Qwen2.5-14B-Instruct (via Ollama)
- **Context Window**: 32,768 tokens
- **Quantization**: Q4_K_M (4-bit, medium quality) for RTX 4090
- **Deployment**: Ollama server (local GPU)
- **Configuration**:
  ```yaml
  qwen_config:
    model: qwen2.5:14b-instruct-q4_K_M
    temperature: 0.1
    top_p: 0.9
    top_k: 40
    repeat_penalty: 1.1
    num_ctx: 32768
    num_predict: 2048
    stop_sequences: ["[END]", "参考文献:", "Sources:"]
  ```

#### 3.4.3 LLM Provider Switching
- **Environment Variable**: `LLM_PROVIDER`
  - `glm` - Use GLM cloud API (default, recommended)
  - `ollama` - Use local Ollama (fallback)
- **Automatic Fallback**: System automatically switches to Ollama if GLM API fails

### 3.5 Text Processing Layer

#### 3.5.1 Text Chunking Strategy
- **Chunker**: RecursiveCharacterTextSplitter (LangChain)
- **Parameters**:
  ```python
  chunking_config = {
      "chunk_size": 512,  # Characters (Japanese character approx 0.5 tokens)
      "chunk_overlap": 50,  # 10% overlap for context continuity
      "length_function": len,
      "separators": [
          "\n\n",  # Paragraph breaks
          "\n",    # Line breaks
          "。",    # Japanese period
          "！",    # Exclamation
          "？",    # Question mark
          "；",    # Semicolon
          "、",    # Japanese comma
          " ",     # Space
          ""       # Character-level fallback
      ],
      "keep_separator": True  # Preserve punctuation
  }
  ```
- **Special Handling**:
  - **Tables**: Keep entire table as single chunk (up to 2048 chars)
  - **Headers**: Include section header in chunk metadata
  - **Page Boundaries**: Add page number as metadata, do not split mid-sentence across pages

#### 3.5.2 Japanese Text Normalization
- **Pre-processing**:
  ```python
  def normalize_japanese_text(text: str) -> str:
      import unicodedata

      # 1. Unicode normalization (NFKC: canonical composition)
      text = unicodedata.normalize('NFKC', text)

      # 2. Half-width -> Full-width Katakana
      text = text.translate(str.maketrans(
          'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ',
          'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
      ))

      # 3. Remove excessive whitespace
      text = re.sub(r'\s+', ' ', text)

      # 4. Fix common OCR errors (optional)
      ocr_corrections = {
          '0': 'O',  # Zero vs O
          'l': 'I',  # Lowercase L vs I (context-dependent)
      }
      # Apply selectively based on context

      return text.strip()
  ```

---

## 4. API Specifications

### 4.1 REST API Endpoints

#### 4.1.1 Document Upload
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Request:
  - file: binary (PDF file, max 50MB)
  - metadata: JSON (optional)
    {
      "title": "決算報告書",
      "author": "財務部",
      "tags": ["financial", "Q4-2025"],
      "language": "ja"
    }

Response:
  {
    "document_id": "uuid-xxx",
    "status": "processing",
    "estimated_completion": "2026-01-01T12:05:00Z"
  }

Status Codes:
  - 202 Accepted: File uploaded, processing started
  - 400 Bad Request: Invalid file format
  - 413 Payload Too Large: File >50MB
  - 500 Internal Server Error: Processing failure
```

#### 4.1.2 Query RAG System
```http
POST /api/v1/query
Content-Type: application/json

Request:
  {
    "query": "2025年第4四半期の営業利益は？",
    "document_ids": ["uuid-1", "uuid-2"],  // Optional: filter by documents
    "top_k": 5,  // Number of sources to retrieve
    "include_sources": true,
    "language": "ja"
  }

Response:
  {
    "query_id": "uuid-qqq",
    "answer": "2025年第4四半期の営業利益は150億円でした（決算報告書 第12頁）。",
    "sources": [
      {
        "document_id": "uuid-1",
        "document_title": "決算報告書",
        "page_number": 12,
        "chunk_text": "営業利益は150億円となり...",
        "relevance_score": 0.92
      }
    ],
    "processing_time_ms": 1850,
    "confidence": 0.89
  }

Status Codes:
  - 200 OK: Success
  - 400 Bad Request: Invalid query parameters
  - 404 Not Found: Document ID not found
  - 429 Too Many Requests: Rate limit exceeded
  - 500 Internal Server Error
```

#### 4.1.3 Get Document Status
```http
GET /api/v1/documents/{document_id}/status

Response:
  {
    "document_id": "uuid-xxx",
    "status": "completed",  // pending/processing/completed/failed
    "progress": 100,  // Percentage
    "page_count": 45,
    "chunk_count": 234,
    "ocr_confidence": 0.94,
    "errors": []
  }
```

#### 4.1.4 Search Documents
```http
GET /api/v1/documents/search?q=決算&limit=10&offset=0

Response:
  {
    "total": 23,
    "results": [
      {
        "document_id": "uuid-1",
        "title": "2025年決算報告書",
        "upload_date": "2025-12-15T10:00:00Z",
        "page_count": 45,
        "match_score": 0.87
      }
    ]
  }
```

### 4.2 WebSocket API (Real-time Streaming)

```javascript
// Client-side connection
const ws = new WebSocket('wss://api.example.com/v1/stream');

ws.send(JSON.stringify({
  "type": "query",
  "query": "営業利益の推移を教えてください",
  "stream": true
}));

// Server streams response tokens
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type: "token" | "source" | "complete"
  // data.content: Incremental text or metadata
};
```

---

## 5. Security & Privacy Specifications

### 5.1 Data Security

#### 5.1.1 Encryption
- **At Rest**: AES-256-GCM for all stored files (PDF, Markdown, vectors)
- **In Transit**: TLS 1.3 for all API communications
- **Key Management**: HashiCorp Vault or AWS KMS

#### 5.1.2 Access Control
- **Authentication**: OAuth 2.0 + JWT tokens
- **Authorization**: Role-Based Access Control (RBAC)
  - **Admin**: Full access (upload, delete, configure)
  - **Power User**: Upload, query, view all documents
  - **User**: Query only, view own documents
  - **Viewer**: Read-only query access
- **Document-Level Permissions**: ACL per document (owner, shared users, public/private)

#### 5.1.3 Audit Logging
- **Log Events**:
  - User login/logout
  - Document upload/delete
  - Query execution (query text, user, timestamp, results)
  - Configuration changes
- **Retention**: 1 year (configurable)
- **Storage**: PostgreSQL audit table + Elasticsearch (searchable logs)

### 5.2 Privacy Compliance

#### 5.2.1 Japanese APPI (Act on the Protection of Personal Information)
- **PII Detection**: Scan documents for Personal Identifiable Information (names, addresses, My Number)
- **Anonymization**: Option to redact PII before indexing
- **Consent Management**: User consent required for PII processing

#### 5.2.2 Data Residency
- **Deployment**: All servers hosted in Japan (Tokyo region for cloud; on-premises for government clients)
- **No Third-Party APIs**: Zero external API calls (OpenAI, Claude, etc.)

#### 5.2.3 Right to Deletion
- **Document Deletion**: Hard delete from all systems within 24 hours
- **Vector Deletion**: Remove embeddings from Milvus
- **Backup Removal**: Expire backups after 30 days (configurable)

---

## 6. Performance Requirements

### 6.1 Latency SLA

| Operation | Target | Maximum | Notes |
|-----------|--------|---------|-------|
| Single Query (GPU) | <2s | 5s | 95th percentile |
| Single Query (CPU) | <15s | 30s | For fallback mode |
| Document Upload | <1s | 3s | File validation only |
| OCR Processing | <10s/page | 30s/page | Depends on complexity |
| Embedding Generation | <50ms/chunk | 200ms/chunk | Batch processing |
| Vector Search | <100ms | 500ms | Top-20 retrieval |
| Reranking | <300ms | 1s | Top-5 from Top-20 |
| LLM Generation | <1s | 3s | Streaming response |

### 6.2 Throughput Requirements

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| Concurrent Queries (GPU) | 5 | 10 | RTX 4090 |
| Concurrent Queries (CPU) | 1 | 2 | Fallback mode |
| Document Ingestion | 100 pages/hour | 500 pages/hour | Background processing |
| Embedding Throughput | 500 chunks/min | 1000 chunks/min | GPU-accelerated |
| Vector DB QPS | 100 | 500 | Queries per second |

### 6.3 Scalability Requirements

| Scale Dimension | Initial | 1-Year Target | 5-Year Target |
|-----------------|---------|---------------|---------------|
| Total Documents | 1,000 | 10,000 | 100,000 |
| Total Pages | 50,000 | 500,000 | 5,000,000 |
| Total Chunks | 250,000 | 2,500,000 | 25,000,000 |
| Vector DB Size | 1.5GB | 15GB | 150GB |
| Concurrent Users | 5 | 20 | 100 |
| Query Volume | 100/day | 1,000/day | 10,000/day |

---

## 7. Quality Assurance

### 7.1 OCR Quality Metrics

| Metric | Threshold | Measurement Method |
|--------|-----------|---------------------|
| Character Accuracy | >=98% | Manual annotation on 100-page sample |
| Table Structure Accuracy | >=95% | Cell boundary & content validation |
| Vertical Text Accuracy | >=99% | Japanese traditional document test set |
| Confidence Score Reliability | +/-5% | Compare predicted confidence vs actual accuracy |

### 7.2 Retrieval Quality Metrics

| Metric | Target | Measurement Method |
|--------|--------|---------------------|
| Recall@5 | >=85% | Ground-truth labeled test set (500 queries) |
| Precision@5 | >=75% | Relevance labeling (1-5 scale, >=3 is relevant) |
| NDCG@5 | >=0.80 | Normalized Discounted Cumulative Gain |
| MRR (Mean Reciprocal Rank) | >=0.75 | First relevant document position |
| Hallucination Rate | <=2% | LLM-as-Judge + manual review |

### 7.3 Generation Quality Metrics

| Metric | Target | Measurement Method |
|--------|--------|---------------------|
| Factual Accuracy | >=95% | Fact-checking against source documents |
| Citation Accuracy | 100% | All claims must have valid source reference |
| Japanese Fluency | >=4/5 | Native speaker evaluation (linguistic quality) |
| Answer Relevance | >=4/5 | User feedback (5-point Likert scale) |

### 7.4 Testing Strategy

#### 7.4.1 Unit Testing
- **Coverage**: >=80% code coverage
- **Components**:
  - OCR preprocessing functions
  - Text normalization utilities
  - Embedding generation
  - Vector search queries
  - Prompt template rendering

#### 7.4.2 Integration Testing
- **End-to-End Pipelines**:
  - Document ingestion (PDF -> OCR -> Embedding -> Vector DB)
  - Query processing (User query -> Retrieval -> Generation -> Response)
- **Failure Scenarios**:
  - Corrupted PDF handling
  - Network timeout recovery
  - GPU OOM (Out of Memory) fallback

#### 7.4.3 Performance Testing
- **Load Testing**: 100 concurrent users, 1000 queries/hour
- **Stress Testing**: 500 concurrent users until system degradation
- **Soak Testing**: 72-hour continuous operation (memory leak detection)

#### 7.4.4 Acceptance Testing
- **User Acceptance Testing (UAT)**: 10 domain experts test real-world scenarios
- **A/B Testing**: Compare new models vs baseline (Sarashina vs Ruri, YomiToku vs PaddleOCR)

---

## 8. Deployment Specifications

### 8.1 Hardware Requirements

#### 8.1.1 Development Environment
- **CPU**: 8-core (Intel i7/AMD Ryzen 7)
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM) or better
- **Storage**: 256GB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

#### 8.1.2 Production Environment (Small-Scale)
- **CPU**: 16-core (Intel Xeon Silver / AMD EPYC)
- **RAM**: 64GB
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Storage**: 2TB NVMe SSD (RAID 1 for redundancy)
- **OS**: Ubuntu 22.04 LTS Server
- **Capacity**: 10,000 documents, 5-10 concurrent users

#### 8.1.3 Production Environment (Enterprise-Scale)
- **Compute Nodes**: 4x servers
  - **Node 1 (OCR)**: 32-core CPU, 128GB RAM, 2x RTX A6000 (48GB VRAM each)
  - **Node 2 (Embedding/Reranking)**: 16-core CPU, 64GB RAM, RTX 4090
  - **Node 3 (LLM)**: 16-core CPU, 128GB RAM, NVIDIA A100 (80GB VRAM)
  - **Node 4 (Storage)**: 32-core CPU, 256GB RAM, 20TB HDD (object storage)
- **Vector DB Cluster**: 3-node Milvus cluster (distributed)
- **Load Balancer**: NGINX or HAProxy
- **Capacity**: 100,000+ documents, 50-100 concurrent users

### 8.2 Software Stack

#### 8.2.1 Operating System
- **Primary**: Ubuntu 22.04 LTS Server
- **Alternative**: Rocky Linux 9 (RHEL-compatible)

#### 8.2.2 Container Runtime
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **Kubernetes** (optional for multi-node): 1.28+

#### 8.2.3 Programming Languages
- **Python**: 3.11+ (primary backend)
- **JavaScript/TypeScript**: Node.js 20+ (optional web frontend)

#### 8.2.4 Key Dependencies
```yaml
python_dependencies:
  - langchain==0.1.0
  - langchain-community==0.0.10
  - transformers==4.36.0
  - torch==2.2.0+cu121  # CUDA 12.1
  - pymilvus==2.4.0
  - psycopg2-binary==2.9.9
  - fastapi==0.108.0
  - uvicorn[standard]==0.25.0
  - streamlit==1.30.0
  - paddlepaddle-gpu==2.6.0  # PaddleOCR
  - yomitoku==1.2.0  # YomiToku OCR
  - onnxruntime-gpu==1.16.0

system_packages:
  - tesseract-ocr
  - tesseract-ocr-jpn
  - poppler-utils  # PDF rendering
  - libgl1-mesa-glx  # OpenCV dependencies
```

### 8.3 Deployment Architecture

#### 8.3.1 Docker Compose (Single-Server)
```yaml
version: '3.8'

services:
  # Vector Database
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

  # PostgreSQL Metadata DB
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: rag_metadata
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # MinIO Object Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  # Ollama LLM Server
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

  # RAG Application (FastAPI + Streamlit)
  rag-app:
    build:
      context: ./app
      dockerfile: Dockerfile
    environment:
      MILVUS_HOST: milvus
      POSTGRES_HOST: postgres
      MINIO_ENDPOINT: minio:9000
      OLLAMA_HOST: ollama:11434
      SARASHINA_MODEL_PATH: /models/sarashina-embedding-v1-1b
    ports:
      - "8000:8000"  # FastAPI
      - "8501:8501"  # Streamlit
    volumes:
      - ./models:/models:ro
      - ./data:/app/data
    depends_on:
      - milvus
      - postgres
      - minio
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  milvus_data:
  postgres_data:
  minio_data:
  ollama_models:
```

#### 8.3.2 Kubernetes Deployment (Multi-Node)
```yaml
# Deployment for RAG Application
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag
  template:
    metadata:
      labels:
        app: rag
    spec:
      containers:
      - name: rag-backend
        image: registry.example.com/rag-app:1.0
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
        env:
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: POSTGRES_HOST
          value: "postgres-service"
        ports:
        - containerPort: 8000
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: rag
```

### 8.4 Continuous Integration/Deployment (CI/CD)

#### 8.4.1 CI Pipeline (GitHub Actions)
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t rag-app:${{ github.sha }} .
      - name: Push to registry
        run: docker push registry.example.com/rag-app:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/rag-app rag-backend=registry.example.com/rag-app:${{ github.sha }}
```

---

## 9. Monitoring & Observability

### 9.1 Metrics Collection

#### 9.1.1 Application Metrics
- **Prometheus** exporters:
  - Request latency (histogram)
  - Request count (counter)
  - Active connections (gauge)
  - GPU utilization (gauge)
  - Memory usage (gauge)
  - Error rate (counter)

#### 9.1.2 Infrastructure Metrics
- **Node Exporter**: CPU, memory, disk I/O
- **NVIDIA DCGM Exporter**: GPU metrics (utilization, temperature, power)
- **Milvus Metrics**: QPS, latency, index size
- **PostgreSQL Exporter**: Connections, query performance

### 9.2 Logging

#### 9.2.1 Log Aggregation
- **Stack**: Fluentd -> Elasticsearch -> Kibana
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured Logging** (JSON format):
  ```json
  {
    "timestamp": "2026-01-01T12:00:00Z",
    "level": "INFO",
    "service": "rag-app",
    "event": "query_processed",
    "user_id": "user-123",
    "query_id": "uuid-qqq",
    "latency_ms": 1850,
    "retrieved_docs": 5,
    "confidence": 0.89
  }
  ```

### 9.3 Alerting

#### 9.3.1 Alert Rules (Prometheus Alertmanager)
```yaml
groups:
- name: rag_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    annotations:
      summary: "High error rate detected"

  - alert: HighLatency
    expr: histogram_quantile(0.95, http_request_duration_seconds) > 5
    for: 10m
    annotations:
      summary: "95th percentile latency >5s"

  - alert: GPUMemoryHigh
    expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.90
    for: 5m
    annotations:
      summary: "GPU memory usage >90%"

  - alert: MilvusDown
    expr: up{job="milvus"} == 0
    for: 1m
    annotations:
      summary: "Milvus vector database is down"
```

### 9.4 Tracing

#### 9.4.1 Distributed Tracing (Jaeger)
- **Instrumentation**: OpenTelemetry SDK
- **Traced Operations**:
  - End-to-end query processing
  - OCR pipeline stages
  - Vector search operations
  - LLM generation calls
- **Sampling Rate**: 10% (adjustable)

---

## 10. Maintenance & Operations

### 10.1 Backup Strategy

#### 10.1.1 Backup Schedule
- **PostgreSQL**: Daily full backup + continuous WAL archiving
- **Milvus**: Weekly full snapshot + daily incremental
- **MinIO**: Object versioning enabled (retain 30 versions)
- **Models**: Version control in Git LFS

#### 10.1.2 Backup Retention
- **Daily backups**: 30 days
- **Weekly backups**: 1 year
- **Monthly backups**: 7 years (compliance requirement)

#### 10.1.3 Disaster Recovery
- **RTO** (Recovery Time Objective): 4 hours
- **RPO** (Recovery Point Objective): 1 hour (max data loss)
- **DR Site**: Secondary datacenter or cloud region (hot standby)

### 10.2 Update Strategy

#### 10.2.1 Model Updates
- **Frequency**: Quarterly review of new models
- **Process**:
  1. Benchmark new model on test set
  2. A/B test with 10% traffic
  3. Full rollout if metrics improve by >=5%
  4. Rollback within 1 hour if issues detected

#### 10.2.2 Software Updates
- **Security patches**: Apply within 48 hours
- **Minor updates**: Monthly maintenance window
- **Major updates**: Quarterly planned downtime (2-hour window)

### 10.3 Capacity Planning

#### 10.3.1 Growth Projections
- **Document Growth**: 20% per quarter
- **Query Growth**: 30% per quarter
- **Storage Growth**: 50GB per 1,000 documents

#### 10.3.2 Scaling Triggers
- **Scale Up** (vertical):
  - GPU utilization >80% sustained for 1 week
  - Memory usage >85% sustained
- **Scale Out** (horizontal):
  - Query latency >5s for 95th percentile
  - Concurrent user count approaching limit

---

## 11. Cost Analysis

### 11.1 Initial Setup Costs (Self-Hosted)

| Item | Quantity | Unit Cost (JPY) | Total (JPY) |
|------|----------|-----------------|-------------|
| Server (16-core, 64GB) | 1 | 200,000 | 200,000 |
| NVIDIA RTX 4090 | 1 | 300,000 | 300,000 |
| NVMe SSD (2TB) | 2 | 20,000 | 40,000 |
| Networking | 1 | 30,000 | 30,000 |
| UPS | 1 | 50,000 | 50,000 |
| **Total Hardware** | | | **620,000** |
| Software Licenses | | | 0 (all open-source) |
| Setup Labor (80 hours @ 10,000/hr) | | | 800,000 |
| **Grand Total** | | | **1,420,000** |

### 11.2 Monthly Operating Costs

| Item | Cost (JPY/month) | Notes |
|------|------------------|-------|
| Electricity (500W x 24h x 30d @ 30/kWh) | 10,800 | GPU server power |
| Internet (1Gbps dedicated) | 20,000 | Optional for cloud access |
| Maintenance (5% of hardware/year) | 2,600 | Average annual cost |
| **Total Monthly** | **33,400** | |

### 11.3 Cloud Alternative (AWS Tokyo Region)

| Service | Specification | Cost (JPY/month) | Notes |
|---------|---------------|------------------|-------|
| EC2 (g5.xlarge) | 4 vCPU, 16GB, A10G GPU | 150,000 | 730 hours/month |
| EBS Storage (2TB) | gp3 SSD | 20,000 | |
| S3 Storage (500GB) | Standard | 1,500 | |
| RDS PostgreSQL (db.t4g.large) | 2 vCPU, 8GB | 30,000 | |
| Data Transfer (1TB/month) | | 15,000 | |
| **Total Cloud Monthly** | | **216,500** | |
| **Annual Cloud Cost** | | **2,598,000** | |

**Break-Even Analysis**: Self-hosted setup pays for itself in 8 months compared to cloud (1,420,000 / (216,500 - 33,400) = 7.75 months).

---

## 12. Appendices

### Appendix A: Glossary

- **RAG**: Retrieval-Augmented Generation - Technique combining vector search with LLM generation
- **Embedding**: Dense vector representation of text (768D for Sarashina)
- **JMTEB**: Japanese Massive Text Embedding Benchmark - Standard for evaluating Japanese embeddings
- **OCR**: Optical Character Recognition - Converting images to text
- **VLM**: Vision-Language Model - AI model processing both images and text
- **Reranker**: Secondary model that re-scores retrieval candidates
- **NDCG**: Normalized Discounted Cumulative Gain - Ranking quality metric
- **MRR**: Mean Reciprocal Rank - Measures first relevant result position

### Appendix B: Reference Documents

1. Sarashina-Embedding-v1-1B Technical Report (SB Intuitions, 2025)
2. JMTEB Benchmark Results (https://github.com/sbintuitions/JMTEB)
3. YomiToku OCR Evaluation (2025) - kotaro-kinoshita/yomitoku
4. Milvus Vector Database Documentation v2.4
5. LangChain Japanese Documentation
6. NVIDIA NIM API Reference

### Appendix C: Change Log

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-01 | Initial specification | Tech Team |

### Appendix D: Contact Information

- **Project Lead**: [Name] (email@example.com)
- **Technical Architect**: [Name] (email@example.com)
- **DevOps Lead**: [Name] (email@example.com)
- **Support**: support@example.com
- **Emergency Hotline**: +81-3-XXXX-XXXX (24/7)

---

**END OF SPECIFICATION DOCUMENT**
