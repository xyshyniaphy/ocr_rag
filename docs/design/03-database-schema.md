# Database Schema Design

**Version:** 1.0
**Date:** 2026-01-01
**Databases:** Milvus 2.4+ (Vector), PostgreSQL 16+ (Metadata)

---

## 1. Milvus Vector Database Schema

### 1.1 Collection: `document_chunks`

Stores text chunks with their embeddings for semantic search.

**Schema Definition:**

```python
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

# Define fields
fields = [
    # Primary key
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),

    # Vector embedding (768D for Sarashina-1B)
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),

    # Content
    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=4096),

    # References
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="page_number", dtype=DataType.INT32),
    FieldSchema(name="chunk_index", dtype=DataType.INT32),

    # Metadata (JSON)
    FieldSchema(name="metadata", dtype=DataType.JSON),

    # Timestamps
    FieldSchema(name="created_at", dtype=DataType.INT64),  # Unix timestamp
    FieldSchema(name="updated_at", dtype=DataType.INT64),  # Unix timestamp

    # Optimization fields
    FieldSchema(name="token_count", dtype=DataType.INT32),
    FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),  # 'text', 'table', 'header'
]

# Create schema
schema = CollectionSchema(
    fields=fields,
    description="Document chunks with embeddings for semantic search",
    enable_dynamic_field=True  # Allow custom fields
)

# Create collection
collection = Collection(
    name="document_chunks",
    schema=schema
)
```

**Index Configuration:**

```python
# IVF_FLAT index (balance between speed and accuracy)
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # Inner Product for L2-normalized vectors
    "params": {
        "nlist": 1024  # Number of clusters (sqrt of expected vectors)
    }
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)
```

**Search Parameters:**

```python
search_params = {
    "metric_type": "IP",
    "params": {
        "nprobe": 128  # Number of clusters to search (nlist/8)
    },
    "expr": "language == 'ja'"  # Optional filter expression
}
```

**Example Data:**

```json
{
  "chunk_id": "chk_20260101_0001",
  "embedding": [0.0123, -0.0456, ..., 0.0789],  // 768 floats (L2-normalized)
  "text_content": "第4四半期の営業利益は150億円となり、前年同期比で20%の増加を達成しました。",
  "document_id": "doc_20260101_0001",
  "page_number": 12,
  "chunk_index": 45,
  "metadata": {
    "section_header": "第2章 営業成績",
    "paragraph_index": 3,
    "is_table": false,
    "confidence": 0.94
  },
  "created_at": 1704096000,
  "updated_at": 1704096000,
  "token_count": 45,
  "language": "ja",
  "chunk_type": "text"
}
```

### 1.2 Milvus Collection Operations

**Insert:**

```python
import numpy as np
from pymilvus import utility

# Prepare data
data = [
    [chunk_id],           # chunk_id
    [embedding_vector],   # embedding (768D list)
    [text_content],       # text_content
    [document_id],        # document_id
    [page_number],        # page_number
    [chunk_index],        # chunk_index
    [metadata_dict],      # metadata
    [created_ts],         # created_at
    [updated_ts],         # updated_at
    [token_count],        # token_count
    [language],           # language
    [chunk_type]          # chunk_type
]

# Insert (batch)
collection.insert(data)

# Flush to disk
collection.flush()
```

**Search:**

```python
# Load collection into memory
collection.load()

# Search
results = collection.search(
    data=[query_embedding],  # Query vector (768D)
    anns_field="embedding",
    param=search_params,
    limit=20,  # Top-K results
    expr=None,  # Optional filter
    output_fields=["text_content", "document_id", "page_number", "metadata"]
)

# Process results
for hit in results[0]:
    print(f"Score: {hit.score:.4f}")
    print(f"Text: {hit.entity.get('text_content')}")
    print(f"Document: {hit.entity.get('document_id')}")
    print(f"Page: {hit.entity.get('page_number')}")
```

**Delete:**

```python
# Delete by chunk_id
collection.delete(expr=f"chunk_id in ['chk_001', 'chk_002']")

# Delete by document_id (cascade)
collection.delete(expr=f"document_id == 'doc_001'")
```

---

## 2. PostgreSQL Metadata Database Schema

### 2.1 Schema Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL Schema                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │   users      │────▶│  documents   │────▶│   chunks     │          │
│  │              │     │              │     │              │          │
│  └──────────────┘     └──────────────┘     └──────────────┘          │
│         │                    │                    │                    │
│         │                    ▼                    ▼                    │
│         │            ┌──────────────┐     ┌──────────────┐          │
│         │            │   queries    │     │  permissions │          │
│         │            │              │     │              │          │
│         │            └──────────────┘     └──────────────┘          │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                   │
│  │   audit_log  │                                                   │
│  │              │                                                   │
│  └──────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Table: `users`

Stores user account information.

```sql
CREATE TABLE users (
    -- Primary key
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Authentication
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- bcrypt hash

    -- Profile
    full_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),

    -- Role & Permissions
    role VARCHAR(20) NOT NULL DEFAULT 'user',  -- 'admin', 'power_user', 'user', 'viewer'
    permissions JSONB DEFAULT '{}',  -- Additional custom permissions

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,  -- Soft delete

    -- Constraints
    CONSTRAINT valid_role CHECK (role IN ('admin', 'power_user', 'user', 'viewer'))
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_is_active ON users(is_active) WHERE is_active = true;

-- Comments
COMMENT ON TABLE users IS 'User accounts with authentication and role-based access control';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hash of the password (not plain text)';
COMMENT ON COLUMN users.permissions IS 'JSON object with custom permissions like {"upload": true, "delete": false}';
```

**Example Data:**

```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "john.doe@example.com",
  "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7IXEBJxqLu",
  "full_name": "John Doe",
  "display_name": "John",
  "role": "power_user",
  "permissions": {
    "upload:document": true,
    "query:all": true,
    "view:own": true,
    "delete:own": true,
    "share:own": true
  },
  "is_active": true,
  "is_verified": true,
  "created_at": "2025-01-01T00:00:00Z",
  "last_login_at": "2026-01-01T11:30:00Z"
}
```

### 2.3 Table: `documents`

Stores document metadata and status.

```sql
CREATE TABLE documents (
    -- Primary key
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Ownership
    owner_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- File Information
    filename VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 hash for deduplication
    content_type VARCHAR(100) NOT NULL,  -- 'application/pdf'

    -- Metadata (customizable)
    title VARCHAR(512),
    author VARCHAR(255),
    subject VARCHAR(255),
    keywords TEXT[],  -- Array of keywords/tags
    language VARCHAR(10) DEFAULT 'ja',
    category VARCHAR(100),

    -- Extended metadata (JSONB for flexibility)
    metadata JSONB DEFAULT '{}',  -- Custom fields like department, date, etc.

    -- Processing Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    ocr_status VARCHAR(20),  -- 'pending', 'in_progress', 'completed', 'failed'
    ocr_engine VARCHAR(50),  -- 'yomitoku', 'paddleocr_vl'
    ocr_confidence FLOAT,

    -- Document Statistics
    page_count INT,
    chunk_count INT DEFAULT 0,
    total_tokens INT,

    -- Storage References
    storage_path VARCHAR(1024),  -- MinIO path to original PDF
    ocr_output_path VARCHAR(1024),  -- MinIO path to OCR output
    thumbnail_path VARCHAR(1024),  -- MinIO path to thumbnail

    -- Timestamps
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,  -- Soft delete

    -- Error Handling
    error_message TEXT,
    retry_count INT DEFAULT 0,

    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_ocr_status CHECK (ocr_status IN ('pending', 'in_progress', 'completed', 'failed')),
    CONSTRAINT valid_file_size CHECK (file_size_bytes > 0 AND file_size_bytes <= 52428800),  -- Max 50MB
    CONSTRAINT positive_confidence CHECK (ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1))
);

-- Indexes
CREATE INDEX idx_documents_owner_id ON documents(owner_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_language ON documents(language);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_uploaded_at ON documents(uploaded_at DESC);
CREATE INDEX idx_documents_keywords ON documents USING GIN(keywords);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- Full-text search (Japanese)
CREATE INDEX idx_documents_title_fts ON documents USING GIN(to_tsvector('japanese', title));
CREATE INDEX idx_documents_author_fts ON documents USING GIN(to_tsvector('japanese', author));

-- Comments
COMMENT ON TABLE documents IS 'Document metadata and processing status';
COMMENT ON COLUMN documents.file_hash IS 'SHA256 hash for deduplication detection';
COMMENT ON COLUMN documents.metadata IS 'Custom metadata fields (JSONB) for flexibility';
COMMENT ON COLUMN documents.keywords IS 'Array of tags/keywords for filtering';
```

**Example Data:**

```json
{
  "document_id": "660e8400-e29b-41d4-a716-446655440001",
  "owner_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "2025年度決算報告書.pdf",
  "file_size_bytes": 2048576,
  "file_hash": "a1b2c3d4e5f6...",
  "content_type": "application/pdf",
  "title": "2025年度 決算報告書",
  "author": "財務部",
  "keywords": ["financial", "決算", "2025", "annual_report"],
  "language": "ja",
  "category": "financial_report",
  "metadata": {
    "fiscal_year": "2025",
    "department": "finance",
    "report_type": "annual",
    "publication_date": "2025-12-15"
  },
  "status": "completed",
  "ocr_status": "completed",
  "ocr_engine": "yomitoku",
  "ocr_confidence": 0.94,
  "page_count": 45,
  "chunk_count": 234,
  "total_tokens": 156780,
  "storage_path": "minio://raw-pdfs/2025/12/660e8400-e29b-41d4-a716-446655440001.pdf",
  "ocr_output_path": "minio://ocr-outputs/2025/12/660e8400-e29b-41d4-a716-446655440001.md",
  "thumbnail_path": "minio://thumbnails/2025/12/660e8400-e29b-41d4-a716-446655440001.png",
  "uploaded_at": "2026-01-01T12:00:00Z",
  "processing_completed_at": "2026-01-01T12:05:30Z"
}
```

### 2.4 Table: `chunks`

Stores text chunk metadata (references Milvus).

```sql
CREATE TABLE chunks (
    -- Primary key
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- References
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    milvus_id VARCHAR(64) NOT NULL,  -- References Milvus chunk_id

    -- Position
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,  -- Sequential order within document

    -- Content
    text_content TEXT NOT NULL,
    token_count INT,

    -- Metadata
    chunk_type VARCHAR(20) DEFAULT 'text',  -- 'text', 'table', 'header', 'footer'
    section_header VARCHAR(512),
    paragraph_index INT,

    -- Quality
    confidence FLOAT,
    is_table BOOLEAN DEFAULT false,
    table_row_count INT,
    table_col_count INT,

    -- Embedding info
    embedding_model VARCHAR(100),
    embedding_dimension INT DEFAULT 768,
    embedding_created_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_chunk_type CHECK (chunk_type IN ('text', 'table', 'header', 'footer', 'list')),
    CONSTRAINT positive_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    CONSTRAINT unique_milvus_id UNIQUE(milvus_id)
);

-- Indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_page_number ON chunks(page_number);
CREATE INDEX idx_chunks_milvus_id ON chunks(milvus_id);
CREATE INDEX idx_chunks_chunk_type ON chunks(chunk_type);
CREATE INDEX idx_chunks_document_page ON chunks(document_id, page_number);

-- Full-text search
CREATE INDEX idx_chunks_text_fts ON chunks USING GIN(to_tsvector('japanese', text_content));

-- Comments
COMMENT ON TABLE chunks IS 'Text chunk metadata (actual embeddings stored in Milvus)';
COMMENT ON COLUMN chunks.milvus_id IS 'References chunk_id in Milvus vector database';
```

**Example Data:**

```json
{
  "chunk_id": "770e8400-e29b-41d4-a716-446655440002",
  "document_id": "660e8400-e29b-41d4-a716-446655440001",
  "milvus_id": "chk_20260101_0001",
  "page_number": 12,
  "chunk_index": 45,
  "text_content": "第4四半期の営業利益は150億円となり、前年同期比で20%の増加を達成しました。",
  "token_count": 45,
  "chunk_type": "text",
  "section_header": "第2章 営業成績",
  "paragraph_index": 3,
  "confidence": 0.94,
  "embedding_model": "sbintuitions/sarashina-embedding-v1-1b",
  "embedding_dimension": 768
}
```

### 2.5 Table: `queries`

Stores query history and feedback.

```sql
CREATE TABLE queries (
    -- Primary key
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- User
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,

    -- Query
    query_text TEXT NOT NULL,
    query_language VARCHAR(10) DEFAULT 'ja',
    query_type VARCHAR(20) DEFAULT 'semantic',  -- 'semantic', 'keyword', 'hybrid'

    -- Filters
    filtered_document_ids TEXT[],  -- Array of document_ids if filtered
    top_k INT DEFAULT 5,

    -- Results
    retrieved_chunk_ids TEXT[],  -- Milvus chunk_ids
    retrieved_count INT,
    reranked_count INT,

    -- Response
    answer TEXT,
    confidence FLOAT,
    sources JSONB,  -- Array of source references

    -- Performance
    processing_time_ms INT,
    stage_timings_ms JSONB,  -- {"vector_search": 120, "llm_generation": 1280}

    -- Model info
    llm_model VARCHAR(100),
    embedding_model VARCHAR(100),
    reranker_model VARCHAR(100),

    -- Feedback
    user_rating INT,  -- 1-5 stars
    user_feedback TEXT,
    is_helpful BOOLEAN,
    reported BOOLEAN DEFAULT false,
    report_reason VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_query_type CHECK (query_type IN ('semantic', 'keyword', 'hybrid')),
    CONSTRAINT valid_rating CHECK (user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)),
    CONSTRAINT positive_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Indexes
CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_created_at ON queries(created_at DESC);
CREATE INDEX idx_queries_query_type ON queries(query_type);
CREATE INDEX idx_queries_llm_model ON queries(llm_model);
CREATE INDEX idx_queries_user_rating ON queries(user_rating);

-- Full-text search (for analytics)
CREATE INDEX idx_queries_text_fts ON queries USING GIN(to_tsvector('japanese', query_text));

-- Comments
COMMENT ON TABLE queries IS 'Query history and user feedback for analytics';
COMMENT ON COLUMN queries.stage_timings_ms IS 'Breakdown of time spent in each pipeline stage';
COMMENT ON COLUMN queries.sources IS 'Array of source references with document_id, page_number, etc.';
```

**Example Data:**

```json
{
  "query_id": "880e8400-e29b-41d4-a716-446655440003",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_text": "2025年第4四半期の営業利益は？",
  "query_language": "ja",
  "query_type": "hybrid",
  "top_k": 5,
  "retrieved_chunk_ids": ["chk_20260101_0001", "chk_20260101_0002"],
  "retrieved_count": 20,
  "reranked_count": 5,
  "answer": "2025年第4四半期の営業利益は150億円でした...",
  "confidence": 0.89,
  "sources": [
    {
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "document_title": "2025年度 決算報告書",
      "page_number": 12,
      "relevance_score": 0.92
    }
  ],
  "processing_time_ms": 1850,
  "stage_timings_ms": {
    "query_understanding": 10,
    "vector_search": 120,
    "keyword_search": 80,
    "reranking": 340,
    "context_assembly": 20,
    "llm_generation": 1280
  },
  "llm_model": "qwen2.5:14b-instruct-q4_K_M",
  "embedding_model": "sbintuitions/sarashina-embedding-v1-1b",
  "reranker_model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
  "user_rating": 5,
  "is_helpful": true,
  "created_at": "2026-01-01T12:10:00Z"
}
```

### 2.6 Table: `permissions`

Document-level access control (ACLs).

```sql
CREATE TABLE permissions (
    -- Primary key
    permission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Resource
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    -- Subject (who has permission)
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(20),  -- 'viewer', 'editor', 'owner' (for role-based permissions)

    -- Permissions
    can_view BOOLEAN DEFAULT false,
    can_download BOOLEAN DEFAULT false,
    can_delete BOOLEAN DEFAULT false,
    can_share BOOLEAN DEFAULT false,

    -- Timestamps
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    granted_by UUID REFERENCES users(user_id),
    expires_at TIMESTAMPTZ,  -- Optional expiration

    -- Constraints
    CONSTRAINT has_subject CHECK (user_id IS NOT NULL OR role IS NOT NULL),
    CONSTRAINT valid_role CHECK (role IS NULL OR role IN ('viewer', 'editor', 'owner')),
    CONSTRAINT no_self_grant CHECK (granted_by IS NULL OR granted_by != user_id)
);

-- Indexes
CREATE INDEX idx_permissions_document_id ON permissions(document_id);
CREATE INDEX idx_permissions_user_id ON permissions(user_id);
CREATE INDEX idx_permissions_role ON permissions(role);
CREATE INDEX idx_permissions_expires_at ON permissions(expires_at) WHERE expires_at IS NOT NULL;

-- Unique constraint (one permission record per user per document)
CREATE UNIQUE INDEX idx_permissions_unique ON permissions(document_id, user_id) WHERE user_id IS NOT NULL;

-- Comments
COMMENT ON TABLE permissions IS 'Document-level access control lists (ACLs)';
```

### 2.7 Table: `audit_log`

Comprehensive audit trail for compliance.

```sql
CREATE TABLE audit_log (
    -- Primary key
    audit_id BIGSERIAL PRIMARY KEY,

    -- Actor
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    user_email VARCHAR(255),
    user_role VARCHAR(20),

    -- Action
    action VARCHAR(100) NOT NULL,  -- 'document.upload', 'document.delete', 'query.execute', 'user.create'
    resource_type VARCHAR(50),  -- 'document', 'user', 'query', 'system'
    resource_id VARCHAR(255),

    -- Details
    action_details JSONB,
    ip_address INET,
    user_agent TEXT,

    -- Result
    status VARCHAR(20) NOT NULL,  -- 'success', 'failure', 'partial'
    error_message TEXT,

    -- Performance
    response_time_ms INT,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('success', 'failure', 'partial'))
);

-- Indexes
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);
CREATE INDEX idx_audit_log_status ON audit_log(status);

-- Partitioning (by month for performance)
CREATE TABLE audit_log_partitioned (
    LIKE audit_log INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE audit_log_2025_01 PARTITION OF audit_log_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Comments
COMMENT ON TABLE audit_log IS 'Comprehensive audit trail for security and compliance';
COMMENT ON COLUMN audit_log.action_details IS 'JSON object with action-specific details';
```

---

## 3. Database Functions & Triggers

### 3.1 Auto-update Timestamps

```sql
-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 3.2 Document Status Trigger

```sql
-- Function to update chunk count on chunk insert/delete
CREATE OR REPLACE FUNCTION update_document_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documents
        SET chunk_count = chunk_count + 1
        WHERE document_id = NEW.document_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documents
        SET chunk_count = GREATEST(chunk_count - 1, 0)
        WHERE document_id = OLD.document_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger
CREATE TRIGGER trigger_update_chunk_count
    AFTER INSERT OR DELETE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_document_chunk_count();
```

### 3.3 Full-text Search Function

```sql
-- Function for Japanese full-text search
CREATE OR REPLACE FUNCTION search_japanese(
    search_query TEXT,
    limit_count INT DEFAULT 20,
    offset_count INT DEFAULT 0
)
RETURNS TABLE (
    document_id UUID,
    title VARCHAR(512),
    filename VARCHAR(512),
    snippet TEXT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.document_id,
        d.title,
        d.filename,
        ts_headline('japanese', d.title, query) AS snippet,
        ts_rank(vector, query) AS rank
    FROM documents d,
         to_tsquery('japanese', search_query) query
    WHERE to_tsvector('japanese', d.title || ' ' || COALESCE(d.author, '')) @@ query
      AND d.deleted_at IS NULL
      AND d.status = 'completed'
    ORDER BY rank DESC
    LIMIT limit_count OFFSET offset_count;
END;
$$ LANGUAGE plpgsql;

-- Usage: SELECT * FROM search_japanese('決算 & 報告書');
```

---

## 4. Database Views

### 4.1 Document Summary View

```sql
CREATE VIEW document_summary AS
SELECT
    d.document_id,
    d.title,
    d.filename,
    d.status,
    d.page_count,
    d.chunk_count,
    d.ocr_confidence,
    d.uploaded_at,
    u.user_id,
    u.full_name AS owner_name,
    u.email AS owner_email,
    COUNT(DISTINCT q.query_id) AS query_count,
    AVG(q.user_rating) AS average_rating
FROM documents d
JOIN users u ON d.owner_id = u.user_id
LEFT JOIN queries q ON q.retrieved_chunk_ids @> (
    SELECT array_agg(c.milvus_id::TEXT)
    FROM chunks c
    WHERE c.document_id = d.document_id
)
WHERE d.deleted_at IS NULL
GROUP BY d.document_id, u.user_id
ORDER BY d.uploaded_at DESC;

COMMENT ON VIEW document_summary IS 'Aggregated document information with user and query statistics';
```

### 4.2 User Activity View

```sql
CREATE VIEW user_activity AS
SELECT
    u.user_id,
    u.full_name,
    u.email,
    u.role,
    u.created_at AS joined_at,
    u.last_login_at,
    COUNT(DISTINCT d.document_id) AS document_count,
    COUNT(DISTINCT q.query_id) AS query_count,
    AVG(q.processing_time_ms) AS avg_query_time_ms,
    AVG(q.user_rating) AS avg_query_rating
FROM users u
LEFT JOIN documents d ON u.user_id = d.owner_id AND d.deleted_at IS NULL
LEFT JOIN queries q ON u.user_id = q.user_id
WHERE u.deleted_at IS NULL
GROUP BY u.user_id
ORDER BY u.created_at DESC;

COMMENT ON VIEW user_activity IS 'User activity summary with document and query statistics';
```

---

## 5. Backup & Migration

### 5.1 Backup Strategy

```bash
# Full backup
pg_dump -h localhost -U postgres -d rag_metadata -F c -b -v -f rag_metadata_backup.dump

# Schema-only backup
pg_dump -h localhost -U postgres -d rag_metadata --schema-only -f schema_backup.sql

# Data-only backup
pg_dump -h localhost -U postgres -d rag_metadata --data-only -f data_backup.sql

# Restore
pg_restore -h localhost -U postgres -d rag_metadata_new -v rag_metadata_backup.dump
```

### 5.2 Migration Example

```sql
-- Migration: Add index on document metadata
BEGIN;

-- Add GIN index for JSONB queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_gin
    ON documents USING GIN(metadata);

-- Add comment
COMMENT ON INDEX idx_documents_metadata_gin IS 'GIN index for efficient JSONB queries';

COMMIT;
```

---

## 6. Monitoring Queries

### 6.1 Storage Statistics

```sql
-- Database size
SELECT
    pg_size_pretty(pg_database_size('rag_metadata')) AS database_size;

-- Table sizes
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 6.2 Performance Statistics

```sql
-- Slow queries (requires pg_stat_statements extension)
SELECT
    query,
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time AS avg_time_ms,
    max_exec_time AS max_time_ms
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Table bloat
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    n_live_tup,
    n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

---

**END OF DATABASE SCHEMA DESIGN**
