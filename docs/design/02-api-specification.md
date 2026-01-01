# API Specification Design

**Version:** 1.0
**Date:** 2026-01-01
**Base URL:** `https://api.example.com/api/v1`
**WebSocket URL:** `wss://api.example.com/api/v1/stream`
**Authentication:** Bearer JWT Token
**Content-Type:** `application/json`

---

## 1. Authentication & Authorization

### 1.1 Authentication Flow

#### POST /auth/login
Authenticate user and receive JWT tokens.

**Request:**
```http
POST /api/v1/auth/login HTTP/1.1
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "user": {
    "user_id": "uuid-user-123",
    "name": "John Doe",
    "email": "user@example.com",
    "role": "power_user",
    "permissions": [
      "upload:document",
      "query:all",
      "view:own",
      "delete:own"
    ]
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid credentials
- `429 Too Many Requests`: Too many login attempts (rate limit: 5/minute)

#### POST /auth/refresh
Refresh access token using refresh token.

**Request:**
```http
POST /api/v1/auth/refresh HTTP/1.1
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 900
}
```

#### POST /auth/logout
Invalidate refresh token.

**Request:**
```http
POST /api/v1/auth/logout HTTP/1.1
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (204 No Content)**

---

## 2. Document Management APIs

### 2.1 Upload Document

#### POST /documents/upload
Upload a PDF document for processing.

**Request:**
```http
POST /api/v1/documents/upload HTTP/1.1
Authorization: Bearer <access_token>
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="report.pdf"
Content-Type: application/pdf

<PDF binary data>
------WebKitFormBoundary
Content-Disposition: form-data; name="metadata"

{
  "title": "2025年度 決算報告書",
  "author": "財務部",
  "tags": ["financial", "決算", "2025"],
  "language": "ja",
  "category": "financial_report"
}
------WebKitFormBoundary--
```

**Response (202 Accepted):**
```json
{
  "document_id": "uuid-doc-123",
  "status": "pending",
  "filename": "report.pdf",
  "file_size_bytes": 2048576,
  "content_type": "application/pdf",
  "upload_timestamp": "2026-01-01T12:00:00Z",
  "estimated_completion": "2026-01-01T12:05:00Z",
  "message": "Document uploaded successfully. Processing started."
}
```

**Error Responses:**
- `400 Bad Request`: Invalid file format, missing required fields
- `413 Payload Too Large`: File size exceeds 50MB limit
- `409 Conflict`: Duplicate document (same SHA256 hash already exists)
- `429 Too Many Requests`: Upload rate limit exceeded (10/minute per user)

### 2.2 Get Document Status

#### GET /documents/{document_id}/status
Retrieve the processing status of a document.

**Request:**
```http
GET /api/v1/documents/uuid-doc-123/status HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "document_id": "uuid-doc-123",
  "status": "processing",
  "progress": 45,
  "current_stage": "ocr",
  "stages": {
    "upload": {
      "status": "completed",
      "started_at": "2026-01-01T12:00:00Z",
      "completed_at": "2026-01-01T12:00:05Z",
      "duration_ms": 5000
    },
    "ocr": {
      "status": "in_progress",
      "started_at": "2026-01-01T12:00:05Z",
      "pages_processed": 9,
      "total_pages": 20,
      "current_engine": "yomitoku",
      "confidence": 0.92
    },
    "chunking": {
      "status": "pending"
    },
    "embedding": {
      "status": "pending"
    }
  },
  "created_at": "2026-01-01T12:00:00Z",
  "updated_at": "2026-01-01T12:02:30Z"
}
```

**Status Values:**
- `pending`: Queued for processing
- `processing`: Currently being processed
- `completed`: Successfully processed and indexed
- `failed`: Processing failed (check `errors` field)
- `cancelled`: User cancelled the processing

### 2.3 Get Document Details

#### GET /documents/{document_id}
Retrieve detailed information about a document.

**Request:**
```http
GET /api/v1/documents/uuid-doc-123 HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "document_id": "uuid-doc-123",
  "filename": "report.pdf",
  "file_size_bytes": 2048576,
  "file_hash": "sha256:a1b2c3d4...",
  "metadata": {
    "title": "2025年度 決算報告書",
    "author": "財務部",
    "tags": ["financial", "決算", "2025"],
    "language": "ja",
    "category": "financial_report"
  },
  "status": "completed",
  "page_count": 20,
  "chunk_count": 156,
  "ocr_confidence": 0.94,
  "thumbnail_url": "https://api.example.com/api/v1/documents/uuid-doc-123/thumbnail",
  "upload_timestamp": "2026-01-01T12:00:00Z",
  "processing_completed_at": "2026-01-01T12:05:30Z",
  "processing_duration_ms": 330000,
  "owner": {
    "user_id": "uuid-user-123",
    "name": "John Doe"
  },
  "permissions": {
    "can_view": true,
    "can_delete": true,
    "can_share": true
  }
}
```

**Error Responses:**
- `404 Not Found`: Document not found or user lacks permission

### 2.4 List Documents

#### GET /documents
List all documents with filtering and pagination.

**Request:**
```http
GET /api/v1/documents?limit=20&offset=0&status=completed&tag=financial&sort_by=upload_timestamp&order=desc HTTP/1.1
Authorization: Bearer <access_token>
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Number of results per page (default: 20, max: 100) |
| `offset` | integer | No | Number of results to skip (default: 0) |
| `status` | string | No | Filter by status: `pending`, `processing`, `completed`, `failed` |
| `tag` | string | No | Filter by tag |
| `language` | string | No | Filter by language (e.g., `ja`, `en`) |
| `owner_id` | string | No | Filter by owner user ID |
| `search` | string | No | Full-text search in title and metadata |
| `sort_by` | string | No | Sort field: `upload_timestamp`, `filename`, `file_size`, `status` |
| `order` | string | No | Sort order: `asc`, `desc` (default: `desc`) |

**Response (200 OK):**
```json
{
  "total": 156,
  "limit": 20,
  "offset": 0,
  "results": [
    {
      "document_id": "uuid-doc-123",
      "filename": "report.pdf",
      "title": "2025年度 決算報告書",
      "status": "completed",
      "page_count": 20,
      "chunk_count": 156,
      "upload_timestamp": "2026-01-01T12:00:00Z",
      "thumbnail_url": "https://api.example.com/api/v1/documents/uuid-doc-123/thumbnail",
      "tags": ["financial", "決算", "2025"],
      "owner": {
        "user_id": "uuid-user-123",
        "name": "John Doe"
      }
    }
  ]
}
```

### 2.5 Delete Document

#### DELETE /documents/{document_id}
Delete a document and all associated data.

**Request:**
```http
DELETE /api/v1/documents/uuid-doc-123 HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (204 No Content)**

**Error Responses:**
- `403 Forbidden`: User lacks delete permission
- `404 Not Found`: Document not found

### 2.6 Download Document

#### GET /documents/{document_id}/download
Download the original PDF file.

**Request:**
```http
GET /api/v1/documents/uuid-doc-123/download HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="report.pdf"
Content-Length: 2048576

<PDF binary data>
```

### 2.7 Get Document Thumbnail

#### GET /documents/{document_id}/thumbnail
Retrieve document thumbnail image.

**Request:**
```http
GET /api/v1/documents/uuid-doc-123/thumbnail HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```
Content-Type: image/png
Content-Length: 15360

<Image binary data>
```

---

## 3. Query & Search APIs

### 3.1 Query RAG System

#### POST /query
Submit a query to the RAG system.

**Request:**
```http
POST /api/v1/query HTTP/1.1
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "query": "2025年第4四半期の営業利益は？",
  "document_ids": ["uuid-doc-123", "uuid-doc-456"],
  "top_k": 5,
  "include_sources": true,
  "language": "ja",
  "stream": false
}
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | User query text (1-500 characters) |
| `document_ids` | array | No | Filter to specific documents (default: search all) |
| `top_k` | integer | No | Number of sources to retrieve (default: 5, max: 20) |
| `include_sources` | boolean | No | Include source chunks in response (default: true) |
| `language` | string | No | Query language (default: `ja`) |
| `stream` | boolean | No | Stream response tokens (default: false) |
| `rerank` | boolean | No | Apply reranker (default: true) |

**Response (200 OK) - Non-Streaming:**
```json
{
  "query_id": "uuid-query-789",
  "query": "2025年第4四半期の営業利益は？",
  "answer": "2025年第4四半期の営業利益は150億円でした。これは前年同期比で20%の増加となりました。主な要因は、売上高の増加とコスト削減効果によるものです。",
  "sources": [
    {
      "document_id": "uuid-doc-123",
      "document_title": "2025年度 決算報告書",
      "page_number": 12,
      "chunk_index": 45,
      "chunk_text": "第4四半期の営業利益は150億円となり、前年同期比120%を達成しました...",
      "relevance_score": 0.92,
      "rerank_score": 0.89
    },
    {
      "document_id": "uuid-doc-456",
      "document_title": "四半期別業績推移",
      "page_number": 5,
      "chunk_index": 23,
      "chunk_text": "営業利益の推移：Q1: 100億円、Q2: 120億円、Q3: 135億円、Q4: 150億円...",
      "relevance_score": 0.87,
      "rerank_score": 0.82
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
  "confidence": 0.89,
  "timestamp": "2026-01-01T12:10:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid query parameters
- `429 Too Many Requests`: Query rate limit exceeded (60/minute per user)
- `503 Service Unavailable`: RAG system temporarily unavailable

### 3.2 Streaming Query Response

#### POST /query (stream=true)
Submit a query and receive streaming response.

**Request:**
```http
POST /api/v1/query HTTP/1.1
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "query": "営業利益の推移をグラフで説明してください",
  "stream": true
}
```

**Response (200 OK) - Server-Sent Events:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: metadata
data: {"query_id":"uuid-query-789","timestamp":"2026-01-01T12:10:00Z"}

event: sources
data: {"document_id":"uuid-doc-123","document_title":"2025年度 決算報告書","page_number":12,"relevance_score":0.92}

event: token
data: 2025年の営業

event: token
data: 利益は、

event: token
data: 四半期ごとに

event: token
data: 増加傾向にあります。

...

event: complete
data: {"processing_time_ms":1850,"confidence":0.89}
```

**Event Types:**
- `metadata`: Query metadata (query_id, timestamp)
- `sources`: Retrieved source chunks
- `token`: Individual response token
- `complete`: Processing complete (timing, confidence)

### 3.3 WebSocket Streaming

#### WebSocket Connection
Establish a persistent WebSocket connection for real-time queries.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.example.com/api/v1/stream?token=<access_token>');

// Send query
ws.send(JSON.stringify({
  type: 'query',
  query: '営業利益の推移を教えてください',
  top_k: 5,
  include_sources: true
}));

// Receive messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'metadata':
      console.log('Query ID:', data.query_id);
      break;
    case 'sources':
      console.log('Sources:', data.sources);
      break;
    case 'token':
      appendToResponse(data.content);
      break;
    case 'complete':
      console.log('Complete in', data.processing_time_ms, 'ms');
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};
```

**WebSocket Message Format:**

Client → Server:
```json
{
  "type": "query",
  "query": "営業利益の推移を教えてください",
  "document_ids": ["uuid-doc-123"],
  "top_k": 5,
  "include_sources": true,
  "rerank": true
}
```

Server → Client (metadata):
```json
{
  "type": "metadata",
  "query_id": "uuid-query-789",
  "timestamp": "2026-01-01T12:10:00Z"
}
```

Server → Client (sources):
```json
{
  "type": "sources",
  "sources": [
    {
      "document_id": "uuid-doc-123",
      "document_title": "2025年度 決算報告書",
      "page_number": 12,
      "relevance_score": 0.92
    }
  ]
}
```

Server → Client (token):
```json
{
  "type": "token",
  "content": "2025年の営業利益は、",
  "index": 0
}
```

Server → Client (complete):
```json
{
  "type": "complete",
  "processing_time_ms": 1850,
  "confidence": 0.89
}
```

Server → Client (error):
```json
{
  "type": "error",
  "code": "rate_limit_exceeded",
  "message": "Query rate limit exceeded. Please try again later."
}
```

### 3.4 Search Documents

#### GET /documents/search
Full-text search across document metadata.

**Request:**
```http
GET /api/v1/documents/search?q=決算&limit=10&offset=0 HTTP/1.1
Authorization: Bearer <access_token>
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query |
| `limit` | integer | No | Results per page (default: 10, max: 100) |
| `offset` | integer | No | Results to skip (default: 0) |
| `filters` | object | No | Additional filters (JSON string) |

**Response (200 OK):**
```json
{
  "total": 23,
  "limit": 10,
  "offset": 0,
  "results": [
    {
      "document_id": "uuid-doc-123",
      "filename": "report.pdf",
      "title": "2025年度 決算報告書",
      "upload_date": "2025-12-15T10:00:00Z",
      "page_count": 45,
      "snippet": "...2025年度の決算について...",
      "match_score": 0.87,
      "tags": ["financial", "決算", "2025"]
    }
  ]
}
```

---

## 4. Administration APIs

### 4.1 System Health

#### GET /health
Check system health status.

**Request:**
```http
GET /api/v1/health HTTP/1.1
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-01T12:00:00Z",
  "services": {
    "api": {
      "status": "healthy",
      "uptime_seconds": 86400
    },
    "milvus": {
      "status": "healthy",
      "collections": 3,
      "total_vectors": 1250000
    },
    "postgres": {
      "status": "healthy",
      "connections": 15,
      "latency_ms": 2
    },
    "minio": {
      "status": "healthy",
      "buckets": 3
    },
    "redis": {
      "status": "healthy",
      "memory_used_bytes": 536870912,
      "memory_peak_bytes": 1073741824
    },
    "ollama": {
      "status": "healthy",
      "loaded_models": ["qwen2.5:14b", "llama-3.2-nv-rerankqa-1b"]
    }
  },
  "gpu": {
    "gpu_id": 0,
    "name": "NVIDIA RTX 4090",
    "utilization_percent": 45,
    "memory_used_mb": 10240,
    "memory_total_mb": 24576
  }
}
```

### 4.2 System Statistics

#### GET /stats
Get system usage statistics.

**Request:**
```http
GET /api/v1/stats HTTP/1.1
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "documents": {
    "total": 1250,
    "by_status": {
      "completed": 1200,
      "processing": 15,
      "pending": 25,
      "failed": 10
    },
    "total_pages": 62500,
    "total_chunks": 312500
  },
  "queries": {
    "total": 15000,
    "last_24h": 500,
    "average_latency_ms": 1850,
    "cache_hit_rate": 0.35
  },
  "users": {
    "total": 50,
    "active_today": 15
  },
  "storage": {
    "vector_db_size_gb": 15.5,
    "object_storage_size_gb": 125.0,
    "database_size_gb": 2.5
  }
}
```

### 4.3 User Management

#### GET /users
List all users (Admin only).

**Request:**
```http
GET /api/v1/users?limit=20&offset=0 HTTP/1.1
Authorization: Bearer <admin_access_token>
```

**Response (200 OK):**
```json
{
  "total": 50,
  "limit": 20,
  "offset": 0,
  "results": [
    {
      "user_id": "uuid-user-123",
      "name": "John Doe",
      "email": "john@example.com",
      "role": "power_user",
      "document_count": 25,
      "query_count": 500,
      "created_at": "2025-01-01T00:00:00Z",
      "last_active": "2026-01-01T11:30:00Z"
    }
  ]
}
```

#### POST /users
Create a new user (Admin only).

**Request:**
```http
POST /api/v1/users HTTP/1.1
Authorization: Bearer <admin_access_token>
Content-Type: application/json

{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "password": "secure_password",
  "role": "user"
}
```

**Response (201 Created):**
```json
{
  "user_id": "uuid-user-456",
  "name": "Jane Smith",
  "email": "jane@example.com",
  "role": "user",
  "created_at": "2026-01-01T12:00:00Z"
}
```

#### DELETE /users/{user_id}
Delete a user (Admin only).

**Request:**
```http
DELETE /api/v1/users/uuid-user-456 HTTP/1.1
Authorization: Bearer <admin_access_token>
```

**Response (204 No Content)**

---

## 5. Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameter",
    "details": {
      "field": "query",
      "reason": "Query must be between 1 and 500 characters"
    },
    "timestamp": "2026-01-01T12:00:00Z",
    "request_id": "req-abc123"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| `400` | `invalid_request` | Invalid request parameters |
| `401` | `unauthorized` | Missing or invalid authentication |
| `403` | `forbidden` | Insufficient permissions |
| `404` | `not_found` | Resource not found |
| `409` | `conflict` | Resource already exists |
| `413` | `payload_too_large` | Request exceeds size limit |
| `429` | `rate_limit_exceeded` | Rate limit exceeded |
| `500` | `internal_error` | Internal server error |
| `503` | `service_unavailable` | Service temporarily unavailable |

---

## 6. Rate Limiting

### Rate Limit Headers

All rate-limited endpoints include these headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704096900
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window | Scope |
|----------|-------|--------|-------|
| `POST /auth/login` | 5 | 1 minute | per IP |
| `POST /documents/upload` | 10 | 1 minute | per user |
| `POST /query` | 60 | 1 minute | per user |
| `GET /documents` | 200 | 1 minute | per user |
| All other endpoints | 1000 | 1 hour | per user |

### Rate Limit Exceeded Response

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60

{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please try again in 60 seconds.",
    "retry_after": 60
  }
}
```

---

## 7. Webhook Notifications

### Webhook Events

The system can send webhook notifications for certain events:

#### Document Processing Complete

```json
{
  "event": "document.completed",
  "timestamp": "2026-01-01T12:05:30Z",
  "data": {
    "document_id": "uuid-doc-123",
    "status": "completed",
    "page_count": 20,
    "chunk_count": 156,
    "ocr_confidence": 0.94
  }
}
```

#### Document Processing Failed

```json
{
  "event": "document.failed",
  "timestamp": "2026-01-01T12:05:30Z",
  "data": {
    "document_id": "uuid-doc-456",
    "status": "failed",
    "error": "OCR confidence below threshold",
    "stage": "ocr"
  }
}
```

---

**END OF API SPECIFICATION**
