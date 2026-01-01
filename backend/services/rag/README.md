# RAG Orchestration Service

End-to-end Retrieval-Augmented Generation pipeline that orchestrates retrieval, reranking, and LLM generation services.

## Overview

The RAG Orchestration Service provides a unified interface for processing user queries through a complete RAG pipeline:

1. **Query Understanding** - Validate and normalize user queries
2. **Retrieval** - Hybrid vector + keyword search (via Retrieval Service)
3. **Reranking** - Cross-encoder reranking for improved relevance (via Reranking Service)
4. **Context Assembly** - Prepare retrieved documents for LLM
5. **LLM Generation** - Generate answer with retrieved contexts (via LLM Service)

## Features

- **End-to-end RAG pipeline** - Single API call for complete query processing
- **Stage-by-stage metrics** - Detailed timing and success/failure for each stage
- **Flexible configuration** - Customizable retrieval, reranking, and generation options
- **Error handling** - Graceful degradation with detailed error reporting
- **Query result caching** - Optional caching for improved performance
- **Japanese language support** - Full Japanese text normalization and processing
- **Multiple retrieval methods** - Vector, keyword, or hybrid search

## Architecture

```
User Query
    ↓
[Query Understanding] - Validate, normalize
    ↓
[Retrieval] - Hybrid search (Milvus + PostgreSQL)
    ↓ (optional)
[Reranking] - Cross-encoder reranking
    ↓
[Context Assembly] - Prepare documents for LLM
    ↓
[LLM Generation] - Qwen3:4b with RAG contexts
    ↓
Answer + Sources + Stage Timings
```

## Usage

### Basic Query

```python
from backend.services.rag import get_rag_service, RAGQueryOptions

# Get service instance
service = await get_rag_service()

# Process query
result = await service.query(
    query="機械学習と深層学習の違いは何ですか？",
    options=RAGQueryOptions(
        top_k=10,
        rerank=True,
        include_sources=True,
    ),
)

print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)} documents")
print(f"Time: {result.processing_time_ms}ms")
print(f"Confidence: {result.confidence}")
```

### Custom Options

```python
from backend.services.rag import RAGQueryOptions

options = RAGQueryOptions(
    top_k=5,                      # Number of sources to return
    retrieval_top_k=20,           # Initial retrieval count
    rerank_top_k=5,               # Documents to keep after reranking
    rerank=True,                  # Enable/disable reranking
    retrieval_method="hybrid",    # hybrid, vector, or keyword
    min_score=0.3,                # Minimum relevance score
    document_ids=["doc_123"],     # Filter by documents
    include_sources=True,         # Include source chunks
    use_cache=True,               # Use query cache
    language="ja",                # Query language
)

result = await service.query(query="...", options=options)
```

### Without Reranking

```python
# Faster queries without reranking
result = await service.query(
    query="日本の首都はどこですか？",
    options=RAGQueryOptions(
        top_k=5,
        rerank=False,  # Skip reranking
    ),
)
```

### Convenience Function

```python
from backend.services.rag import query_rag, RAGQueryOptions

# Quick query without getting service instance
result = await query_rag(
    query="Pythonとはどのようなプログラミング言語ですか？",
    options=RAGQueryOptions(top_k=5),
)
```

## Models

### RAGQueryOptions

Options for RAG query processing:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | int | 10 | Number of sources to return (1-100) |
| `retrieval_top_k` | int | 20 | Initial retrieval count |
| `rerank_top_k` | int | 10 | Documents to keep after reranking |
| `rerank` | bool | True | Enable reranking |
| `retrieval_method` | str | "hybrid" | hybrid, vector, or keyword |
| `min_score` | float | 0.0 | Minimum relevance score |
| `document_ids` | Sequence[str] | None | Filter by document IDs |
| `include_sources` | bool | True | Include source chunks |
| `use_cache` | bool | True | Use query cache |
| `language` | str | "ja" | Query language |

### RAGResult

Result from RAG query processing:

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original query text |
| `answer` | str | Generated answer |
| `sources` | List[RAGSource] | Source documents used |
| `query_id` | str | Query identifier |
| `processing_time_ms` | float | Total processing time |
| `stage_timings` | List[RAGStageMetrics] | Timing breakdown by stage |
| `confidence` | float | Answer confidence score |
| `llm_model` | str | LLM model used |
| `embedding_model` | str | Embedding model used |
| `reranker_model` | str | Reranker model used |
| `metadata` | Dict | Additional metadata |

### RAGSource

A single source document:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Chunk identifier |
| `document_id` | str | Document ID |
| `document_title` | str | Document title |
| `text` | str | Chunk text content |
| `score` | float | Relevance score |
| `rerank_score` | float | Reranking score |
| `page_number` | int | Page number |
| `chunk_index` | int | Chunk index |
| `metadata` | Dict | Additional metadata |

### RAGStageMetrics

Metrics for a single pipeline stage:

| Field | Type | Description |
|-------|------|-------------|
| `stage_name` | str | Stage name |
| `duration_ms` | float | Stage duration |
| `success` | bool | Whether stage succeeded |
| `error` | str | Error message if failed |
| `metadata` | Dict | Additional stage metadata |

## Configuration

The RAG service can be configured with `RAGPipelineConfig`:

```python
from backend.services.rag import RAGService, RAGPipelineConfig

config = RAGPipelineConfig(
    # Retrieval settings
    retrieval_method="hybrid",
    retrieval_top_k=20,
    min_score=0.0,

    # Reranking settings
    enable_reranking=True,
    rerank_top_k=10,

    # LLM settings
    llm_temperature=0.1,
    llm_max_tokens=2048,
    llm_top_p=0.9,

    # System prompt
    system_prompt=None,  # Use default

    # Cache settings
    enable_cache=True,

    # Language settings
    default_language="ja",
)

service = RAGService(config=config)
await service.initialize()
```

## Stage Timings

Each RAG query produces detailed stage timing metrics:

```python
result = await service.query(query="...")

for stage in result.stage_timings:
    print(f"{stage.stage_name}: {stage.duration_ms:.0f}ms")
    if not stage.success:
        print(f"  Error: {stage.error}")
    if stage.metadata:
        for key, value in stage.metadata.items():
            print(f"  {key}: {value}")
```

Example output:
```
query_understanding: 2ms
retrieval: 145ms (retrieved_count=20, method=hybrid)
reranking: 523ms (input_count=20, output_count=10)
context_assembly: 2ms (context_count=10)
llm_generation: 1868ms (llm_model=qwen3:4b, tokens=250)
```

## Error Handling

The RAG service provides detailed error handling:

### RAGValidationError

Raised when query validation fails:

```python
try:
    result = await service.query(query="")  # Empty query
except RAGValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Details: {e.details}")
```

### RAGProcessingError

Raised when pipeline processing fails:

```python
try:
    result = await service.query(query="...")
except RAGProcessingError as e:
    print(f"Processing error: {e.message}")
    print(f"Stage: {e.stage}")  # Which stage failed
    print(f"Details: {e.details}")
```

### RAGServiceError

Raised when service initialization fails:

```python
try:
    service = await get_rag_service()
except RAGServiceError as e:
    print(f"Service error: {e.message}")
```

## Health Check

Check the health of all RAG components:

```python
service = await get_rag_service()

health = await service.health_check()

print(f"Status: {health['status']}")
print(f"Initialized: {health['initialized']}")

for component, status in health['components'].items():
    print(f"  {component}: {status}")
```

## Performance

Expected performance on RTX 4090:

| Operation | Time | Notes |
|-----------|------|-------|
| Query Understanding | ~2ms | Normalization |
| Retrieval (hybrid) | ~100-200ms | 20 documents |
| Reranking | ~400-600ms | 20 documents |
| Context Assembly | ~2ms | 10 contexts |
| LLM Generation | ~2-4s | 200-500 tokens |
| **Total** | **~2.5-5s** | End-to-end |

### Optimization Tips

1. **Disable reranking** for faster queries (saves ~500ms)
2. **Reduce top_k** to limit retrieval and generation
3. **Lower llm_temperature** for faster generation
4. **Enable cache** for repeated queries
5. **Filter by document_ids** to reduce search space

## API Integration

The RAG service is integrated with the FastAPI query endpoint:

```python
# POST /api/v1/query
{
    "query": "機械学習とは何ですか？",
    "top_k": 5,
    "rerank": true,
    "language": "ja",
    "include_sources": true
}

# Response
{
    "query_id": "uuid",
    "query": "機械学習とは何ですか？",
    "answer": "機械学習は...",
    "sources": [...],
    "processing_time_ms": 2540,
    "stage_timings_ms": {
        "query_understanding": 2,
        "retrieval": 145,
        "reranking": 523,
        "context_assembly": 2,
        "llm_generation": 1868
    },
    "confidence": 0.85,
    "timestamp": "2026-01-02T12:00:00"
}
```

## Testing

Run manual tests:

```bash
# From within the app container
docker exec ocr-rag-app-dev python /app/tests/manual/test_rag.py

# Run specific test
docker exec ocr-rag-app-dev python /app/tests/manual/test_rag.py::test_rag_basic_query
```

Test cases:
- Basic RAG query
- RAG with reranking
- RAG without reranking
- Query validation
- Multiple queries
- Custom options
- Health check
- Stage timing breakdown

## Dependencies

The RAG service depends on:

- **Retrieval Service** - Hybrid vector + keyword search
- **Reranking Service** - Cross-encoder reranking (optional)
- **LLM Service** - Qwen3:4b generation via Ollama
- **Milvus** - Vector database
- **PostgreSQL** - Metadata database
- **Redis** - Query cache

## Troubleshooting

**Issue**: Slow query performance
- Check GPU utilization: `nvidia-smi`
- Reduce `retrieval_top_k` and `top_k`
- Disable reranking: `rerank=False`
- Enable caching: `use_cache=True`

**Issue**: Poor answer quality
- Increase `retrieval_top_k` for more context
- Enable reranking: `rerank=True`
- Adjust `min_score` threshold
- Check document quality in database

**Issue**: Service initialization failed
- Check all dependent services are running
- Verify Milvus and PostgreSQL connectivity
- Check Ollama is accessible
- Review service logs

**Issue**: Reranking errors
- Check reranker model is loaded
- Verify GPU memory availability
- Disable reranking if not critical: `rerank=False`

## Future Enhancements

- [ ] Streaming responses
- [ ] Query result pagination
- [ ] Multi-document summarization
- [ ] Follow-up question handling
- [ ] Query explanation/interpretation
- [ ] Advanced context window management
- [ ] Multi-turn conversation support
