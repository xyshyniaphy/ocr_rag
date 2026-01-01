# Reranking Service

Cross-encoder reranking for improved search result relevance using NVIDIA Llama-3.2-NV-RerankQA-1B-v2.

## Overview

The Reranking Service provides document reranking capabilities using NVIDIA's Llama-3.2-NV-RerankQA-1B-v2 cross-encoder model. It takes a query and a list of retrieved documents, then reorders them based on relevance scores.

## Features

- **Cross-encoder reranking**: More accurate than bi-encoder embeddings for relevance scoring
- **GPU-accelerated**: Optimized for NVIDIA GPUs (cuda:0)
- **Japanese support**: Multilingual model optimized for Japanese text
- **Threshold filtering**: Filter out low-relevance documents
- **Batch processing**: Efficient batch inference for multiple documents
- **Model caching**: Downloads model once and caches via Docker volume

## Model Details

- **Model**: `nvidia/Llama-3.2-NV-RerankQA-1B-v2`
- **Parameters**: 1B
- **Architecture**: Cross-encoder (BERT-like)
- **Max Length**: 512 tokens per query-document pair
- **Languages**: Multilingual (including Japanese)
- **Inference**: GPU-accelerated with fp16

## Usage

### Basic Usage

```python
from backend.services.reranker import get_reranking_service

# Get service instance
service = await get_reranking_service()

# Rerank documents
query = "機械学習とは何ですか？"
documents = [
    {"text": "機械学習は...", "doc_id": "doc1"},
    {"text": "深層学習は...", "doc_id": "doc2"},
]

result = await service.rerank(query, documents)

# Access reranked results
for r in result.results:
    print(f"Rank {r.rank}: {r.doc_id} (score: {r.score:.3f})")
    print(f"  {r.text[:100]}...")
```

### Simple API

```python
# For simple list of texts
texts = ["doc1 text", "doc2 text", "doc3 text"]
results = await service.rerank_simple(
    query="query text",
    texts=texts,
    top_k=5
)

# Returns: List[(original_index, text, score)]
for idx, text, score in results:
    print(f"[{score:.3f}] {text[:50]}...")
```

### Custom Options

```python
from backend.services.reranker.models import RerankerOptions

options = RerankerOptions(
    top_k_input=20,      # Number of input documents
    top_k_output=5,      # Number of output documents
    threshold=0.65,      # Minimum relevance score (0-1)
    batch_size=32,       # Batch size for inference
)

result = await service.rerank(query, documents, options=options)
```

## Configuration

Environment variables in `backend/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_MODEL` | `nvidia/Llama-3.2-NV-RerankQA-1B-v2` | Model name |
| `RERANKER_MODEL_PATH` | `/app/reranker_models/llama-nv-reranker` | Local model path |
| `RERANKER_DEVICE` | `cuda:0` | Device for inference |
| `RERANKER_TOP_K_INPUT` | `20` | Default input documents |
| `RERANKER_TOP_K_OUTPUT` | `5` | Default output documents |
| `RERANKER_THRESHOLD` | `0.65` | Default relevance threshold |
| `RERANKER_BATCH_SIZE` | `32` | Default batch size |

## Model Caching

The reranker model is **NOT included in the base Docker image**. It is downloaded on first use and cached in a Docker volume:

```yaml
# docker-compose.dev.yml
volumes:
  - reranker_models_dev:/app/reranker_models:rw

volumes:
  reranker_models_dev:
    name: ocr-rag-reranker-models-dev
```

On first startup, the model (~2GB) will be downloaded from HuggingFace Hub to `/app/reranker_models/huggingface_cache/`. Subsequent runs will use the cached model.

## Testing

Run manual tests:

```bash
# From within the app container
docker exec ocr-rag-app-dev python /app/tests/manual/test_reranker.py
```

Test cases:
- Basic reranking functionality
- Custom options (threshold, top_k)
- Simple API usage
- Threshold filtering
- Multilingual support (Japanese/English)
- Health check
- Performance benchmarks

## Performance

Expected performance on RTX 4090:
- **First query**: ~500-1000ms (model loading + inference)
- **Subsequent queries**: ~50-200ms (20 documents)
- **Per-document**: ~2-10ms

## Integration with RAG Pipeline

The reranker is typically used after initial retrieval:

```python
# 1. Initial retrieval (vector/keyword search)
retrieval_result = await retrieval_service.retrieve(query, top_k=20)

# 2. Rerank top results
documents = [{"text": c.text, "doc_id": c.chunk_id} for c in retrieval_result.chunks]
rerank_result = await reranker_service.rerank(query, documents)

# 3. Use reranked results for LLM generation
context = "\n\n".join([r.text for r in rerank_result.results])
```

## Architecture

```
Reranking Service
├── models.py          # Pydantic models and exceptions
├── llama_nv.py        # Llama-3.2-NV Reranker model
├── service.py         # Main service class
└── __init__.py        # Package exports
```

## GPU Memory Usage

Typical GPU memory allocation:
- **Model loading**: ~2GB VRAM
- **Inference (batch=32)**: ~3-4GB VRAM peak
- **Recommended**: 4GB+ VRAM available

Total system allocation (with other services):
- OCR (YomiToku): 40% VRAM (~9.6GB)
- Embedding (Sarashina): 30% VRAM (~7.2GB)
- Reranker: 10% VRAM (~2.4GB)
- LLM (Qwen): 20% VRAM (~4.8GB)
