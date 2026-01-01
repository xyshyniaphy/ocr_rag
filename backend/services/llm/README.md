# LLM Service

Large Language Model generation using Qwen models via Ollama.

## Overview

The LLM Service provides text generation capabilities using Alibaba's Qwen models served through Ollama. It supports chat completions, text completions, and RAG-augmented generation with retrieved contexts.

## Features

- **Chat Completions**: Multi-turn conversation support with message history
- **Text Completions**: Simple prompt-based generation
- **RAG-augmented Generation**: Generate answers with retrieved context documents
- **Streaming Responses**: Real-time token streaming (async generator)
- **Model Management**: Automatic model pulling and availability checks
- **Japanese Support**: Full Japanese language support with Qwen models
- **GPU-accelerated**: Runs on NVIDIA GPUs via Ollama

## Model Details

- **Model**: Qwen3 (4B parameters)
- **Architecture**: Decoder-only transformer
- **Context Window**: 32768 tokens (configurable)
- **Languages**: Multilingual (optimized for Japanese, Chinese, English)
- **Quantization**: Default via Ollama (typically Q4_K_M)
- **Inference**: GPU-accelerated via Ollama

## Usage

### Basic Chat Completion

```python
from backend.services.llm import get_llm_service, Message

# Get service instance
service = await get_llm_service()

# Chat completion
response = await service.chat(
    messages=[
        Message(role="system", content="あなたは役立つAIアシスタントです。"),
        Message(role="user", content="日本の首都はどこですか？"),
    ]
)

print(response.content)
# Output: "日本の首都は東京です..."

print(f"Tokens: {response.total_tokens}")
print(f"Time: {response.processing_time_ms}ms")
```

### Simple Text Completion

```python
# Simple completion (single prompt)
response = await service.complete(
    prompt="機械学習について1行で説明してください。",
    system_prompt="あなたは専門家です。简潔に答えてください。",
)

print(response.content)
```

### RAG-augmented Generation

```python
from backend.services.llm import RAGContext

# Retrieved contexts from vector search
contexts = [
    RAGContext(
        text="機械学習は人工智能の一分野であり...",
        doc_id="doc_001",
        score=0.95,
        metadata={"source": "wikipedia"},
    ),
    RAGContext(
        text="深層学習は機械学習の一種で...",
        doc_id="doc_002",
        score=0.88,
    ),
]

# RAG generation
response = await service.generate_rag(
    query="機械学習と深層学習の違いは何ですか？",
    contexts=contexts,
)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents")
```

### Custom Generation Options

```python
from backend.services.llm import LLMOptions

options = LLMOptions(
    temperature=0.7,        # Sampling temperature (0.0-1.0)
    top_p=0.9,             # Nucleus sampling
    top_k=40,              # Top-k sampling
    num_ctx=32768,         # Context window size
    num_predict=2048,      # Max tokens to generate
    repeat_penalty=1.1,    # Repeat penalty (1.0 = no penalty)
)

response = await service.chat(
    messages=[Message(role="user", content="创意的な物語を書いてください。")],
    options=options,
)
```

### Streaming Responses

```python
# Streaming chat completion
async for chunk in service.chat_stream(
    messages=[Message(role="user", content="長いテキストを生成してください。")],
):
    print(chunk, end="", flush=True)
```

## Configuration

Environment variables in `backend/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:4b` | Default model name |
| `OLLAMA_NUM_CTX` | `32768` | Context window size (tokens) |
| `OLLAMA_TEMPERATURE` | `0.1` | Default temperature (0.0-1.0) |
| `OLLAMA_TOP_P` | `0.9` | Nucleus sampling parameter |
| `OLLAMA_TOP_K` | `40` | Top-k sampling parameter |
| `OLLAMA_NUM_PREDICT` | `2048` | Max tokens to generate |
| `OLLAMA_REPEAT_PENALTY` | `1.1` | Repeat penalty (1.0 = none) |

## Model Management

### Model Pulling

The model is automatically pulled on first use if not available:

```bash
# Manual pull from within Ollama container
docker exec ocr-rag-ollama-dev ollama pull qwen3:4b

# Or it will be pulled automatically on first API call
```

### Model Listing

```python
service = await get_llm_service()

# List available models
models = await service.client.list_models()
print(f"Available models: {models}")
# Output: ['qwen3:4b', 'qwen2.5:14b-instruct-q4_K_M', ...]
```

### Model Availability Check

```python
# Check if model is available
is_available = await service.client.check_model("qwen3:4b")
print(f"Model available: {is_available}")
```

## RAG Integration

The LLM Service is designed to work with the RAG pipeline:

```python
from backend.services.retrieval import get_retrieval_service
from backend.services.reranker import get_reranking_service

# 1. Retrieve relevant documents
retrieval_service = await get_retrieval_service()
retrieval_result = await retrieval_service.retrieve(query, top_k=20)

# 2. Rerank results
reranker_service = await get_reranking_service()
rerank_result = await reranker_service.rerank(
    query=query,
    documents=[
        {"text": c.text, "doc_id": c.chunk_id, "score": c.score}
        for c in retrieval_result.chunks
    ]
)

# 3. Generate answer with LLM
llm_service = await get_llm_service()
rag_response = await llm_service.generate_rag(
    query=query,
    contexts=[
        RAGContext(
            text=r.text,
            doc_id=r.doc_id,
            score=r.score,
        )
        for r in rerank_result.results
    ]
)

print(f"Answer: {rag_response.answer}")
```

## Architecture

```
LLM Service
├── models.py          # Pydantic models and exceptions
├── ollama_client.py   # Ollama HTTP client
├── service.py         # Main service class
└── __init__.py        # Package exports
```

### Component Overview

- **models.py**: Pydantic models for requests, responses, and configuration
- **ollama_client.py**: HTTP client for Ollama API (chat, generate, streaming)
- **service.py**: High-level service with singleton pattern

## Testing

Run manual tests:

```bash
# From within the app container
docker exec ocr-rag-app-dev python /app/tests/manual/test_llm.py

# Run specific test
docker exec ocr-rag-app-dev python /app/tests/manual/test_llm.py::test_chat_basic
```

Test cases:
- Basic chat completion
- Simple text completion
- Multi-turn conversation
- RAG-augmented generation
- Custom generation options
- Japanese language support
- Health check
- Model listing
- Performance benchmarks

## Performance

Expected performance on RTX 4090 with qwen3:4b:
- **First query**: ~500-1000ms (model loading + generation)
- **Subsequent queries**: ~50-200ms (100 tokens, varies by prompt)
- **Throughput**: ~20-50 tokens/second

### Optimization Tips

1. **Lower temperature** (0.0-0.3) for faster, deterministic responses
2. **Reduce num_predict** to limit generation length
3. **Use shorter prompts** when possible
4. **Enable streaming** for better UX on long generations
5. **Keep model loaded** (OLLAMA_KEEP_ALIVE setting)

## GPU Memory Usage

Typical GPU memory allocation for qwen3:4b:
- **Model loading**: ~3-4GB VRAM (Q4 quantized)
- **Inference**: ~4-5GB VRAM peak
- **Recommended**: 6GB+ VRAM available

Total system allocation (with other services):
- OCR (YomiToku): 40% VRAM (~9.6GB)
- Embedding (Sarashina): 30% VRAM (~7.2GB)
- LLM (Qwen): 20% VRAM (~4.8GB)
- Reranker: 10% VRAM (~2.4GB)

## Troubleshooting

**Issue**: Model not found
- Check Ollama is running: `docker logs ocr-rag-ollama-dev`
- Verify model is pulled: `docker exec ocr-rag-ollama-dev ollama list`
- Manually pull model: `docker exec ocr-rag-ollama-dev ollama pull qwen3:4b`

**Issue**: Slow generation
- Check GPU utilization: `nvidia-smi`
- Reduce `num_predict` parameter
- Lower `temperature` for faster inference
- Ensure Ollama has GPU access

**Issue**: Connection refused
- Verify Ollama container is running
- Check `OLLAMA_HOST` environment variable
- Ensure Docker network connectivity

**Issue**: Out of memory
- Reduce `OLLAMA_NUM_CTX` (context window)
- Use smaller model variant
- Reduce concurrent requests
- Free GPU memory: restart Ollama container

## API Reference

### LLMService

Main service class for LLM generation.

#### Methods

- **`chat(messages, options, model)`**: Chat completion with conversation history
- **`complete(prompt, system_prompt, options, model)`**: Simple text completion
- **`generate_rag(query, contexts, system_prompt, options, model)`**: RAG-augmented generation
- **`chat_stream(messages, options, model)`**: Streaming chat completion
- **`health_check()`**: Service health check
- **`shutdown()`**: Cleanup and free resources

### OllamaClient

Low-level HTTP client for Ollama API.

#### Methods

- **`chat(messages, model, options, stream)`**: Chat completion
- **`chat_stream(messages, model, options)`**: Streaming chat
- **`generate(prompt, model, options, system)`**: Text completion
- **`list_models(force_refresh)`**: List available models
- **`check_model(model)`**: Check model availability
- **`pull_model(model)`**: Pull model from registry
- **`health_check()`**: Ollama service health check

## Model Storage

Models are stored in Ollama's library directory:

| Component | Location | Storage Type |
|-----------|----------|--------------|
| Qwen LLM | `/root/.ollama/models/` | Docker volume (Ollama) |

The model is downloaded from Ollama's model registry and cached in the `ollama_models_dev` Docker volume.

## Future Enhancements

- [ ] Multi-modal support (vision)
- [ ] Function calling / tool use
- [ ] JSON mode output
- [ ] Structured output generation
- [ ] Batch inference
- [ ] Request queuing and throttling
- [ ] Response caching
