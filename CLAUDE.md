# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Japanese OCR RAG System** - a production-grade Retrieval-Augmented Generation system optimized for Japanese PDF document processing. The system is privacy-first (air-gapped deployment supported) and designed for legal, financial, academic, and enterprise document analysis.

## High-Level Architecture

The system consists of **3 parallel processing pipelines**:

1. **OCR Pipeline**: Extracts text from Japanese PDFs using YomiToku (primary) with PaddleOCR-VL fallback
2. **Embedding Pipeline**: Generates 768D vectors using Sarashina-Embedding-v1-1B with Llama-3.2-NV-RerankQA-1B-v2 reranking
3. **LLM Generation Pipeline**: Qwen2.5-14B (via Ollama) generates Japanese responses with citations

**Storage Layer**:
- Milvus 2.4+ (Vector DB, 768D embeddings, IVF_FLAT index)
- PostgreSQL 16+ (Metadata DB: documents, chunks, queries)
- MinIO (Object storage for raw PDFs, OCR outputs, thumbnails)

**Application Layer**:
- LangChain RAG Pipeline (orchestration)
- FastAPI (REST API)
- Streamlit (Admin UI)

## Data Flow

**Document Ingestion**: `PDF Upload → OCR → Markdown → Chunking → Embedding → Milvus → Metadata Index`

**Query Processing**: `User Query → Hybrid Search (Vector + BM25) → Reranking (Top 20→5) → Context Assembly → LLM → Citation Injection`

## Technology Stack

| Component | Technology |
|-----------|------------|
| OCR | YomiToku (primary), PaddleOCR-VL (fallback) |
| Embedding | sbintuitions/sarashina-embedding-v1-1b (768D) |
| Reranker | nvidia/llama-3.2-nv-rerankqa-1b-v2 |
| LLM | Qwen/Qwen2.5-14B-Instruct (Ollama) |
| Vector DB | Milvus 2.4+ (IVF_FLAT, nlist=1024, nprobe=128) |
| Metadata DB | PostgreSQL 16+ |
| Object Storage | MinIO |
| Backend | FastAPI + LangChain |
| Frontend | Streamlit |
| Container | Docker Compose / Kubernetes |

## Key Configuration Parameters

**Chunking** (LangChain RecursiveCharacterTextSplitter):
- chunk_size: 512 characters
- chunk_overlap: 50
- Japanese separators: `\n\n`, `\n`, `。`, `！`, `？`, `；`, `、`

**Japanese Text Normalization**:
- Unicode NFKC normalization
- Half-width → Full-width Katakana conversion
- Space insertion between Kanji and Latin characters

**OCR Quality Assurance**:
- Pre-processing: Deskew, noise reduction, CLAHE, Otsu binarization
- Post-processing: Character normalization, table structure validation
- Confidence threshold: 0.85 (retry), 0.80 (fallback to PaddleOCR)

**Milvus Search**:
- Initial retrieval: Top 20 (IVF_FLAT, nprobe=128)
- Hybrid search: 70% semantic + 30% BM25 keyword
- Reranker: Top 20 → Top 5 (score threshold 0.65)

## Hardware Requirements

**Development**: RTX 3060 Ti (8GB VRAM), 32GB RAM, Ubuntu 22.04 LTS

**Production (Small)**: RTX 4090 (24GB VRAM), 64GB RAM, 2TB NVMe SSD

**Production (Enterprise)**: Multi-node with dedicated OCR/Embedding/LLM/Storage nodes

## API Endpoints

- `POST /api/v1/documents/upload` - Upload PDF (max 50MB)
- `POST /api/v1/query` - Query RAG system
- `GET /api/v1/documents/{document_id}/status` - Get processing status
- `GET /api/v1/documents/search` - Search documents

## Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Query (GPU) | <2s | 5s (95th percentile) |
| OCR Processing | <10s/page | 30s/page |
| Embedding | <50ms/chunk | 200ms/chunk |
| Vector Search | <100ms | 500ms |

## Privacy & Security

- All processing local (no external API calls to OpenAI/Claude)
- AES-256-GCM encryption at rest
- TLS 1.3 in transit
- OAuth 2.0 + JWT authentication
- RBAC (Admin, Power User, User, Viewer)
- Japanese APPI compliant (PII detection, right to deletion)

## Development Notes

### Japanese Language Handling
- Primary language is Japanese (日本語)
- Use Japanese-specific separators for text chunking
- Apply Unicode NFKC normalization before processing
- Preserve vertical text (縦書き) layout in OCR

### OCR Fallback Logic
1. Try YomiToku (Japanese-specialized)
2. If confidence <0.80 or multi-language >20%: fallback to PaddleOCR-VL
3. Log low-confidence regions for manual review

### Vector Index Management
- Milvus collection schema includes: chunk_id, embedding (768D), text_content, document_id, page_number, chunk_index, metadata (JSON)
- Enable dynamic fields for custom metadata
- Use IVF_FLAT index for balance between speed and accuracy

### LLM Prompt Template (Japanese)
```
あなたは正確で信頼できる日本語の文書分析AIアシスタントです。

【重要な制約】
1. 以下の【参考文献】に記載された情報のみを使用して回答してください
2. 文献に情報が無い場合は「提供された文書には該当情報が見つかりませんでした」と明言してください
3. 推測や一般知識での補完は一切行わないでください
4. 情報源を必ず明記してください
```

### Quality Metrics
- OCR Accuracy: >=95% on tables, >=99% on vertical text
- Retrieval: Recall@5 >=85%, NDCG@5 >=0.80
- Generation: Factual accuracy >=95%, citation accuracy 100%
