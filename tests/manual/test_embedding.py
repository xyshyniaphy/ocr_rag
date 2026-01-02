#!/usr/bin/env python3
"""
Test script for Embedding Service
Tests Sarashina model loading and embedding generation
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.logging import get_logger
from backend.services.embedding import (
    get_embedding_service,
    SarashinaEmbeddingModel,
    EmbeddingOptions,
)

logger = get_logger(__name__)


async def test_single_embedding():
    """Test embedding a single text"""
    print("\n" + "="*60)
    print("TEST 1: Single Text Embedding")
    print("="*60)

    service = await get_embedding_service()

    # Test text
    text = "これはテストテキストです。日本語の埋め込みをテストしています。"

    print(f"\nInput text: {text}")
    print(f"Model: {service.model.model_name}")
    print(f"Dimension: {service.model.dimension}")
    print(f"Max length: {service.model.max_length}")

    result = await service.embed_text(text, use_cache=False)

    print(f"\n✓ Embedding generated successfully!")
    print(f"  - Vector dimension: {len(result.embedding.vector)}")
    print(f"  - Token count: {result.token_count}")
    print(f"  - Processing time: {result.processing_time_ms}ms")
    print(f"  - Text hash: {result.embedding.text_hash}")
    print(f"  - First 5 values: {result.embedding.vector[:5]}")

    return True


async def test_batch_embedding():
    """Test embedding multiple texts"""
    print("\n" + "="*60)
    print("TEST 2: Batch Text Embedding")
    print("="*60)

    service = await get_embedding_service()

    # Test texts
    texts = [
        "最初のテキストです。",
        "二番目のテキストです。少し長めの文章を用意しています。",
        "人工知能は現代の技術において非常に重要な役割を果たしています。",
        "自然言語処理はAIの一分野です。",
    ]

    print(f"\nInput texts: {len(texts)} texts")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    result = await service.embed_texts(texts, use_cache=False)

    print(f"\n✓ Batch embedding generated successfully!")
    print(f"  - Total embeddings: {result.total_embeddings}")
    print(f"  - Dimension: {result.dimension}")
    print(f"  - Total tokens: {result.total_tokens}")
    print(f"  - Processing time: {result.processing_time_ms}ms")
    print(f"  - Avg time per text: {result.processing_time_ms // len(texts)}ms")

    return True


async def test_chunk_embedding():
    """Test embedding document chunks"""
    print("\n" + "="*60)
    print("TEST 3: Document Chunk Embedding")
    print("="*60)

    service = await get_embedding_service()

    # Test chunks
    chunks = {
        "chunk_001": "これは最初のチャンクのテキストです。文書の冒頭部分を含みます。",
        "chunk_002": "二番目のチャンクです。文書の内容を継続しています。",
        "chunk_003": "三番目のチャンクです。文書の最後の部分を含みます。",
    }

    document_id = "test_doc_001"

    print(f"\nDocument ID: {document_id}")
    print(f"Total chunks: {len(chunks)}")
    for chunk_id, text in chunks.items():
        print(f"  {chunk_id}: {text[:50]}...")

    result = await service.embed_chunks(chunks, document_id, use_cache=False)

    print(f"\n✓ Chunk embedding generated successfully!")
    print(f"  - Total chunks: {result.total_chunks}")
    print(f"  - Dimension: {result.dimension}")
    print(f"  - Processing time: {result.processing_time_ms}ms")

    return True


async def test_health_check():
    """Test health check"""
    print("\n" + "="*60)
    print("TEST 4: Health Check")
    print("="*60)

    service = await get_embedding_service()

    health = await service.health_check()

    print(f"\nHealth Status: {health['status']}")
    print(f"  - Model: {health['model']}")
    print(f"  - Dimension: {health['dimension']}")
    print(f"  - Max length: {health['max_length']}")
    print(f"  - Cache enabled: {health['cache_enabled']}")
    print(f"  - Is loaded: {health['is_loaded']}")

    if 'test_embedding_time_ms' in health:
        print(f"  - Test embedding time: {health['test_embedding_time_ms']}ms")
        print(f"  - Test token count: {health['test_token_count']}")
        print(f"  - Test dimension: {health['test_dimension']}")

    return health['status'] == 'healthy'


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("EMBEDDING SERVICE TEST SUITE")
    print("="*60)

    tests = [
        ("Single Text Embedding", test_single_embedding),
        ("Batch Text Embedding", test_batch_embedding),
        ("Document Chunk Embedding", test_chunk_embedding),
        ("Health Check", test_health_check),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result, None))
        except Exception as e:
            logger.error(f"Test '{name}' failed: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result, _ in results if result)
    failed = len(results) - passed

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
