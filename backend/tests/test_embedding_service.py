#!/usr/bin/env python3
"""
Embedding Service Test Script
Tests the Sarashina embedding service
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging import get_logger
from backend.services.embedding import (
    embed_text,
    embed_texts,
    EmbeddingService,
    EmbeddingOptions,
)
from backend.services.processing.chunking import ChunkingService, ChunkingOptions
from backend.services.ocr import ocr_pdf

logger = get_logger(__name__)


async def test_single_embedding():
    """Test embedding a single text"""

    print("=" * 60)
    print("Single Text Embedding Test")
    print("=" * 60)

    # Sample Japanese texts
    test_texts = [
        "日本語のテキスト埋め込みテストです。",
        "This is an English text for embedding test.",
        "機械学習と自然言語処理は密接に関連しています。",
        "第1条　この法律は、電磁的方法による契約の締結等に関する規律を整備するものとする。",
    ]

    print("\n" + "-" * 40)
    print("Test 1: Single Text Embedding")
    print("-" * 40)

    for i, text in enumerate(test_texts, 1):
        result = await embed_text(text)

        print(f"\nText {i}:")
        print(f"  Input: {text[:50]}...")
        print(f"  Dimension: {result.embedding.dimension}")
        print(f"  Vector (first 5): {result.embedding.vector[:5]}")
        print(f"  Tokens: {result.token_count}")
        print(f"  Time: {result.processing_time_ms}ms")
        print(f"  Hash: {result.embedding.text_hash}")


async def test_batch_embedding():
    """Test embedding multiple texts in batch"""

    print("\n" + "=" * 60)
    print("Batch Embedding Test")
    print("=" * 60)

    # Generate test texts
    test_texts = [
        "人工知能はコンピュータサイエンスの一分野です。",
        "ディープラーニングは機械学習の手法の一つです。",
        "自然言語処理はテキストデータを扱います。",
        "コンピュータビジョンは画像データを扱います。",
        "強化学習はエージェントが環境と相互作用しながら学習します。",
        "転移学習は既存の知識を新しいタスクに応用します。",
        "データマイニングは大量のデータからパターンを発見します。",
        "クラスタリングはデータをグループ化する手法です。",
        "回帰分析は連続値を予測する手法です。",
        "分類問題はデータをカテゴリに割り当てます。",
    ]

    print("\n" + "-" * 40)
    print("Test 2: Batch Embedding (10 texts)")
    print("-" * 40)

    result = await embed_texts(test_texts)

    print(f"\nGenerated {result.total_embeddings} embeddings:")
    print(f"  Dimension: {result.dimension}")
    print(f"  Model: {result.model}")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Processing time: {result.processing_time_ms}ms")
    print(f"  Avg time per text: {result.processing_time_ms / len(test_texts):.1f}ms")
    print(f"  Batch size: {result.options.batch_size}")

    if result.warnings:
        print(f"\nWarnings: {len(result.warnings)}")
        for warning in result.warnings[:3]:
            print(f"  - {warning}")


async def test_embedding_similarity():
    """Test embedding semantic similarity"""

    print("\n" + "=" * 60)
    print("Semantic Similarity Test")
    print("=" * 60)

    print("\n" + "-" * 40)
    print("Test 3: Cosine Similarity Between Embeddings")
    print("-" * 40)

    import numpy as np

    # Similar texts (same topic)
    text1 = "日本の首都は東京です。"
    text2 = "東京は日本の首都です。"

    # Different topic
    text3 = "りんごは赤い果物です。"

    emb1 = await embed_text(text1)
    emb2 = await embed_text(text2)
    emb3 = await embed_text(text3)

    def cosine_similarity(a, b):
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    sim_12 = cosine_similarity(emb1.embedding.vector, emb2.embedding.vector)
    sim_13 = cosine_similarity(emb1.embedding.vector, emb3.embedding.vector)

    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")

    print(f"\nSimilarity (Text 1 ↔ Text 2): {sim_12:.4f} (similar topic)")
    print(f"Similarity (Text 1 ↔ Text 3): {sim_13:.4f} (different topic)")

    if sim_12 > sim_13:
        print("\n✓ Semantic similarity working correctly!")
    else:
        print("\n✗ Warning: Similarity scores unexpected")


async def test_chunk_embedding():
    """Test embedding document chunks"""

    print("\n" + "=" * 60)
    print("Chunk Embedding Test")
    print("=" * 60)

    # Create sample chunks
    sample_texts = [
        "第1章 総則\n第1条 この法律は、電子署名に関する規律を整備することを目的とする。",
        "第2条 この法律において「電子署名」とは、電磁的記録に記録することができる "
        "情報について行われる署名をいう。",
        "第2章 電子署名の効力\n第3条 電子署名は、押印と同一の効力を有する。",
    ]

    # Create mock TextChunk objects
    from backend.services.processing.chunking.models import TextChunk, ChunkMetadata

    chunks = []
    for i, text in enumerate(sample_texts):
        chunk = TextChunk(
            chunk_id=f"test-chunk-{i}",
            text=text,
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=i,
                document_id="test-doc-001",
                chunk_type="text",
            ),
        )
        chunks.append(chunk)

    print("\n" + "-" * 40)
    print("Test 4: Document Chunk Embedding")
    print("-" * 40)

    result = await EmbeddingService.embed_chunks(chunks)

    print(f"\nEmbedded {result.total_chunks} chunks:")
    print(f"  Document ID: {result.document_id}")
    print(f"  Dimension: {result.dimension}")
    print(f"  Model: {result.model}")
    print(f"  Processing time: {result.processing_time_ms}ms")

    print("\nChunk embeddings:")
    for chunk_id, embedding in result.chunk_embeddings.items():
        print(f"  {chunk_id}: dimension={embedding.dimension}, "
              f"hash={embedding.text_hash}")


async def test_gpu_management():
    """Test GPU memory management"""

    print("\n" + "=" * 60)
    print("GPU Management Test")
    print("=" * 60)

    print("\n" + "-" * 40)
    print("Test 5: Model Load/Unload")
    print("-" * 40)

    try:
        import torch

        # Check GPU availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            print(f"\nGPU Info:")
            print(f"  Available: Yes")
            print(f"  Device count: {device_count}")
            print(f"  Current device: {current_device}")
            print(f"  Device name: {device_name}")

            # Get memory before loading
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1024**2
            print(f"\nMemory before load: {mem_before:.1f} MB")

            # Load model
            model = await EmbeddingService.get_model("sarashina")
            mem_loaded = torch.cuda.memory_allocated() / 1024**2
            print(f"Memory after load: {mem_loaded:.1f} MB")
            print(f"Model size: {mem_loaded - mem_before:.1f} MB")

            # Unload model
            await EmbeddingService.unload_model("sarashina")
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / 1024**2
            print(f"Memory after unload: {mem_after:.1f} MB")

            if mem_after < mem_loaded:
                print("\n✓ GPU memory freed successfully!")
            else:
                print("\n⚠ Warning: GPU memory may not be fully freed")

        else:
            print("\nGPU Info:")
            print("  Available: No")
            print("  Running on CPU")

    except ImportError:
        print("\nPyTorch not available, skipping GPU test")


async def test_batch_sizes():
    """Test different batch sizes"""

    print("\n" + "=" * 60)
    print("Batch Size Optimization Test")
    print("=" * 60)

    # Generate 64 test texts
    test_texts = [f"テストテキスト番号{i}です。" for i in range(64)]

    batch_sizes = [1, 8, 16, 32, 64]

    print("\n" + "-" * 40)
    print("Test 6: Batch Size Performance")
    print("-" * 40)

    print(f"\nEmbedding {len(test_texts)} texts with different batch sizes:\n")

    for batch_size in batch_sizes:
        options = EmbeddingOptions(batch_size=batch_size)
        result = await embed_texts(test_texts, options=options)

        avg_time = result.processing_time_ms / len(test_texts)
        print(f"  Batch size {batch_size:2d}: "
              f"{result.processing_time_ms:4d}ms total, "
              f"{avg_time:5.1f}ms avg/text")


async def main():
    """Main test runner"""

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Embedding Test Suite" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")

    success = True

    # Test 1: Single embedding
    try:
        await test_single_embedding()
    except Exception as e:
        print(f"\nERROR in single embedding test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: Batch embedding
    try:
        await test_batch_embedding()
    except Exception as e:
        print(f"\nERROR in batch embedding test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 3: Semantic similarity
    try:
        await test_embedding_similarity()
    except Exception as e:
        print(f"\nERROR in similarity test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 4: Chunk embedding
    try:
        await test_chunk_embedding()
    except Exception as e:
        print(f"\nERROR in chunk embedding test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 5: GPU management
    try:
        await test_gpu_management()
    except Exception as e:
        print(f"\nERROR in GPU management test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 6: Batch sizes
    try:
        await test_batch_sizes()
    except Exception as e:
        print(f"\nERROR in batch size test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("\n" + "🎉" + " All tests passed! " + "🎉")
        sys.exit(0)
    else:
        print("\n" + "❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
