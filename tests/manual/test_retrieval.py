"""
Retrieval Service Tests
Manual tests for retrieval functionality
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.db.vector import (
    delete_by_document,
    get_collection_stats,
    health_check as milvus_health_check,
    init_milvus,
    insert_chunks,
)
from backend.services.embedding import get_embedding_service
from backend.services.retrieval import (
    get_retrieval_service,
    retrieve,
    retrieval_health_check,
)
from backend.services.retrieval.models import RetrievalOptions

logger = get_logger(__name__)


# Sample document chunks for testing
SAMPLE_CHUNKS = [
    {
        "chunk_id": "test_chunk_001",
        "text": "機械学習は人工智能の一分野であり、コンピュータがデータから学習するアルゴリズムを研究する。"
                "深層学習は機械学習の手法の一つで、多層ニューラルネットワークを使用する。",
        "document_id": "test_doc_001",
        "page_number": 1,
        "chunk_index": 0,
    },
    {
        "chunk_id": "test_chunk_002",
        "text": "自然言語処理（NLP）は、人間の言語をコンピュータに理解させ、処理させる技術である。"
                "近年、トランスフォーマーモデルの登場により、大きな進歩を遂げた。",
        "document_id": "test_doc_001",
        "page_number": 2,
        "chunk_index": 1,
    },
    {
        "chunk_id": "test_chunk_003",
        "text": "ベクトルデータベースは、高次元ベクトルの高速な類似性検索を目的としたデータベースである。"
                "Milvusはオープンソースのベクトルデータベースで、大規模なベクトル検索に適している。",
        "document_id": "test_doc_002",
        "page_number": 1,
        "chunk_index": 0,
    },
    {
        "chunk_id": "test_chunk_004",
        "text": "RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法である。"
                "関連する文書を検索し、それをコンテキストとしてLLMに渡すことで、より正確な回答を生成する。",
        "document_id": "test_doc_002",
        "page_number": 2,
        "chunk_index": 1,
    },
    {
        "chunk_id": "test_chunk_005",
        "text": "日本語のOCR処理は、漢字の多様性とひらがな・カタカナの混在により、"
                "英語よりも複雑である。YomiTokuは日本語に最適化されたOCRエンジンである。",
        "document_id": "test_doc_003",
        "page_number": 1,
        "chunk_index": 0,
    },
]


async def setup_test_data():
    """Setup test data in Milvus"""
    logger.info("Setting up test data...")

    # Initialize embedding service
    from backend.services.embedding import embed_texts
    logger.info("Embedding service initialized")

    # Generate embeddings for all chunks
    texts = [chunk["text"] for chunk in SAMPLE_CHUNKS]
    embedding_result = await embed_texts(texts)
    # Extract vector from each TextEmbedding.embedding.vector
    embeddings = [e.embedding.vector for e in embedding_result.embeddings]

    # Prepare data for insertion
    chunk_ids = [chunk["chunk_id"] for chunk in SAMPLE_CHUNKS]
    document_ids = [chunk["document_id"] for chunk in SAMPLE_CHUNKS]
    page_numbers = [chunk["page_number"] for chunk in SAMPLE_CHUNKS]
    chunk_indices = [chunk["chunk_index"] for chunk in SAMPLE_CHUNKS]
    metadata = [{"test": True} for _ in SAMPLE_CHUNKS]

    # Delete existing test data
    for doc_id in set(document_ids):
        try:
            await delete_by_document(doc_id)
        except Exception:
            pass

    # Insert chunks
    count = await insert_chunks(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        texts=texts,
        document_ids=document_ids,
        page_numbers=page_numbers,
        chunk_indices=chunk_indices,
        metadata=metadata,
    )

    logger.info(f"Inserted {count} test chunks")

    # Verify collection stats
    stats = await get_collection_stats()
    logger.info(f"Collection stats: {stats}")


async def test_milvus_connection():
    """Test Milvus connection"""
    logger.info("=== Testing Milvus Connection ===")

    try:
        # First connect to Milvus
        await init_milvus()
        logger.info("✓ Milvus initialized")

        from pymilvus import utility, Collection

        # Check if collection exists and has correct dimension
        if utility.has_collection("document_chunks"):
            collection = Collection("document_chunks")
            collection.load()

            # Check dimension
            schema = collection.schema
            needs_recreate = False
            for field in schema.fields:
                if field.name == "embedding":
                    dim = field.params.get("dim")
                    if dim != 1792:
                        logger.warning(f"Collection has wrong dimension ({dim}), dropping and recreating with 1792...")
                        utility.drop_collection("document_chunks")
                        logger.info("Old collection dropped")
                        needs_recreate = True
                    break

            # Recreate collection if needed
            if needs_recreate:
                from backend.db.vector.client import _create_collection
                await _create_collection()

        # Health check
        healthy = await milvus_health_check()
        logger.info(f"{'✓' if healthy else '✗'} Milvus health check: {healthy}")

        # Get stats
        stats = await get_collection_stats()
        logger.info(f"✓ Collection stats: {stats}")

        return healthy

    except Exception as e:
        logger.error(f"✗ Milvus connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_search():
    """Test vector search"""
    logger.info("\n=== Testing Vector Search ===")

    try:
        service = get_retrieval_service()
        await service.initialize()

        # Test queries
        queries = [
            "機械学習とは何ですか？",  # What is machine learning?
            "ベクトルデータベースの特徴",  # Vector database features
            "日本語のOCR処理について",  # Japanese OCR processing
        ]

        for query in queries:
            start = time.time()
            result = await service.retrieve(query, method="vector")
            elapsed = (time.time() - start) * 1000

            logger.info(f"\nQuery: {query}")
            logger.info(f"Results: {len(result.chunks)} chunks")
            logger.info(f"Time: {elapsed:.2f}ms")

            for i, chunk in enumerate(result.chunks[:3], 1):
                logger.info(f"  {i}. [{chunk.score:.3f}] {chunk.text[:60]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_keyword_search():
    """Test keyword search"""
    logger.info("\n=== Testing Keyword Search ===")

    try:
        service = get_retrieval_service()

        # Test queries
        queries = [
            "機械学習",
            "ベクトルデータベース",
            "OCR",
        ]

        for query in queries:
            start = time.time()
            result = await service.retrieve(query, method="keyword")
            elapsed = (time.time() - start) * 1000

            logger.info(f"\nQuery: {query}")
            logger.info(f"Results: {len(result.chunks)} chunks")
            logger.info(f"Time: {elapsed:.2f}ms")

            for i, chunk in enumerate(result.chunks[:3], 1):
                logger.info(f"  {i}. [{chunk.score:.3f}] {chunk.text[:60]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Keyword search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_search():
    """Test hybrid search"""
    logger.info("\n=== Testing Hybrid Search ===")

    try:
        service = get_retrieval_service()

        # Test queries
        queries = [
            "機械学習と深層学習の違いは？",  # Difference between ML and DL?
            "Milvusの特徴を教えて",  # Tell me about Milvus features
            "日本語テキストの処理方法",  # How to process Japanese text
        ]

        for query in queries:
            start = time.time()
            result = await service.retrieve(query, method="hybrid")
            elapsed = (time.time() - start) * 1000

            logger.info(f"\nQuery: {query}")
            logger.info(f"Results: {len(result.chunks)} chunks")
            logger.info(f"Time: {elapsed:.2f}ms")

            if result.metadata:
                logger.info(f"  Metadata: {result.metadata}")

            for i, chunk in enumerate(result.chunks[:3], 1):
                logger.info(f"  {i}. [{chunk.score:.3f}] {chunk.text[:60]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_retrieval_with_options():
    """Test retrieval with custom options"""
    logger.info("\n=== Testing Retrieval with Options ===")

    try:
        service = get_retrieval_service()

        # Test with custom options
        options = RetrievalOptions(
            top_k=5,
            min_score=0.1,
            document_ids=["test_doc_001"],
        )

        result = await service.retrieve(
            "機械学習",
            method="hybrid",
            options=options,
        )

        logger.info(f"Query: 機械学習 (filtered by document)")
        logger.info(f"Results: {len(result.chunks)} chunks")
        logger.info(f"Filter: document_id=test_doc_001")

        for i, chunk in enumerate(result.chunks, 1):
            logger.info(f"  {i}. [{chunk.score:.3f}] {chunk.text[:60]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Retrieval with options failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_check():
    """Test retrieval service health check"""
    logger.info("\n=== Testing Health Check ===")

    try:
        health = await retrieval_health_check()
        logger.info(f"Health status: {health}")

        for key, value in health.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


async def test_convenience_function():
    """Test convenience function"""
    logger.info("\n=== Testing Convenience Function ===")

    try:
        result = await retrieve("RAGについて教えて")  # Tell me about RAG
        logger.info(f"Query: RAGについて教えて")
        logger.info(f"Results: {len(result.chunks)} chunks")

        for i, chunk in enumerate(result.chunks[:2], 1):
            logger.info(f"  {i}. [{chunk.score:.3f}] {chunk.text[:60]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Convenience function failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    logger.info("=== Retrieval Service Tests ===\n")

    # Initialize database
    logger.info("Initializing PostgreSQL...")
    from backend.db.session import init_db
    await init_db()
    logger.info("✓ PostgreSQL initialized\n")

    # Test 1: Milvus connection
    if not await test_milvus_connection():
        logger.error("Failed to connect to Milvus, aborting tests")
        return

    # Setup test data
    await setup_test_data()

    # Test 2: Vector search
    await test_vector_search()

    # Test 3: Keyword search
    await test_keyword_search()

    # Test 4: Hybrid search
    await test_hybrid_search()

    # Test 5: Retrieval with options
    await test_retrieval_with_options()

    # Test 6: Health check
    await test_health_check()

    # Test 7: Convenience function
    await test_convenience_function()

    logger.info("\n=== All Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
