"""
Reranker Service Tests
Manual tests for reranking functionality
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.services.reranker import get_reranking_service
from backend.services.reranker.models import RerankerOptions

logger = get_logger(__name__)


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "text": "機械学習は人工智能の一分野であり、コンピュータがデータから学習するアルゴリズムを研究する。"
                "深層学習は機械学習の手法の一つで、多層ニューラルネットワークを使用する。",
        "doc_id": "doc_001",
        "metadata": {"category": "ml"},
    },
    {
        "text": "自然言語処理（NLP）は、人間の言語をコンピュータに理解させ、処理させる技術である。"
                "近年、トランスフォーマーモデルの登場により、大きな進歩を遂げた。",
        "doc_id": "doc_002",
        "metadata": {"category": "nlp"},
    },
    {
        "text": "ベクトルデータベースは、高次元ベクトルの高速な類似性検索を目的としたデータベースである。"
                "Milvusはオープンソースのベクトルデータベースで、大規模なベクトル検索に適している。",
        "doc_id": "doc_003",
        "metadata": {"category": "database"},
    },
    {
        "text": "RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法である。"
                "関連する文書を検索し、それをコンテキストとしてLLMに渡すことで、より正確な回答を生成する。",
        "doc_id": "doc_004",
        "metadata": {"category": "rag"},
    },
    {
        "text": "日本語のOCR処理は、漢字の多様性とひらがな・カタカナの混在により、"
                "英語よりも複雑である。YomiTokuは日本語に最適化されたOCRエンジンである。",
        "doc_id": "doc_005",
        "metadata": {"category": "ocr"},
    },
    {
        "text": "量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータである。"
                "特定の問題に対して、古典コンピュータよりも指数関数的に高速な計算が可能である。",
        "doc_id": "doc_006",
        "metadata": {"category": "quantum"},
    },
    {
        "text": "ブロックチェーンは、分散型台帳技術の一種であり、ビットコインなどの仮想通貨で"
                "使用されている。改ざんが困難なデータ構造を持ち、透明性と信頼性を提供する。",
        "doc_id": "doc_007",
        "metadata": {"category": "blockchain"},
    },
    {
        "text": "ReactはFacebookが開発したJavaScriptライブラリであり、"
                "ユーザーインターフェースの構築に使用される。コンポーネントベースのアプローチを採用している。",
        "doc_id": "doc_008",
        "metadata": {"category": "frontend"},
    },
    {
        "text": "Dockerはコンテナ仮想化技術を使用してアプリケーションを"
                "パッケージ化・実行するプラットフォームである。環境に依存しないデプロイが可能になる。",
        "doc_id": "doc_009",
        "metadata": {"category": "devops"},
    },
    {
        "text": "気候変動は、地球規模の気候パターンの長期的な変化を指す。"
                "人間活動による温室効果ガスの排出が主要原因であり、気温上昇や極端気象の増加をもたらしている。",
        "doc_id": "doc_010",
        "metadata": {"category": "climate"},
    },
]


async def test_reranking_basic():
    """Test basic reranking functionality"""
    logger.info("=== Testing Basic Reranking ===")

    try:
        service = get_reranking_service()
        await service.initialize()

        # Test query
        query = "機械学習と深層学習の違いは何ですか？"  # What's the difference between ML and DL?

        start = time.time()
        result = await service.rerank(query, SAMPLE_DOCUMENTS)
        elapsed = (time.time() - start) * 1000

        logger.info(f"Query: {query}")
        logger.info(f"Processing time: {elapsed:.2f}ms")
        logger.info(f"Total input: {result.total_input}")
        logger.info(f"Total output: {result.total_output}")
        logger.info(f"Threshold applied: {result.threshold_applied}")

        logger.info("\nTop 5 reranked results:")
        for i, r in enumerate(result.results[:5], 1):
            logger.info(
                f"  {i}. [Rank: {r.rank}, Score: {r.score:.3f}] "
                f"{r.text[:60]}... (doc_id: {r.doc_id})"
            )

        # Verify the most relevant document is ranked first
        if result.results:
            best_doc = result.results[0]
            logger.info(f"\nBest matched document: {best_doc.doc_id} (score: {best_doc.score:.3f})")

        return True

    except Exception as e:
        logger.error(f"✗ Basic reranking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking_with_options():
    """Test reranking with custom options"""
    logger.info("\n=== Testing Reranking with Custom Options ===")

    try:
        service = get_reranking_service()

        # Custom options - lower threshold, more results
        options = RerankerOptions(
            top_k_input=20,
            top_k_output=10,
            threshold=0.3,  # Lower threshold
            batch_size=16,
        )

        query = "ベクトルデータベースの特徴を教えて"  # Tell me about vector databases

        start = time.time()
        result = await service.rerank(query, SAMPLE_DOCUMENTS, options=options)
        elapsed = (time.time() - start) * 1000

        logger.info(f"Query: {query}")
        logger.info(f"Options: top_k_input={options.top_k_input}, "
                   f"top_k_output={options.top_k_output}, threshold={options.threshold}")
        logger.info(f"Results: {result.total_output} documents (time: {elapsed:.2f}ms)")

        for i, r in enumerate(result.results, 1):
            logger.info(f"  {i}. [{r.score:.3f}] {r.text[:50]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Reranking with options failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking_simple_api():
    """Test simple reranking API"""
    logger.info("\n=== Testing Simple Reranking API ===")

    try:
        service = get_reranking_service()

        query = "日本語のOCR処理について"  # About Japanese OCR processing
        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]

        start = time.time()
        results = await service.rerank_simple(query, texts, top_k=3)
        elapsed = (time.time() - start) * 1000

        logger.info(f"Query: {query}")
        logger.info(f"Results: {len(results)} documents (time: {elapsed:.2f}ms)")

        for idx, text, score in results:
            logger.info(f"  [{idx}] {score:.3f}: {text[:50]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Simple API failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking_threshold_filtering():
    """Test threshold filtering"""
    logger.info("\n=== Testing Threshold Filtering ===")

    try:
        service = get_reranking_service()

        # Query about blockchain (should match doc_007)
        query = "ブロックチェーンとは何ですか？"  # What is blockchain?

        # Test with different thresholds
        thresholds = [0.2, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            options = RerankerOptions(threshold=threshold)
            result = await service.rerank(query, SAMPLE_DOCUMENTS, options=options)

            logger.info(
                f"Threshold {threshold}: {result.total_output}/{result.total_input} documents passed"
            )

            if result.results:
                scores = [r.score for r in result.results]
                logger.info(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")

        return True

    except Exception as e:
        logger.error(f"✗ Threshold filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking_multilingual():
    """Test reranking with mixed Japanese/English queries"""
    logger.info("\n=== Testing Multilingual Reranking ===")

    try:
        service = get_reranking_service()

        # Test cases
        test_cases = [
            ("What is machine learning?", "English query"),
            ("ベクトル検索とは", "Japanese query"),
            ("RAGシステムの仕組み", "Japanese technical query"),
            ("量子コンピュータの応用", "Japanese query about quantum computing"),
        ]

        for query, description in test_cases:
            result = await service.rerank(query, SAMPLE_DOCUMENTS)

            logger.info(f"\n{description}: {query}")
            logger.info(f"  Results: {result.total_output} documents")

            if result.results:
                top_3 = result.results[:3]
                logger.info(f"  Top 3:")
                for i, r in enumerate(top_3, 1):
                    logger.info(f"    {i}. [{r.score:.3f}] {r.text[:40]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Multilingual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_check():
    """Test reranker health check"""
    logger.info("\n=== Testing Health Check ===")

    try:
        service = get_reranking_service()

        health = await service.health_check()
        logger.info(f"Health status: {health['status']}")
        logger.info(f"Model: {health['model']}")
        logger.info(f"Is loaded: {health['is_loaded']}")
        logger.info(f"Cache enabled: {health['cache_enabled']}")

        if "test_rerank_time_ms" in health:
            logger.info(f"Test rerank time: {health['test_rerank_time_ms']:.2f}ms")
            logger.info(f"Test num results: {health['test_num_results']}")

        return health["status"] == "healthy"

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


async def test_performance():
    """Test reranking performance"""
    logger.info("\n=== Testing Performance ===")

    try:
        service = get_reranking_service()

        # Test with different batch sizes
        batch_sizes = [5, 10, 20, 50]
        query = "機械学習の概要"  # Overview of machine learning

        for batch_size in batch_sizes:
            docs = SAMPLE_DOCUMENTS * (batch_size // len(SAMPLE_DOCUMENTS) + 1)
            docs = docs[:batch_size]

            start = time.time()
            result = await service.rerank(query, docs)
            elapsed = (time.time() - start) * 1000

            logger.info(
                f"Batch size {batch_size}: {elapsed:.2f}ms "
                f"({elapsed/batch_size:.2f}ms per document)"
            )

        return True

    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    logger.info("=== Reranker Service Tests ===\n")

    # Test 1: Basic reranking
    await test_reranking_basic()

    # Test 2: Custom options
    await test_reranking_with_options()

    # Test 3: Simple API
    await test_reranking_simple_api()

    # Test 4: Threshold filtering
    await test_reranking_threshold_filtering()

    # Test 5: Multilingual support
    await test_reranking_multilingual()

    # Test 6: Health check
    await test_health_check()

    # Test 7: Performance
    await test_performance()

    logger.info("\n=== All Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
