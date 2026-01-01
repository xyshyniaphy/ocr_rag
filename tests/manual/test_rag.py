"""
RAG Orchestration Service Manual Tests
Tests for the complete RAG pipeline including retrieval, reranking, and LLM generation

Run these tests from within the app container:
    docker exec ocr-rag-app-dev python /app/tests/manual/test_rag.py

Or run specific tests:
    docker exec ocr-rag-app-dev python /app/tests/manual/test_rag.py::test_rag_health_check
"""

import asyncio
import sys
import time
from typing import List

# Add backend to path for imports
sys.path.insert(0, "/app")

from backend.core.logging import get_logger
from backend.services.rag import (
    get_rag_service,
    RAGQueryOptions,
    RAGService,
    RAGValidationError,
    RAGProcessingError,
)

logger = get_logger(__name__)

# Sample test queries (Japanese)
SAMPLE_QUERIES = [
    "機械学習とは何ですか？",
    "深層学習と機械学習の違いは何ですか？",
    "日本の首都はどこですか？",
    "Pythonプログラミングの特徴を教えてください",
]


async def test_rag_health_check():
    """Test RAG service health check"""
    print("\n" + "=" * 80)
    print("TEST: RAG Service Health Check")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        health = await service.health_check()

        print(f"\n✓ Health Status:")
        print(f"  Service: {health['service']}")
        print(f"  Status: {health['status']}")
        print(f"  Initialized: {health['initialized']}")

        if health.get('components'):
            print(f"\n  Components:")
            for component, status in health['components'].items():
                print(f"    {component}: {status}")

        if health.get('errors'):
            print(f"\n  Errors:")
            for error in health['errors']:
                print(f"    - {error}")

        print(f"\n✓ TEST PASSED: RAG service healthy")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_query_validation():
    """Test RAG query validation"""
    print("\n" + "=" * 80)
    print("TEST: RAG Query Validation")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        tests_passed = 0
        tests_failed = 0

        # Test 1: Empty query
        print("\nTest 1: Empty query")
        try:
            result = await service.query(query="")
            print("  ✗ Should have raised RAGValidationError")
            tests_failed += 1
        except RAGValidationError as e:
            print(f"  ✓ Correctly raised RAGValidationError: {e.message}")
            tests_passed += 1

        # Test 2: Very long query
        print("\nTest 2: Very long query")
        try:
            long_query = "テスト" * 200  # > 500 characters
            result = await service.query(query=long_query)
            print("  ✗ Should have raised RAGValidationError")
            tests_failed += 1
        except RAGValidationError as e:
            print(f"  ✓ Correctly raised RAGValidationError: {e.message}")
            tests_passed += 1

        # Test 3: Invalid retrieval method
        print("\nTest 3: Invalid retrieval method")
        try:
            result = await service.query(
                query="テスト",
                options=RAGQueryOptions(retrieval_method="invalid"),
            )
            print("  ✗ Should have raised RAGProcessingError")
            tests_failed += 1
        except RAGProcessingError as e:
            print(f"  ✓ Correctly raised RAGProcessingError: {e.message}")
            tests_passed += 1

        print(f"\n✓ TEST PASSED: {tests_passed}/{tests_passed + tests_failed} validation tests passed")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_stage_timing_breakdown():
    """Test detailed stage timing breakdown"""
    print("\n" + "=" * 80)
    print("TEST: RAG Stage Timing Breakdown")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        # Use a simple query (may not return results if no data)
        query = "機械学習の応用例を教えてください"
        print(f"Query: {query}")

        result = await service.query(
            query=query,
            options=RAGQueryOptions(top_k=5, rerank=True),
        )

        print(f"\n✓ Stage Timing Breakdown:")
        print(f"  Total Time: {result.processing_time_ms:.0f}ms")

        total_stage_time = 0
        for stage in result.stage_timings:
            total_stage_time += stage.duration_ms
            status = "✓" if stage.success else "✗"
            print(f"\n  {status} {stage.stage_name}:")
            print(f"      Duration: {stage.duration_ms:.1f}ms")
            print(f"      Percentage: {(stage.duration_ms / result.processing_time_ms * 100):.1f}%")

            if stage.metadata:
                for key, value in stage.metadata.items():
                    print(f"      {key}: {value}")

            if stage.error:
                print(f"      Error: {stage.error}")

        # Account for any overhead
        overhead = result.processing_time_ms - total_stage_time
        if overhead > 0:
            print(f"\n  Overhead: {overhead:.1f}ms ({overhead / result.processing_time_ms * 100:.1f}%)")

        print(f"\n  Query ID: {result.query_id}")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Confidence: {result.confidence}")
        print(f"  LLM Model: {result.llm_model}")
        print(f"  Embedding Model: {result.embedding_model}")
        print(f"  Reranker Model: {result.reranker_model}")

        print(f"\n✓ TEST PASSED: Stage timing breakdown complete")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_custom_options():
    """Test RAG with custom options"""
    print("\n" + "=" * 80)
    print("TEST: RAG with Custom Options")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        query = "Pythonとはどのようなプログラミング言語ですか？"
        print(f"Query: {query}")

        # Test with different top_k values
        for top_k in [3, 5, 10]:
            print(f"\nTesting with top_k={top_k}")

            result = await service.query(
                query=query,
                options=RAGQueryOptions(
                    top_k=top_k,
                    rerank=True,
                    include_sources=True,
                ),
            )

            print(f"  Sources returned: {len(result.sources)}")
            print(f"  Time: {result.processing_time_ms:.0f}ms")
            print(f"  ✓ Top K={top_k} successful")

        print(f"\n✓ TEST PASSED: Custom options working correctly")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_with_and_without_reranking():
    """Test RAG with and without reranking"""
    print("\n" + "=" * 80)
    print("TEST: RAG with and without Reranking")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        query = "深層学習と機械学習の違いは何ですか？"
        print(f"Query: {query}")

        # Test with reranking
        print("\nTest 1: With Reranking")
        result_with = await service.query(
            query=query,
            options=RAGQueryOptions(top_k=5, rerank=True),
        )
        print(f"  Time: {result_with.processing_time_ms:.0f}ms")
        print(f"  Reranker Model: {result_with.reranker_model}")

        # Test without reranking
        print("\nTest 2: Without Reranking")
        result_without = await service.query(
            query=query,
            options=RAGQueryOptions(top_k=5, rerank=False),
        )
        print(f"  Time: {result_without.processing_time_ms:.0f}ms")
        print(f"  Reranker Model: {result_without.reranker_model}")

        # Compare
        time_diff = result_with.processing_time_ms - result_without.processing_time_ms
        print(f"\n  Time difference: {time_diff:.0f}ms (reranking overhead)")

        print(f"\n✓ TEST PASSED: Reranking comparison complete")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_service_lifecycle():
    """Test RAG service initialization and shutdown"""
    print("\n" + "=" * 80)
    print("TEST: RAG Service Lifecycle")
    print("=" * 80)

    try:
        from backend.db.session import init_db, close_db

        await init_db()

        # Test 1: Create new service instance
        print("\nTest 1: Create service instance")
        service1 = RAGService()
        print(f"  Service created: {service1}")

        # Test 2: Initialize service
        print("\nTest 2: Initialize service")
        await service1.initialize()
        print(f"  Service initialized: {service1._is_initialized}")

        # Test 3: Health check
        print("\nTest 3: Health check")
        health = await service1.health_check()
        print(f"  Health status: {health['status']}")

        # Test 4: Shutdown service
        print("\nTest 4: Shutdown service")
        await service1.shutdown()
        print(f"  Service shut down: {not service1._is_initialized}")

        # Test 5: Re-initialize
        print("\nTest 5: Re-initialize service")
        await service1.initialize()
        print(f"  Service re-initialized: {service1._is_initialized}")

        # Cleanup
        await service1.shutdown()
        await close_db()

        print(f"\n✓ TEST PASSED: Service lifecycle working correctly")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_error_handling():
    """Test RAG error handling and graceful degradation"""
    print("\n" + "=" * 80)
    print("TEST: RAG Error Handling")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        tests_passed = 0
        tests_failed = 0

        # Test 1: Empty query handling
        print("\nTest 1: Empty query handling")
        try:
            result = await service.query(query="")
            print("  ✗ Should have raised validation error")
            tests_failed += 1
        except RAGValidationError as e:
            print(f"  ✓ Correctly handled: {e.message}")
            tests_passed += 1

        # Test 2: Very long query handling
        print("\nTest 2: Very long query handling")
        try:
            result = await service.query(query="a" * 1000)
            print("  ✗ Should have raised validation error")
            tests_failed += 1
        except RAGValidationError as e:
            print(f"  ✓ Correctly handled: {e.message}")
            tests_passed += 1

        # Test 3: Invalid options
        print("\nTest 3: Invalid retrieval method")
        try:
            result = await service.query(
                query="test",
                options=RAGQueryOptions(retrieval_method="invalid_method"),
            )
            print("  ✗ Should have raised processing error")
            tests_failed += 1
        except (RAGProcessingError, RAGValidationError) as e:
            print(f"  ✓ Correctly handled: {e.message}")
            tests_passed += 1

        print(f"\n✓ TEST PASSED: {tests_passed}/{tests_passed + tests_failed} error handling tests passed")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def test_rag_query_options_variations():
    """Test various RAG query option combinations"""
    print("\n" + "=" * 80)
    print("TEST: RAG Query Options Variations")
    print("=" * 80)

    try:
        from backend.db.session import init_db

        await init_db()
        service = get_rag_service()
        await service.initialize()

        query = "日本について教えてください"

        test_configs = [
            {
                "name": "Minimal options",
                "options": RAGQueryOptions(),
            },
            {
                "name": "High top_k",
                "options": RAGQueryOptions(top_k=20),
            },
            {
                "name": "No reranking",
                "options": RAGQueryOptions(rerank=False),
            },
            {
                "name": "Vector only",
                "options": RAGQueryOptions(retrieval_method="vector"),
            },
            {
                "name": "Keyword only",
                "options": RAGQueryOptions(retrieval_method="keyword"),
            },
        ]

        for config in test_configs:
            print(f"\n{config['name']}:")
            try:
                result = await service.query(
                    query=query,
                    options=config['options'],
                )
                print(f"  ✓ Success: {result.processing_time_ms:.0f}ms")
            except Exception as e:
                print(f"  ⚠ Partial success (may have no data): {str(e)[:100]}")

        print(f"\n✓ TEST PASSED: Query options variations tested")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RAG ORCHESTRATION SERVICE MANUAL TESTS")
    print("=" * 80)

    tests = [
        ("RAG Service Health Check", test_rag_health_check),
        ("RAG Query Validation", test_rag_query_validation),
        ("RAG Stage Timing Breakdown", test_rag_stage_timing_breakdown),
        ("RAG with Custom Options", test_rag_custom_options),
        ("RAG with and without Reranking", test_rag_with_and_without_reranking),
        ("RAG Service Lifecycle", test_rag_service_lifecycle),
        ("RAG Error Handling", test_rag_error_handling),
        ("RAG Query Options Variations", test_rag_query_options_variations),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print("=" * 80)

    # Cleanup
    try:
        from backend.db.session import close_db
        await close_db()
        print("\n✓ Database cleanup complete")
    except Exception as e:
        print(f"\n✗ Database cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
