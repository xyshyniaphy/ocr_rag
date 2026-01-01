#!/usr/bin/env python3
"""
LLM Service Manual Tests
Comprehensive tests for LLM Service using Qwen3:4b via Ollama

Usage:
    # From within the app container
    docker exec ocr-rag-app-dev python /app/tests/manual/test_llm.py

    # Run specific test
    docker exec ocr-rag-app-dev python /app/tests/manual/test_llm.py::test_chat_basic
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.core.logging import get_logger
from backend.services.llm import (
    get_llm_service,
    Message,
    LLMOptions,
    RAGContext,
)

logger = get_logger(__name__)


# Sample test data
SAMPLE_MESSAGES = [
    Message(role="system", content="あなたは役立つAIアシスタントです。"),
    Message(role="user", content="日本の首都はどこですか？"),
]

SAMPLE_RAG_CONTEXTS = [
    {
        "text": "機械学習は、人工智能の一分野であり、コンピュータがデータから自動的に学習する技術です。",
        "doc_id": "doc_001",
        "score": 0.95,
        "metadata": {"source": "wikipedia", "category": "ml"},
    },
    {
        "text": "深層学習（ディープラーニング）は、機械学習の一種で、多層ニューラルネットワークを使用します。",
        "doc_id": "doc_002",
        "score": 0.88,
        "metadata": {"source": "textbook", "category": "dl"},
    },
    {
        "text": "自然言語処理（NLP）は、人間の言語をコンピュータに理解させる技術です。",
        "doc_id": "doc_003",
        "score": 0.75,
        "metadata": {"source": "paper", "category": "nlp"},
    },
]


async def test_chat_basic():
    """Test basic chat completion"""
    print("\n" + "=" * 60)
    print("TEST: Basic Chat Completion")
    print("=" * 60)

    service = await get_llm_service()
    await service.initialize()

    response = await service.chat(
        messages=[
            Message(role="user", content="日本の首都はどこですか？简潔に答えてください。")
        ],
        options=LLMOptions(temperature=0.0, num_predict=100),
    )

    print(f"\n✓ Chat Response:")
    print(f"  Content: {response.content[:200]}...")
    print(f"  Model: {response.model}")
    print(f"  Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})")
    print(f"  Time: {response.processing_time_ms:.0f}ms")
    print(f"  Finish Reason: {response.finish_reason}")


async def test_completion_simple():
    """Test simple text completion"""
    print("\n" + "=" * 60)
    print("TEST: Simple Text Completion")
    print("=" * 60)

    service = await get_llm_service()

    response = await service.complete(
        prompt="機械学習について、1行で説明してください。",
        options=LLMOptions(temperature=0.3, num_predict=50),
    )

    print(f"\n✓ Completion Response:")
    print(f"  Content: {response.content[:200]}...")
    print(f"  Tokens: {response.total_tokens}")
    print(f"  Time: {response.processing_time_ms:.0f}ms")


async def test_chat_conversation():
    """Test multi-turn conversation"""
    print("\n" + "=" * 60)
    print("TEST: Multi-turn Conversation")
    print("=" * 60)

    service = await get_llm_service()

    messages = [
        Message(role="system", content="あなたは日本語のAIアシスタントです。简潔に答えてください。"),
        Message(role="user", content="私の名前は太郎です。"),
        Message(role="assistant", content="こんにちは、太郎さん！"),
        Message(role="user", content="私の名前を覚えていますか？"),
    ]

    response = await service.chat(messages=messages)

    print(f"\n✓ Conversation Response:")
    print(f"  Content: {response.content[:300]}...")
    print(f"  Model: {response.model}")
    print(f"  Time: {response.processing_time_ms:.0f}ms")


async def test_rag_generation():
    """Test RAG-augmented generation"""
    print("\n" + "=" * 60)
    print("TEST: RAG-augmented Generation")
    print("=" * 60)

    service = await get_llm_service()

    query = "機械学習と深層学習の違いは何ですか？"

    # Convert dicts to RAGContext objects
    contexts = [RAGContext(**ctx) for ctx in SAMPLE_RAG_CONTEXTS]

    response = await service.generate_rag(
        query=query,
        contexts=contexts,
        options=LLMOptions(temperature=0.2, num_predict=500),
    )

    print(f"\n✓ RAG Response:")
    print(f"  Query: {response.query}")
    print(f"  Answer: {response.answer[:400]}...")
    print(f"  Sources: {len(response.sources)} documents")
    for i, source in enumerate(response.sources[:3], 1):
        print(f"    [{i}] doc_id={source['doc_id']}, score={source['score']:.2f}")
    print(f"  Model: {response.model}")
    print(f"  Time: {response.processing_time_ms:.0f}ms")


async def test_custom_options():
    """Test custom generation options"""
    print("\n" + "=" * 60)
    print("TEST: Custom Generation Options")
    print("=" * 60)

    service = await get_llm_service()

    # Test with different temperature settings
    for temp in [0.0, 0.5, 1.0]:
        response = await service.chat(
            messages=[Message(role="user", content="AIについて创意的な答えを教えてください。")],
            options=LLMOptions(temperature=temp, num_predict=100),
        )

        print(f"\n✓ Temperature {temp}:")
        print(f"  Content: {response.content[:150]}...")
        print(f"  Time: {response.processing_time_ms:.0f}ms")


async def test_health_check():
    """Test health check"""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    service = await get_llm_service()
    await service.initialize()

    health = await service.health_check()

    print(f"\n✓ Health Status:")
    print(f"  Status: {health['status']}")
    print(f"  Model: {health['model']}")
    print(f"  Ollama Status: {health.get('ollama_status', 'N/A')}")
    print(f"  Available Models: {len(health.get('available_models', []))}")

    if 'test_generation_time_ms' in health:
        print(f"  Test Generation Time: {health['test_generation_time_ms']:.0f}ms")
        print(f"  Test Response Length: {health['test_response_length']} chars")


async def test_model_list():
    """Test model listing"""
    print("\n" + "=" * 60)
    print("TEST: Model Listing")
    print("=" * 60)

    service = await get_llm_service()

    models = await service.client.list_models(force_refresh=True)

    print(f"\n✓ Available Models ({len(models)}):")
    for model in models:
        print(f"  - {model}")


async def test_japanese_support():
    """Test Japanese language support"""
    print("\n" + "=" * 60)
    print("TEST: Japanese Language Support")
    print("=" * 60)

    service = await get_llm_service()

    test_cases = [
        "こんにちは、元気ですか？",
        "機械学習とは何ですか？",
        "日本の伝統文化について説明してください。",
    ]

    for query in test_cases:
        response = await service.chat(
            messages=[Message(role="user", content=query)],
            options=LLMOptions(temperature=0.3, num_predict=100),
        )

        print(f"\n✓ Query: {query}")
        print(f"  Response: {response.content[:100]}...")


async def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "=" * 60)
    print("TEST: Performance Benchmark")
    print("=" * 60)

    service = await get_llm_service()

    test_queries = [
        "AIとは何ですか？" * 10,  # Long prompt
        "日本の首都はどこですか？",  # Short prompt
    ]

    print("\nBenchmarking chat completions...")

    for query in test_queries:
        times = []
        for i in range(3):
            start = time.time()
            response = await service.chat(
                messages=[Message(role="user", content=query)],
                options=LLMOptions(temperature=0.0, num_predict=200),
            )
            times.append(response.processing_time_ms)

        avg_time = sum(times) / len(times)
        print(f"\n✓ Query (length={len(query)}):")
        print(f"  Avg Time: {avg_time:.0f}ms")
        print(f"  Min Time: {min(times):.0f}ms")
        print(f"  Max Time: {max(times):.0f}ms")
        print(f"  Avg Tokens: {response.total_tokens}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" LLM SERVICE MANUAL TESTS")
    print(" Model: Qwen3:4b via Ollama")
    print("=" * 80)

    tests = [
        ("Basic Chat", test_chat_basic),
        ("Simple Completion", test_completion_simple),
        ("Multi-turn Conversation", test_chat_conversation),
        ("RAG Generation", test_rag_generation),
        ("Custom Options", test_custom_options),
        ("Japanese Support", test_japanese_support),
        ("Health Check", test_health_check),
        ("Model List", test_model_list),
        ("Performance Benchmark", run_performance_benchmark),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print("=" * 80)


if __name__ == "__main__":
    # Run specific test if provided
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_funcs = {
            "test_chat_basic": test_chat_basic,
            "test_completion_simple": test_completion_simple,
            "test_chat_conversation": test_chat_conversation,
            "test_rag_generation": test_rag_generation,
            "test_custom_options": test_custom_options,
            "test_health_check": test_health_check,
            "test_model_list": test_model_list,
            "test_japanese_support": test_japanese_support,
            "run_performance_benchmark": run_performance_benchmark,
        }

        if test_name in test_funcs:
            asyncio.run(test_funcs[test_name]())
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_funcs.keys())}")
    else:
        asyncio.run(main())
