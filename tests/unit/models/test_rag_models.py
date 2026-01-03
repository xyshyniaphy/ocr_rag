#!/usr/bin/env python3
"""
Unit Tests for RAG Models
Tests for backend/services/rag/models.py - Query options, results, and configurations
"""

import pytest
from typing import List
from pydantic import ValidationError

from backend.services.rag.models import (
    RAGQueryOptions,
    RAGSource,
    RAGStageMetrics,
    RAGResult,
    RAGPipelineConfig,
    RAGValidationError,
    RAGProcessingError,
    RAGServiceError,
)


class TestRAGQueryOptions:
    """Test RAG query options model"""

    def test_default_query_options(self):
        """Test default query options values"""
        options = RAGQueryOptions()

        assert options.top_k == 10
        assert options.retrieval_top_k == 20
        assert options.rerank_top_k == 10
        assert options.rerank is True
        assert options.retrieval_method == "hybrid"
        assert options.min_score == 0.0
        assert options.document_ids is None
        assert options.include_sources is True
        assert options.use_cache is True
        assert options.language == "ja"

    def test_query_options_with_reranker_enabled(self):
        """Test query options with reranker enabled"""
        options = RAGQueryOptions(
            top_k=5,
            rerank=True,
            rerank_top_k=5,
            retrieval_top_k=20
        )

        assert options.rerank is True
        assert options.rerank_top_k == 5
        assert options.top_k == 5
        assert options.retrieval_top_k == 20

    def test_query_options_with_reranker_disabled(self):
        """Test query options with reranker disabled"""
        options = RAGQueryOptions(
            top_k=5,
            rerank=False,
            retrieval_top_k=5  # When rerank is off, retrieval_top_k = top_k
        )

        assert options.rerank is False
        assert options.top_k == 5
        assert options.retrieval_top_k == 5

    @pytest.mark.parametrize("rerank", [True, False])
    def test_query_options_reranker_toggle(self, rerank):
        """Test query options with reranker toggled on/off"""
        options = RAGQueryOptions(
            top_k=5,
            rerank=rerank,
            retrieval_top_k=20 if rerank else 5
        )

        assert options.rerank == rerank
        assert options.top_k == 5

    def test_query_options_with_sources_1(self):
        """Test query options with 1 source"""
        options = RAGQueryOptions(
            top_k=1,
            include_sources=True
        )

        assert options.top_k == 1
        assert options.include_sources is True

    def test_query_options_with_sources_5(self):
        """Test query options with 5 sources"""
        options = RAGQueryOptions(
            top_k=5,
            include_sources=True
        )

        assert options.top_k == 5
        assert options.include_sources is True

    @pytest.mark.parametrize("top_k", [1, 5, 10, 20, 50])
    def test_query_options_various_source_counts(self, top_k):
        """Test query options with various source counts"""
        options = RAGQueryOptions(top_k=top_k)

        assert options.top_k == top_k

    def test_query_options_language_japanese(self):
        """Test query options with Japanese language"""
        options = RAGQueryOptions(language="ja")

        assert options.language == "ja"

    def test_query_options_language_english(self):
        """Test query options with English language"""
        options = RAGQueryOptions(language="en")

        assert options.language == "en"

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_query_options_various_languages(self, language):
        """Test query options with various languages"""
        options = RAGQueryOptions(language=language)

        assert options.language == language

    def test_query_options_with_document_filter(self):
        """Test query options with document ID filter"""
        doc_ids = ["doc1", "doc2", "doc3"]
        options = RAGQueryOptions(document_ids=doc_ids)

        assert options.document_ids == doc_ids
        assert len(options.document_ids) == 3

    def test_query_options_top_k_validation(self):
        """Test top_k must be between 1 and 100"""
        with pytest.raises(ValidationError):
            RAGQueryOptions(top_k=0)

        with pytest.raises(ValidationError):
            RAGQueryOptions(top_k=101)

    def test_query_options_min_score_validation(self):
        """Test min_score must be between 0 and 1"""
        with pytest.raises(ValidationError):
            RAGQueryOptions(min_score=-0.1)

        with pytest.raises(ValidationError):
            RAGQueryOptions(min_score=1.1)

    def test_query_options_retrieval_method_validation(self):
        """Test valid retrieval methods"""
        for method in ["vector", "keyword", "hybrid"]:
            options = RAGQueryOptions(retrieval_method=method)
            assert options.retrieval_method == method

    def test_query_options_complete_configuration(self):
        """Test complete query options configuration"""
        options = RAGQueryOptions(
            top_k=5,
            retrieval_top_k=20,
            rerank_top_k=5,
            rerank=True,
            retrieval_method="hybrid",
            min_score=0.3,
            document_ids=["doc1"],
            include_sources=True,
            use_cache=False,
            language="ja"
        )

        assert options.top_k == 5
        assert options.rerank is True
        assert options.min_score == 0.3
        assert options.use_cache is False
        assert options.language == "ja"


class TestRAGSource:
    """Test RAG source model"""

    def test_rag_source_minimal(self):
        """Test minimal RAG source creation"""
        source = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Sample text content",
            score=0.85
        )

        assert source.chunk_id == "chunk_001"
        assert source.document_id == "doc_123"
        assert source.text == "Sample text content"
        assert source.score == 0.85
        assert source.rerank_score is None
        assert source.document_title is None
        assert source.page_number is None
        assert source.chunk_index is None
        assert source.metadata is None

    def test_rag_source_with_rerank_score(self):
        """Test RAG source with rerank score"""
        source = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Sample text",
            score=0.75,
            rerank_score=0.92
        )

        assert source.score == 0.75
        assert source.rerank_score == 0.92
        assert source.rerank_score > source.score  # Rerank improved score

    def test_rag_source_without_rerank_score(self):
        """Test RAG source without rerank score (reranking disabled)"""
        source = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Sample text",
            score=0.85
            # No rerank_score provided
        )

        assert source.score == 0.85
        assert source.rerank_score is None

    def test_rag_source_complete(self):
        """Test complete RAG source with all fields"""
        source = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            document_title="Sample Document.pdf",
            text="This is the full chunk text content...",
            score=0.88,
            rerank_score=0.95,
            page_number=3,
            chunk_index=12,
            metadata={"source": "vector", "model": "sarashina"}
        )

        assert source.chunk_id == "chunk_001"
        assert source.document_title == "Sample Document.pdf"
        assert source.page_number == 3
        assert source.chunk_index == 12
        assert source.metadata["source"] == "vector"

    def test_rag_source_score_validation(self):
        """Test score must be between 0 and 1"""
        with pytest.raises(ValidationError):
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                text="Text",
                score=1.5
            )

        with pytest.raises(ValidationError):
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                text="Text",
                score=-0.1
            )

    def test_rag_sources_list_with_1_source(self):
        """Test RAG sources list with single source"""
        sources: List[RAGSource] = [
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                text="Single source text",
                score=0.90
            )
        ]

        assert len(sources) == 1
        assert sources[0].chunk_id == "chunk_001"

    def test_rag_sources_list_with_5_sources(self):
        """Test RAG sources list with 5 sources"""
        sources: List[RAGSource] = [
            RAGSource(
                chunk_id=f"chunk_{i:03d}",
                document_id="doc_123",
                text=f"Source text {i}",
                score=0.9 - (i * 0.1)
            )
            for i in range(5)
        ]

        assert len(sources) == 5
        assert sources[0].score == 0.9
        assert sources[4].score == 0.5
        assert all(s.document_id == "doc_123" for s in sources)

    def test_rag_source_comparison_with_and_without_rerank(self):
        """Test source scores with and without reranking"""
        source_without_rerank = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Text",
            score=0.75
        )

        source_with_rerank = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Text",
            score=0.75,
            rerank_score=0.92
        )

        # Both have same base score
        assert source_without_rerank.score == source_with_rerank.score
        # Only one has rerank score
        assert source_without_rerank.rerank_score is None
        assert source_with_rerank.rerank_score == 0.92


class TestRAGStageMetrics:
    """Test RAG stage metrics model"""

    def test_stage_metrics_minimal(self):
        """Test minimal stage metrics"""
        metrics = RAGStageMetrics(
            stage_name="retrieval",
            duration_ms=150.5,
            success=True
        )

        assert metrics.stage_name == "retrieval"
        assert metrics.duration_ms == 150.5
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.metadata is None

    def test_stage_metrics_with_failure(self):
        """Test stage metrics for failed stage"""
        metrics = RAGStageMetrics(
            stage_name="reranking",
            duration_ms=50.0,
            success=False,
            error="Reranker model not available"
        )

        assert metrics.success is False
        assert metrics.error == "Reranker model not available"

    def test_stage_metrics_complete(self):
        """Test complete stage metrics with metadata"""
        metrics = RAGStageMetrics(
            stage_name="retrieval",
            duration_ms=145.5,
            success=True,
            metadata={"retrieved_count": 20, "method": "hybrid"}
        )

        assert metrics.metadata["retrieved_count"] == 20
        assert metrics.metadata["method"] == "hybrid"

    @pytest.mark.parametrize("stage_name", [
        "query_understanding",
        "retrieval",
        "reranking",
        "context_assembly",
        "llm_generation"
    ])
    def test_stage_metrics_for_all_stages(self, stage_name):
        """Test stage metrics for all RAG pipeline stages"""
        metrics = RAGStageMetrics(
            stage_name=stage_name,
            duration_ms=100.0,
            success=True
        )

        assert metrics.stage_name == stage_name


class TestRAGResult:
    """Test RAG result model"""

    def test_rag_result_minimal(self):
        """Test minimal RAG result"""
        result = RAGResult(
            query="What is machine learning?",
            answer="Machine learning is...",
            sources=[],
            query_id="query_123",
            processing_time_ms=1500.0,
            stage_timings=[]
        )

        assert result.query == "What is machine learning?"
        assert result.answer == "Machine learning is..."
        assert len(result.sources) == 0
        assert result.query_id == "query_123"

    def test_rag_result_with_1_source(self):
        """Test RAG result with single source"""
        source = RAGSource(
            chunk_id="chunk_001",
            document_id="doc_123",
            text="Machine learning is...",
            score=0.92
        )

        result = RAGResult(
            query="Test query",
            answer="Test answer",
            sources=[source],
            query_id="query_001",
            processing_time_ms=1000.0,
            stage_timings=[]
        )

        assert len(result.sources) == 1
        assert result.sources[0].chunk_id == "chunk_001"

    def test_rag_result_with_5_sources(self):
        """Test RAG result with 5 sources"""
        sources = [
            RAGSource(
                chunk_id=f"chunk_{i:03d}",
                document_id="doc_123",
                text=f"Source text {i}",
                score=0.9 - (i * 0.1)
            )
            for i in range(5)
        ]

        result = RAGResult(
            query="Test query",
            answer="Test answer",
            sources=sources,
            query_id="query_002",
            processing_time_ms=2000.0,
            stage_timings=[]
        )

        assert len(result.sources) == 5
        assert result.sources[0].score == 0.9
        assert result.sources[4].score == 0.5

    def test_rag_result_with_reranking_enabled(self):
        """Test RAG result when reranking was enabled"""
        sources = [
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                text="Text 1",
                score=0.75,
                rerank_score=0.92  # Improved by reranking
            ),
            RAGSource(
                chunk_id="chunk_002",
                document_id="doc_123",
                text="Text 2",
                score=0.80,
                rerank_score=0.88
            )
        ]

        result = RAGResult(
            query="Test query",
            answer="Test answer",
            sources=sources,
            query_id="query_003",
            processing_time_ms=2500.0,
            stage_timings=[],
            reranker_model="nvidia/Llama-3.2-NV-RerankQA-1B-v2"
        )

        assert result.reranker_model == "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
        assert all(s.rerank_score is not None for s in sources)
        assert result.sources[0].rerank_score == 0.92

    def test_rag_result_without_reranking(self):
        """Test RAG result when reranking was disabled"""
        sources = [
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                text="Text 1",
                score=0.85  # No rerank_score
            ),
            RAGSource(
                chunk_id="chunk_002",
                document_id="doc_123",
                text="Text 2",
                score=0.78
            )
        ]

        result = RAGResult(
            query="Test query",
            answer="Test answer",
            sources=sources,
            query_id="query_004",
            processing_time_ms=1500.0,
            stage_timings=[]
            # No reranker_model specified
        )

        assert result.reranker_model is None
        assert all(s.rerank_score is None for s in sources)

    def test_rag_result_japanese_language(self):
        """Test RAG result with Japanese query and answer"""
        result = RAGResult(
            query="機械学習とは何ですか？",
            answer="機械学習は...",
            sources=[],
            query_id="query_ja_001",
            processing_time_ms=3000.0,
            stage_timings=[],
            metadata={"language": "ja"}
        )

        assert "機械学習" in result.query
        assert result.metadata["language"] == "ja"

    def test_rag_result_english_language(self):
        """Test RAG result with English query and answer"""
        result = RAGResult(
            query="What is machine learning?",
            answer="Machine learning is...",
            sources=[],
            query_id="query_en_001",
            processing_time_ms=2500.0,
            stage_timings=[],
            metadata={"language": "en"}
        )

        assert "machine learning" in result.query.lower()
        assert result.metadata["language"] == "en"

    @pytest.mark.parametrize("language,query_prefix", [
        ("ja", "日本語"),
        ("en", "English"),
        ("zh", "中文"),
        ("ko", "한국어")
    ])
    def test_rag_result_various_languages(self, language, query_prefix):
        """Test RAG result with various languages"""
        result = RAGResult(
            query=f"{query_prefix} query text",
            answer=f"{query_prefix} answer text",
            sources=[],
            query_id=f"query_{language}_001",
            processing_time_ms=2000.0,
            stage_timings=[],
            metadata={"language": language}
        )

        assert result.metadata["language"] == language

    def test_rag_result_complete_with_all_fields(self):
        """Test complete RAG result with all optional fields"""
        sources = [
            RAGSource(
                chunk_id="chunk_001",
                document_id="doc_123",
                document_title="ML Guide.pdf",
                text="Machine learning is...",
                score=0.92,
                rerank_score=0.95,
                page_number=1,
                chunk_index=0
            )
        ]

        stage_timings = [
            RAGStageMetrics(
                stage_name="retrieval",
                duration_ms=150.0,
                success=True,
                metadata={"retrieved_count": 20}
            ),
            RAGStageMetrics(
                stage_name="reranking",
                duration_ms=500.0,
                success=True
            ),
            RAGStageMetrics(
                stage_name="llm_generation",
                duration_ms=1850.0,
                success=True
            )
        ]

        result = RAGResult(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            sources=sources,
            query_id="query_complete_001",
            processing_time_ms=2500.0,
            stage_timings=stage_timings,
            confidence=0.92,
            llm_model="qwen3:4b",
            embedding_model="sbintuitions/sarashina-embedding-v1-1b",
            reranker_model="nvidia/Llama-3.2-NV-RerankQA-1B-v2",
            metadata={"language": "en", "cache_hit": False}
        )

        assert len(result.sources) == 1
        assert len(result.stage_timings) == 3
        assert result.confidence == 0.92
        assert result.llm_model == "qwen3:4b"
        assert result.embedding_model == "sbintuitions/sarashina-embedding-v1-1b"
        assert result.reranker_model == "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
        assert result.metadata["cache_hit"] is False

    def test_rag_result_confidence_validation(self):
        """Test confidence must be between 0 and 1"""
        sources = []

        # Valid confidence values
        for confidence in [0.0, 0.5, 1.0]:
            result = RAGResult(
                query="Test",
                answer="Answer",
                sources=sources,
                query_id="query_test",
                processing_time_ms=1000.0,
                stage_timings=[],
                confidence=confidence
            )
            assert result.confidence == confidence

        # Invalid confidence values
        with pytest.raises(ValidationError):
            RAGResult(
                query="Test",
                answer="Answer",
                sources=sources,
                query_id="query_test",
                processing_time_ms=1000.0,
                stage_timings=[],
                confidence=-0.1
            )

        with pytest.raises(ValidationError):
            RAGResult(
                query="Test",
                answer="Answer",
                sources=sources,
                query_id="query_test",
                processing_time_ms=1000.0,
                stage_timings=[],
                confidence=1.1
            )


class TestRAGPipelineConfig:
    """Test RAG pipeline configuration model"""

    def test_pipeline_config_default(self):
        """Test default pipeline configuration"""
        config = RAGPipelineConfig()

        assert config.retrieval_method == "hybrid"
        assert config.retrieval_top_k == 20
        assert config.min_score == 0.0
        assert config.enable_reranking is False  # Disabled by default
        assert config.rerank_top_k == 10
        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 2048
        assert config.llm_top_p == 0.9
        assert config.system_prompt is None
        assert config.enable_cache is True
        assert config.default_language == "ja"

    def test_pipeline_config_with_reranking_enabled(self):
        """Test pipeline config with reranking enabled"""
        config = RAGPipelineConfig(
            enable_reranking=True,
            rerank_top_k=5,
            retrieval_top_k=20
        )

        assert config.enable_reranking is True
        assert config.rerank_top_k == 5

    def test_pipeline_config_with_reranking_disabled(self):
        """Test pipeline config with reranking disabled"""
        config = RAGPipelineConfig(
            enable_reranking=False
        )

        assert config.enable_reranking is False

    @pytest.mark.parametrize("enable_reranking", [True, False])
    def test_pipeline_config_reranking_toggle(self, enable_reranking):
        """Test pipeline config with reranking toggled"""
        config = RAGPipelineConfig(enable_reranking=enable_reranking)

        assert config.enable_reranking == enable_reranking

    def test_pipeline_config_japanese_language(self):
        """Test pipeline config with Japanese language"""
        config = RAGPipelineConfig(default_language="ja")

        assert config.default_language == "ja"

    def test_pipeline_config_english_language(self):
        """Test pipeline config with English language"""
        config = RAGPipelineConfig(default_language="en")

        assert config.default_language == "en"

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko", "es"])
    def test_pipeline_config_various_languages(self, language):
        """Test pipeline config with various languages"""
        config = RAGPipelineConfig(default_language=language)

        assert config.default_language == language

    def test_pipeline_config_custom_system_prompt(self):
        """Test pipeline config with custom system prompt"""
        custom_prompt = "You are a helpful assistant for Japanese legal documents."
        config = RAGPipelineConfig(system_prompt=custom_prompt)

        assert config.system_prompt == custom_prompt

    def test_pipeline_config_retrieval_methods(self):
        """Test pipeline config with different retrieval methods"""
        for method in ["vector", "keyword", "hybrid"]:
            config = RAGPipelineConfig(retrieval_method=method)
            assert config.retrieval_method == method

    def test_pipeline_config_cache_disabled(self):
        """Test pipeline config with cache disabled"""
        config = RAGPipelineConfig(enable_cache=False)

        assert config.enable_cache is False


class TestRAGExceptions:
    """Test RAG custom exceptions"""

    def test_rag_validation_error_minimal(self):
        """Test RAG validation error with minimal info"""
        error = RAGValidationError("Invalid query")

        assert error.message == "Invalid query"
        assert error.details == {}
        assert str(error) == "Invalid query"

    def test_rag_validation_error_with_details(self):
        """Test RAG validation error with details"""
        details = {"field": "top_k", "value": 0, "constraint": ">= 1"}
        error = RAGValidationError("Invalid top_k value", details=details)

        assert error.message == "Invalid top_k value"
        assert error.details["field"] == "top_k"
        assert error.details["value"] == 0

    def test_rag_processing_error_minimal(self):
        """Test RAG processing error with minimal info"""
        error = RAGProcessingError("Processing failed")

        assert error.message == "Processing failed"
        assert error.stage is None
        assert error.details == {}
        assert str(error) == "Processing failed"

    def test_rag_processing_error_with_stage(self):
        """Test RAG processing error with stage info"""
        error = RAGProcessingError(
            "Reranking failed",
            stage="reranking"
        )

        assert error.message == "Reranking failed"
        assert error.stage == "reranking"

    def test_rag_processing_error_complete(self):
        """Test RAG processing error with all info"""
        details = {"error_code": "MODEL_UNAVAILABLE", "model": "reranker"}
        error = RAGProcessingError(
            "Reranker service unavailable",
            stage="reranking",
            details=details
        )

        assert error.message == "Reranker service unavailable"
        assert error.stage == "reranking"
        assert error.details["error_code"] == "MODEL_UNAVAILABLE"

    def test_rag_service_error(self):
        """Test RAG service error"""
        error = RAGServiceError("Cannot connect to Milvus")

        assert error.message == "Cannot connect to Milvus"
        assert error.details == {}

    def test_rag_service_error_with_details(self):
        """Test RAG service error with details"""
        details = {"host": "localhost", "port": 19530}
        error = RAGServiceError(
            "Connection timeout",
            details=details
        )

        assert error.message == "Connection timeout"
        assert error.details["host"] == "localhost"


class TestRAGIntegrationScenarios:
    """Test integrated RAG scenarios with various configurations"""

    def test_scenario_japanese_query_with_reranker_5_sources(self):
        """Test Japanese query with reranking enabled, 5 sources"""
        # Query options
        options = RAGQueryOptions(
            top_k=5,
            rerank=True,
            rerank_top_k=5,
            retrieval_top_k=20,
            language="ja"
        )

        # Mock sources with original and reranked scores
        sources = [
            RAGSource(
                chunk_id=f"chunk_{i}",
                document_id="doc_ja_001",
                text=f"日本語のテキスト {i}",
                score=0.8 - (i * 0.1),
                rerank_score=0.95 - (i * 0.05)
            )
            for i in range(5)
        ]

        # Result
        result = RAGResult(
            query="機械学習について教えてください",
            answer="機械学習は...",
            sources=sources,
            query_id="query_ja_rerank",
            processing_time_ms=3500.0,
            stage_timings=[
                RAGStageMetrics(stage_name="retrieval", duration_ms=200.0, success=True),
                RAGStageMetrics(stage_name="reranking", duration_ms=600.0, success=True),
                RAGStageMetrics(stage_name="llm_generation", duration_ms=2700.0, success=True)
            ],
            reranker_model="nvidia/Llama-3.2-NV-RerankQA-1B-v2",
            metadata={"language": "ja"}
        )

        assert options.language == "ja"
        assert options.rerank is True
        assert len(result.sources) == 5
        assert all(s.rerank_score is not None for s in result.sources)
        assert result.metadata["language"] == "ja"

    def test_scenario_english_query_without_reranker_1_source(self):
        """Test English query without reranking, 1 source"""
        # Query options
        options = RAGQueryOptions(
            top_k=1,
            rerank=False,
            retrieval_top_k=1,
            language="en"
        )

        # Single source without rerank score
        sources = [
            RAGSource(
                chunk_id="chunk_en_001",
                document_id="doc_en_001",
                text="Machine learning is a subset of artificial intelligence...",
                score=0.92
                # No rerank_score
            )
        ]

        # Result
        result = RAGResult(
            query="Explain machine learning",
            answer="Machine learning is...",
            sources=sources,
            query_id="query_en_no_rerank",
            processing_time_ms=1800.0,
            stage_timings=[
                RAGStageMetrics(stage_name="retrieval", duration_ms=150.0, success=True),
                RAGStageMetrics(stage_name="llm_generation", duration_ms=1650.0, success=True)
            ],
            metadata={"language": "en"}
        )

        assert options.language == "en"
        assert options.rerank is False
        assert len(result.sources) == 1
        assert result.sources[0].rerank_score is None
        assert result.reranker_model is None
        assert result.metadata["language"] == "en"

    def test_scenario_japanese_query_without_reranker_5_sources(self):
        """Test Japanese query without reranking, 5 sources"""
        options = RAGQueryOptions(
            top_k=5,
            rerank=False,
            retrieval_top_k=5,
            language="ja"
        )

        sources = [
            RAGSource(
                chunk_id=f"chunk_ja_{i}",
                document_id="doc_ja_002",
                text=f"日本語テキスト {i}",
                score=0.9 - (i * 0.1)
            )
            for i in range(5)
        ]

        result = RAGResult(
            query="AIについて説明してください",
            answer="人工知能は...",
            sources=sources,
            query_id="query_ja_no_rerank_5",
            processing_time_ms=2200.0,
            stage_timings=[
                RAGStageMetrics(stage_name="retrieval", duration_ms=180.0, success=True),
                RAGStageMetrics(stage_name="llm_generation", duration_ms=2020.0, success=True)
            ],
            metadata={"language": "ja"}
        )

        assert options.rerank is False
        assert len(result.sources) == 5
        assert all(s.rerank_score is None for s in result.sources)

    def test_scenario_english_query_with_reranker_1_source(self):
        """Test English query with reranking, 1 source"""
        options = RAGQueryOptions(
            top_k=1,
            rerank=True,
            rerank_top_k=1,
            retrieval_top_k=10,
            language="en"
        )

        sources = [
            RAGSource(
                chunk_id="chunk_en_single",
                document_id="doc_en_002",
                text="Single best matching source",
                score=0.75,
                rerank_score=0.98  # Significantly improved
            )
        ]

        result = RAGResult(
            query="Best ML algorithm?",
            answer="The best algorithm depends on...",
            sources=sources,
            query_id="query_en_rerank_1",
            processing_time_ms=2800.0,
            stage_timings=[
                RAGStageMetrics(stage_name="retrieval", duration_ms=200.0, success=True),
                RAGStageMetrics(stage_name="reranking", duration_ms=500.0, success=True),
                RAGStageMetrics(stage_name="llm_generation", duration_ms=2100.0, success=True)
            ],
            reranker_model="nvidia/Llama-3.2-NV-RerankQA-1B-v2",
            metadata={"language": "en"}
        )

        assert len(result.sources) == 1
        assert result.sources[0].rerank_score == 0.98
        assert result.reranker_model is not None

    @pytest.mark.parametrize("language,query,answer_prefix", [
        ("ja", "日本のテスト", "日本語の回答"),
        ("en", "English test", "English answer"),
        ("ja", "別の日本語クエリ", "別の日本語回答"),
        ("en", "Another English query", "Another English answer")
    ])
    def test_scenario_multilingual_queries_without_reranker(
        self, language, query, answer_prefix
    ):
        """Test multilingual queries without reranking"""
        options = RAGQueryOptions(
            top_k=5,
            rerank=False,
            language=language
        )

        sources = [
            RAGSource(
                chunk_id=f"chunk_{language}_{i}",
                document_id=f"doc_{language}",
                text=f"Content {i}",
                score=0.9 - (i * 0.1)
            )
            for i in range(5)
        ]

        result = RAGResult(
            query=query,
            answer=f"{answer_prefix} content",
            sources=sources,
            query_id=f"query_{language}",
            processing_time_ms=2500.0,
            stage_timings=[],
            metadata={"language": language}
        )

        assert options.language == language
        assert options.rerank is False
        assert len(result.sources) == 5
        assert result.metadata["language"] == language

    @pytest.mark.parametrize("rerank,top_k", [
        (True, 1),
        (True, 5),
        (False, 1),
        (False, 5)
    ])
    def test_scenario_combination_of_reranker_and_source_count(
        self, rerank, top_k
    ):
        """Test all combinations of reranker on/off and source counts"""
        options = RAGQueryOptions(
            top_k=top_k,
            rerank=rerank,
            retrieval_top_k=20 if rerank else top_k,
            language="en"
        )

        sources = [
            RAGSource(
                chunk_id=f"chunk_{i}",
                document_id="doc_test",
                text=f"Text {i}",
                score=0.9 - (i * 0.1),
                rerank_score=0.95 - (i * 0.05) if rerank else None
            )
            for i in range(top_k)
        ]

        result = RAGResult(
            query="Test query",
            answer="Test answer",
            sources=sources,
            query_id="query_test",
            processing_time_ms=2000.0,
            stage_timings=[],
            reranker_model="nvidia/Llama-3.2-NV-RerankQA-1B-v2" if rerank else None,
            metadata={"rerank_enabled": rerank}
        )

        assert len(result.sources) == top_k
        assert options.rerank == rerank
        if rerank:
            assert all(s.rerank_score is not None for s in result.sources)
            assert result.reranker_model is not None
        else:
            assert all(s.rerank_score is None for s in result.sources)
            assert result.reranker_model is None
