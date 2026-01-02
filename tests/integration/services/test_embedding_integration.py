#!/usr/bin/env python3
"""
Integration Tests for Embedding Service
Tests for embedding service integration with models
"""

import pytest
import numpy as np

from backend.services.embedding import get_embedding_service, SarashinaEmbeddingModel, EmbeddingOptions


@pytest.mark.integration
class TestEmbeddingServiceIntegration:
    """Test embedding service integration"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initializes with model"""
        service = await get_embedding_service()
        assert service is not None
        assert service.model is not None

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test service health check"""
        service = await get_embedding_service()
        health = await service.health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert "model" in health
        assert "dimension" in health

    @pytest.mark.asyncio
    async def test_embed_single_text_japanese(self):
        """Test embedding single Japanese text"""
        service = await get_embedding_service()
        text = "これはテストです。"

        result = await service.embed_text(text, use_cache=False)

        assert result.embedding.vector is not None
        assert len(result.embedding.vector) == 1792
        assert result.token_count > 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_embed_batch_texts(self):
        """Test embedding batch of texts"""
        service = await get_embedding_service()
        texts = [
            "最初のテキストです。",
            "二番目のテキストです。",
            "三番目のテキストです。",
        ]

        result = await service.embed_texts(texts, use_cache=False)

        assert result.total_embeddings == len(texts)
        assert result.dimension == 1792
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_embed_chunks(self):
        """Test embedding document chunks"""
        service = await get_embedding_service()
        chunks = {
            "chunk_001": "最初のチャンク",
            "chunk_002": "二番目のチャンク",
            "chunk_003": "三番目のチャンク",
        }

        result = await service.embed_chunks(chunks, "doc_001", use_cache=False)

        assert result.total_chunks == len(chunks)
        assert result.dimension == 1792

    @pytest.mark.asyncio
    async def test_embed_long_text_truncation(self):
        """Test long text is truncated correctly"""
        service = await get_embedding_service()
        # Create very long text (> max_length)
        long_text = "テスト" * 200  # Much longer than 512 tokens

        result = await service.embed_text(long_text, use_cache=False)

        # Should truncate to max_length
        assert result.token_count <= service.model.max_length

    @pytest.mark.asyncio
    async def test_embed_empty_text_error(self):
        """Test empty text raises error"""
        service = await get_embedding_service()

        with pytest.raises(Exception):
            await service.embed_text("", use_cache=False)

    @pytest.mark.asyncio
    async def test_embed_cache_hit(self):
        """Test cache returns same embedding"""
        service = await get_embedding_service()
        text = "キャッシュテスト"

        # First call - cache miss
        result1 = await service.embed_text(text, use_cache=True)
        # Second call - cache hit
        result2 = await service.embed_text(text, use_cache=True)

        # Should have same hash
        assert result1.embedding.text_hash == result2.embedding.text_hash
        # Vectors should be identical
        assert result1.embedding.vector == result2.embedding.vector

    @pytest.mark.asyncio
    async def test_embed_cache_miss(self):
        """Test cache miss generates new embedding"""
        service = await get_embedding_service()

        result1 = await service.embed_text("テキスト1", use_cache=False)
        result2 = await service.embed_text("テキスト2", use_cache=False)

        # Different texts should have different hashes
        assert result1.embedding.text_hash != result2.embedding.text_hash


@pytest.mark.integration
@pytest.mark.gpu
class TestEmbeddingModelGPU:
    """Test embedding model on GPU"""

    @pytest.mark.asyncio
    async def test_model_on_cuda(self):
        """Test model runs on CUDA"""
        from backend.core.config import settings

        if settings.EMBEDDING_DEVICE.startswith("cuda"):
            model = SarashinaEmbeddingModel(
                model_path=settings.EMBEDDING_MODEL_PATH,
                device=settings.EMBEDDING_DEVICE
            )
            # Check if model is on GPU
            import torch
            assert torch.cuda.is_available()
            # Model should use GPU
            assert "cuda" in str(model._model.device)


@pytest.mark.integration
class TestEmbeddingQuality:
    """Test embedding quality and properties"""

    @pytest.mark.asyncio
    async def test_embedding_normalized(self):
        """Test embeddings are normalized"""
        service = await get_embedding_service()
        text = "正規化テスト"

        result = await service.embed_text(text, use_cache=False)

        # Check normalization
        vector = np.array(result.embedding.vector)
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5  # Should be unit vector

    @pytest.mark.asyncio
    async def test_embedding_similarity_same_text(self):
        """Test same text has identical embeddings"""
        service = await get_embedding_service()
        text = "同一性テスト"

        result1 = await service.embed_text(text, use_cache=False)
        result2 = await service.embed_text(text, use_cache=False)

        # Compute cosine similarity
        vec1 = np.array(result1.embedding.vector)
        vec2 = np.array(result2.embedding.vector)
        similarity = np.dot(vec1, vec2)

        # Should be nearly identical
        assert abs(similarity - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embedding_similarity_different_texts(self):
        """Test different texts have different embeddings"""
        service = await get_embedding_service()
        text1 = "日本語のテキスト"
        text2 = "全く異なるテキスト"

        result1 = await service.embed_text(text1, use_cache=False)
        result2 = await service.embed_text(text2, use_cache=False)

        # Compute cosine similarity
        vec1 = np.array(result1.embedding.vector)
        vec2 = np.array(result2.embedding.vector)
        similarity = np.dot(vec1, vec2)

        # Should be different (similarity < 1)
        assert similarity < 0.99

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self):
        """Test embedding dimension is consistent"""
        service = await get_embedding_service()
        texts = ["短い", "中程度の長さのテキスト", "非常に長いテキストです。" * 10]

        results = await service.embed_texts(texts, use_cache=False)

        # All should have same dimension
        for result in results.embeddings:
            assert len(result.vector) == 1792


@pytest.mark.integration
class TestEmbeddingErrorHandling:
    """Test embedding service error handling"""

    @pytest.mark.asyncio
    async def test_model_path_not_found(self):
        """Test error when model path is invalid"""
        with pytest.raises(Exception) as exc_info:
            model = SarashinaEmbeddingModel(
                model_path="/invalid/path",
                device="cpu"
            )
            # Try to load
            _ = model._model

        # Should raise appropriate error
        assert "model" in str(exc_info.value).lower() or "path" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_device(self):
        """Test error with invalid device"""
        with pytest.raises(Exception):
            SarashinaEmbeddingModel(
                model_path="/app/models/sarashina",
                device="invalid:device:999"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
