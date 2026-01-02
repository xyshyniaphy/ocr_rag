#!/usr/bin/env python3
"""
Unit Tests for Configuration Management
Tests for backend/core/config.py
"""

import os
import pytest
from unittest.mock import patch

from backend.core.config import Settings, settings


class TestSettingsDefaults:
    """Test default settings values"""

    def test_app_name_default(self):
        """Test default APP_NAME"""
        assert settings.APP_NAME == "Japanese OCR RAG System"

    def test_app_version_default(self):
        """Test default APP_VERSION"""
        assert settings.APP_VERSION == "1.0.0"

    def test_debug_default(self):
        """Test default DEBUG"""
        assert settings.DEBUG is False

    def test_host_default(self):
        """Test default HOST"""
        assert settings.HOST == "0.0.0.0"

    def test_port_default(self):
        """Test default PORT"""
        assert settings.PORT == 8000

    def test_jwt_expiry_defaults(self):
        """Test JWT token expiry defaults"""
        assert settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 15
        assert settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS == 7

    def test_cors_origins_default(self):
        """Test CORS_ORIGINS contains expected defaults"""
        assert "http://localhost:8501" in settings.CORS_ORIGINS
        assert "http://localhost:8000" in settings.CORS_ORIGINS

    def test_postgres_defaults(self):
        """Test PostgreSQL defaults"""
        assert settings.POSTGRES_HOST == "localhost"
        assert settings.POSTGRES_PORT == 5432
        assert settings.POSTGRES_USER == "raguser"
        assert settings.POSTGRES_DB == "rag_metadata"

    def test_milvus_defaults(self):
        """Test Milvus defaults"""
        assert settings.MILVUS_HOST == "localhost"
        assert settings.MILVUS_PORT == 19530
        assert settings.MILVUS_COLLECTION == "document_chunks"

    def test_minio_defaults(self):
        """Test MinIO defaults"""
        assert settings.MINIO_ENDPOINT == "localhost:9000"
        assert settings.MINIO_ACCESS_KEY == "minioadmin"
        assert settings.MINIO_SECRET_KEY == "minioadmin"
        assert settings.MINIO_USE_SSL is False

    def test_redis_defaults(self):
        """Test Redis defaults"""
        assert settings.REDIS_HOST == "localhost"
        assert settings.REDIS_PORT == 6379
        assert settings.REDIS_DB == 0
        assert settings.REDIS_PASSWORD is None

    def test_celery_defaults(self):
        """Test Celery defaults"""
        assert "redis://localhost:6379/1" in settings.CELERY_BROKER_URL
        assert "redis://localhost:6379/2" in settings.CELERY_RESULT_BACKEND

    def test_ollama_defaults(self):
        """Test Ollama defaults"""
        assert settings.OLLAMA_HOST == "localhost:11434"
        assert settings.OLLAMA_MODEL == "qwen3:4b"
        assert settings.OLLAMA_NUM_CTX == 32768
        assert settings.OLLAMA_TEMPERATURE == 0.1
        assert settings.OLLAMA_TOP_P == 0.9
        assert settings.OLLAMA_TOP_K == 40
        assert settings.OLLAMA_NUM_PREDICT == 2048
        assert settings.OLLAMA_REPEAT_PENALTY == 1.1

    def test_ocr_defaults(self):
        """Test OCR defaults"""
        assert settings.OCR_ENGINE == "yomitoku"
        assert settings.OCR_CONFIDENCE_THRESHOLD == 0.85
        assert settings.OCR_FALLBACK_THRESHOLD == 0.80

    def test_embedding_defaults(self):
        """Test embedding defaults"""
        assert settings.EMBEDDING_MODEL == "sbintuitions/sarashina-embedding-v1-1b"
        assert settings.EMBEDDING_MODEL_PATH == "/app/models/sarashina"
        assert settings.EMBEDDING_DEVICE == "cuda:0"
        assert settings.EMBEDDING_BATCH_SIZE == 64
        assert settings.EMBEDDING_DIMENSION == 1792
        assert settings.EMBEDDING_MAX_LENGTH == 512
        assert settings.EMBEDDING_NORMALIZE is True
        assert settings.EMBEDDING_TRUNCATE is True
        assert settings.EMBEDDING_CACHE_ENABLED is True
        assert settings.EMBEDDING_CACHE_TTL == 86400

    def test_reranker_defaults(self):
        """Test reranker defaults"""
        assert settings.RERANKER_MODEL == "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
        assert settings.RERANKER_MODEL_PATH == "/app/reranker_models/llama-nv-reranker"
        assert settings.RERANKER_DEVICE == "cuda:0"
        assert settings.RERANKER_TOP_K_INPUT == 20
        assert settings.RERANKER_TOP_K_OUTPUT == 5
        assert settings.RERANKER_THRESHOLD == 0.65
        assert settings.RERANKER_BATCH_SIZE == 32

    def test_chunking_defaults(self):
        """Test chunking defaults"""
        assert settings.CHUNK_SIZE == 512
        assert settings.CHUNK_OVERLAP == 50
        assert settings.CHUNK_MAX_TABLE_SIZE == 2048

    def test_search_defaults(self):
        """Test search/retrieval defaults"""
        assert settings.SEARCH_TOP_K == 20
        assert settings.SEARCH_NPROBE == 128
        assert settings.HYBRID_SEARCH_VECTOR_WEIGHT == 0.7
        assert settings.HYBRID_SEARCH_KEYWORD_WEIGHT == 0.3
        assert settings.RETRIEVAL_MIN_SCORE == 0.3
        assert settings.RETRIEVAL_RRF_K == 60
        assert settings.RETRIEVAL_CACHE_ENABLED is True

    def test_rate_limit_defaults(self):
        """Test rate limiting defaults"""
        assert settings.RATE_LIMIT_PER_MINUTE == 60
        assert settings.RATE_LIMIT_UPLOAD_PER_MINUTE == 10

    def test_file_upload_defaults(self):
        """Test file upload defaults"""
        assert settings.MAX_UPLOAD_SIZE_MB == 50
        assert "application/pdf" in settings.ALLOWED_CONTENT_TYPES

    def test_monitoring_defaults(self):
        """Test monitoring defaults"""
        assert settings.ENABLE_METRICS is True
        assert settings.PROMETHEUS_PORT == 9100
        assert settings.ENABLE_TRACING is False

    def test_log_level_default(self):
        """Test LOG_LEVEL defaults"""
        assert settings.LOG_LEVEL == "INFO"


class TestSettingsValidation:
    """Test settings validation"""

    def test_environment_valid_values(self):
        """Test ENVIRONMENT accepts valid values"""
        for env in ["development", "staging", "production"]:
            s = Settings(ENVIRONMENT=env)
            assert s.ENVIRONMENT == env

    def test_environment_invalid_value(self):
        """Test ENVIRONMENT rejects invalid values"""
        with pytest.raises(ValueError, match="ENVIRONMENT must be one of"):
            Settings(ENVIRONMENT="invalid")

    def test_log_level_valid_values(self):
        """Test LOG_LEVEL accepts valid values"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            s = Settings(LOG_LEVEL=level)
            assert s.LOG_LEVEL == level

    def test_log_level_invalid_value(self):
        """Test LOG_LEVEL rejects invalid values"""
        with pytest.raises(ValueError, match="LOG_LEVEL must be one of"):
            Settings(LOG_LEVEL="INVALID")

    def test_log_level_case_conversion(self):
        """Test LOG_LEVEL is converted to uppercase"""
        s = Settings(LOG_LEVEL="debug")
        assert s.LOG_LEVEL == "DEBUG"

    def test_secret_key_min_length(self):
        """Test SECRET_KEY minimum length validation"""
        with pytest.raises(Exception):
            Settings(SECRET_KEY="short")

    def test_secret_key_valid_length(self):
        """Test SECRET_KEY with valid length"""
        # Should not raise
        Settings(SECRET_KEY="a" * 32)


class TestSettingsProperties:
    """Test settings properties and computed values"""

    def test_postgres_url_property(self):
        """Test POSTGRES_URL property construction"""
        expected = (
            f"postgresql+asyncpg://{settings.POSTGRES_USER}:"
            f"{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:"
            f"{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        assert settings.POSTGRES_URL == expected

    def test_postgres_url_components(self):
        """Test POSTGRES_URL contains correct components"""
        url = settings.POSTGRES_URL
        assert "postgresql+asyncpg://" in url
        assert settings.POSTGRES_USER in url
        assert settings.POSTGRES_HOST in url
        assert str(settings.POSTGRES_PORT) in url
        assert settings.POSTGRES_DB in url


class TestSettingsEnvironmentOverride:
    """Test environment variable overrides"""

    def test_override_app_name(self):
        """Test overriding APP_NAME via environment"""
        with patch.dict(os.environ, {"APP_NAME": "Test App"}):
            s = Settings()
            assert s.APP_NAME == "Test App"

    def test_override_port(self):
        """Test overriding PORT via environment"""
        with patch.dict(os.environ, {"PORT": "9000"}):
            s = Settings()
            assert s.PORT == 9000

    def test_override_embedding_device(self):
        """Test overriding EMBEDDING_DEVICE via environment"""
        with patch.dict(os.environ, {"EMBEDDING_DEVICE": "cpu"}):
            s = Settings()
            assert s.EMBEDDING_DEVICE == "cpu"

    def test_override_gpu_memory_fraction(self):
        """Test overriding GPU-related settings"""
        with patch.dict(os.environ, {
            "EMBEDDING_DEVICE": "cuda:1",
            "RERANKER_DEVICE": "cuda:1"
        }):
            s = Settings()
            assert s.EMBEDDING_DEVICE == "cuda:1"
            assert s.RERANKER_DEVICE == "cuda:1"


class TestSettingsEdgeCases:
    """Test edge cases and special scenarios"""

    def test_empty_cors_origins(self):
        """Test empty CORS_ORIGINS list"""
        with patch.dict(os.environ, {"CORS_ORIGINS": "[]"}):
            s = Settings()
            assert s.CORS_ORIGINS == []

    def test_multiple_cors_origins(self):
        """Test multiple CORS origins"""
        origins = '["http://localhost:3000", "https://example.com"]'
        with patch.dict(os.environ, {"CORS_ORIGINS": origins}):
            s = Settings()
            assert len(s.CORS_ORIGINS) == 2

    def test_zero_port(self):
        """Test PORT set to 0 (should allow)"""
        with patch.dict(os.environ, {"PORT": "0"}):
            s = Settings()
            assert s.PORT == 0

    def test_negative_port_invalid(self):
        """Test negative PORT (should fail validation)"""
        # Pydantic should handle this
        with pytest.raises(Exception):
            with patch.dict(os.environ, {"PORT": "-1"}):
                Settings()

    def test_boolean_string_parsing(self):
        """Test boolean settings from environment strings"""
        for bool_str, expected in [("true", True), ("True", True), ("1", True),
                                    ("false", False), ("False", False), ("0", False)]:
            with patch.dict(os.environ, {"DEBUG": bool_str}):
                s = Settings()
                assert s.DEBUG is expected


class TestSettingsModelPaths:
    """Test model path settings for base image compliance"""

    def test_sarashina_model_path(self):
        """Test Sarashina model path points to base image"""
        assert settings.EMBEDDING_MODEL_PATH == "/app/models/sarashina"
        assert settings.EMBEDDING_MODEL_PATH.startswith("/app/models/")

    def test_reranker_model_path(self):
        """Test reranker model path points to base image"""
        assert settings.RERANKER_MODEL_PATH == "/app/reranker_models/llama-nv-reranker"
        assert settings.RERANKER_MODEL_PATH.startswith("/app/reranker_models/")

    def test_model_paths_are_absolute(self):
        """Test all model paths are absolute paths"""
        # Should not be relative paths
        assert os.path.isabs(settings.EMBEDDING_MODEL_PATH)
        assert os.path.isabs(settings.RERANKER_MODEL_PATH)


class TestSettingsConfigurationClass:
    """Test Settings Config class"""

    def test_env_file_setting(self):
        """Test env_file is set to .env"""
        assert Settings.Config.env_file == ".env"

    def test_env_file_encoding(self):
        """Test env_file_encoding is utf-8"""
        assert Settings.Config.env_file_encoding == "utf-8"

    def test_case_sensitive(self):
        """Test case_sensitive is True"""
        assert Settings.Config.case_sensitive is True

    def test_extra_allowed(self):
        """Test extra fields are allowed"""
        assert Settings.Config.extra == "allow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
