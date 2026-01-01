"""
Configuration Management
Loads settings from environment variables with type validation
"""

import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Japanese OCR RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
    ]

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "rag_password"
    POSTGRES_DB: str = "rag_metadata"

    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_chunks"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_USE_SSL: bool = False

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Ollama
    OLLAMA_HOST: str = "localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:14b-instruct-q4_K_M"
    OLLAMA_NUM_CTX: int = 32768
    OLLAMA_TEMPERATURE: float = 0.1

    # OCR
    OCR_ENGINE: str = "yomitoku"
    OCR_CONFIDENCE_THRESHOLD: float = 0.85
    OCR_FALLBACK_THRESHOLD: float = 0.80

    # Embedding
    EMBEDDING_MODEL: str = "sbintuitions/sarashina-embedding-v1-1b"
    EMBEDDING_MODEL_PATH: str = "/app/models/sarashina"
    EMBEDDING_DEVICE: str = "cuda:0"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_MAX_LENGTH: int = 512

    # Reranker
    RERANKER_MODEL: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    RERANKER_DEVICE: str = "cuda:0"
    RERANKER_TOP_K_INPUT: int = 20
    RERANKER_TOP_K_OUTPUT: int = 5
    RERANKER_THRESHOLD: float = 0.65

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNK_MAX_TABLE_SIZE: int = 2048

    # Search
    SEARCH_TOP_K: int = 20
    SEARCH_NPROBE: int = 128
    HYBRID_SEARCH_VECTOR_WEIGHT: float = 0.7
    HYBRID_SEARCH_KEYWORD_WEIGHT: float = 0.3

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 10

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_CONTENT_TYPES: List[str] = ["application/pdf"]

    # Monitoring
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9100
    ENABLE_TRACING: bool = False
    JAEGER_HOST: Optional[str] = None
    JAEGER_PORT: int = 6831

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = ["development", "staging", "production"]
        if v not in valid:
            raise ValueError(f"ENVIRONMENT must be one of {valid}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return v_upper

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


# Global settings instance
settings = Settings()
