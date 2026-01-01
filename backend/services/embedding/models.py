"""
Embedding Service Models
Pydantic models for embedding generation and results
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class EmbeddingOptions(BaseModel):
    """
    Options for embedding generation

    Attributes:
        batch_size: Number of texts to process in one batch
        normalize: Whether to normalize embeddings to unit length
        truncate: Whether to truncate text longer than max_length
        show_progress: Show progress bars for batch processing
    """

    batch_size: int = 64
    normalize: bool = True
    truncate: bool = True
    show_progress: bool = False

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        if v > 512:
            raise ValueError("batch_size must not exceed 512")
        return v


class Embedding(BaseModel):
    """
    A single text embedding vector

    Attributes:
        vector: The embedding vector (768D for Sarashina)
        dimension: Dimension of the embedding vector
        model: Model name used to generate the embedding
        text_hash: Hash of the input text for caching
    """

    vector: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., ge=1, description="Embedding dimension")
    model: str = Field(..., description="Model name")
    text_hash: Optional[str] = Field(None, description="Text hash for caching")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: List[float], info: ValidationInfo) -> List[float]:
        """Validate vector matches declared dimension"""
        if "dimension" in info.data:
            expected_dim = info.data["dimension"]
            if len(v) != expected_dim:
                raise ValueError(
                    f"Vector length {len(v)} does not match dimension {expected_dim}"
                )
        return v


class TextEmbedding(BaseModel):
    """
    Text with its embedding

    Attributes:
        text: Original input text
        embedding: Embedding vector
        token_count: Estimated token count
        processing_time_ms: Time taken to generate embedding
    """

    text: str = Field(..., description="Original text")
    embedding: Embedding = Field(..., description="Generated embedding")
    token_count: int = Field(..., ge=0, description="Estimated token count")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in ms")


class EmbeddingResult(BaseModel):
    """
    Result of embedding multiple texts

    Attributes:
        embeddings: List of text embeddings
        total_embeddings: Total number of embeddings generated
        dimension: Embedding dimension
        model: Model name used
        total_tokens: Total tokens processed
        processing_time_ms: Total processing time
        options: Options used for generation
        warnings: List of warnings (truncated texts, etc.)
    """

    embeddings: List[TextEmbedding] = Field(
        ..., description="List of generated embeddings"
    )
    total_embeddings: int = Field(..., ge=0, description="Total embeddings generated")
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model name")
    total_tokens: int = Field(..., ge=0, description="Total tokens processed")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in ms")
    options: EmbeddingOptions = Field(..., description="Generation options")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: List[TextEmbedding], info: ValidationInfo) -> List[TextEmbedding]:
        """Validate all embeddings have consistent dimensions"""
        if "dimension" in info.data:
            expected_dim = info.data["dimension"]
            for emb in v:
                if emb.embedding.dimension != expected_dim:
                    raise ValueError(
                        f"Embedding dimension {emb.embedding.dimension} "
                        f"does not match expected {expected_dim}"
                    )
        return v


class ChunkEmbeddingResult(BaseModel):
    """
    Result of embedding document chunks

    Attributes:
        chunk_embeddings: Mapping from chunk_id to embedding
        document_id: Document UUID
        total_chunks: Total chunks embedded
        dimension: Embedding dimension
        model: Model name used
        processing_time_ms: Total processing time
    """

    chunk_embeddings: Dict[str, Embedding] = Field(
        ..., description="Mapping from chunk_id to embedding"
    )
    document_id: str = Field(..., description="Document UUID")
    total_chunks: int = Field(..., ge=0, description="Total chunks embedded")
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model name")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in ms")


# Custom Exceptions


class EmbeddingError(Exception):
    """Base exception for embedding errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when embedding model is not found"""

    pass


class EmbeddingProcessingError(EmbeddingError):
    """Raised when embedding generation fails"""

    pass


class EmbeddingValidationError(EmbeddingError):
    """Raised when input validation fails"""

    pass
