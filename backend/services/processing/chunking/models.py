"""
Text Chunking Models
Pydantic models for chunking configuration and results
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Chunk index within page")
    document_id: str = Field(..., description="Document ID")

    # Position info
    char_start: Optional[int] = Field(None, description="Character start position")
    char_end: Optional[int] = Field(None, description="Character end position")

    # Structural info
    chunk_type: str = Field(
        default="text",
        description="Type: text, title, table, list, code",
    )
    section_header: Optional[str] = Field(None, description="Section header if available")
    paragraph_index: Optional[int] = Field(None, description="Paragraph index")

    # Quality metrics
    token_count: Optional[int] = Field(None, description="Estimated token count")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="OCR confidence")

    model_config = {"from_attributes": True}


class TextChunk(BaseModel):
    """A single text chunk"""

    chunk_id: str = Field(..., description="Unique chunk ID")
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")

    model_config = {"from_attributes": True}

    @property
    def token_count(self) -> int:
        """Get or estimate token count"""
        if self.metadata.token_count is not None:
            return self.metadata.token_count
        # Rough estimate: Japanese chars ≈ 0.5 tokens, ASCII ≈ 0.25 tokens
        japanese_chars = sum(1 for c in self.text if ord(c) > 0x3000)
        ascii_chars = len(self.text) - japanese_chars
        return int(japanese_chars * 0.5 + ascii_chars * 0.25)


class ChunkingOptions(BaseModel):
    """Text chunking configuration options"""

    # Size constraints
    chunk_size: int = Field(
        default=512,
        ge=50,
        le=4096,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Character overlap between chunks",
    )

    # Japanese-specific separators (in priority order)
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "！", "？", "；", "、"],
        description="Text separators for splitting (Japanese-aware)",
    )

    # Table handling
    max_table_size: int = Field(
        default=2048,
        ge=100,
        description="Maximum size for table chunks",
    )
    preserve_tables: bool = Field(
        default=True,
        description="Keep table content together",
    )

    # Quality filters
    min_chunk_size: int = Field(
        default=20,
        ge=1,
        description="Minimum chunk size to keep",
    )
    filter_headers: bool = Field(
        default=True,
        description="Filter page headers/footers",
    )

    # Metadata
    preserve_section_headers: bool = Field(
        default=True,
        description="Include section headers in chunks",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size"""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    model_config = {"from_attributes": True}


class ChunkingResult(BaseModel):
    """Result of text chunking operation"""

    document_id: str = Field(..., description="Document ID")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    chunks: List[TextChunk] = Field(
        default_factory=list, description="All text chunks"
    )

    # Statistics
    total_characters: int = Field(
        default=0, ge=0, description="Total characters across all chunks"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Estimated total tokens"
    )
    avg_chunk_size: float = Field(
        default=0.0, ge=0.0, description="Average chunk size"
    )

    # Processing info
    processing_time_ms: int = Field(
        default=0, ge=0, description="Processing time in milliseconds"
    )
    options: ChunkingOptions = Field(
        default_factory=ChunkingOptions,
        description="Chunking options used",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Processing warnings"
    )

    model_config = {"from_attributes": True}

    @field_validator("chunks")
    @classmethod
    def validate_chunks(cls, v: List[TextChunk], info) -> List[TextChunk]:
        """Validate chunks are properly ordered"""
        if not v:
            return v

        # Check chunk indices are sequential
        prev_page = None
        prev_index = -1

        for chunk in v:
            if prev_page is not None and chunk.metadata.page_number < prev_page:
                raise ValueError("Chunks must be ordered by page number")

            if prev_page == chunk.metadata.page_number:
                if chunk.metadata.chunk_index != prev_index + 1:
                    raise ValueError(
                        f"Chunk indices must be sequential: "
                        f"expected {prev_index + 1}, got {chunk.metadata.chunk_index}"
                    )
                prev_index = chunk.metadata.chunk_index
            else:
                prev_page = chunk.metadata.page_number
                prev_index = chunk.metadata.chunk_index

        return v
