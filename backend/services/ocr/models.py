"""
OCR Service Models
Pydantic models for OCR results and configuration
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box for text regions"""

    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

    model_config = {"from_attributes": True}


class OCRBlock(BaseModel):
    """A single text block detected by OCR"""

    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box of the text block")
    block_type: str = Field(
        default="text",
        description="Type of block: text, title, header, table, figure, etc.",
    )

    model_config = {"from_attributes": True}


class OCRPage(BaseModel):
    """OCR results for a single page"""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    text: str = Field(..., description="Full text content of the page")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    blocks: List[OCRBlock] = Field(
        default_factory=list, description="List of text blocks with positions"
    )
    width: Optional[int] = Field(None, description="Page width in pixels")
    height: Optional[int] = Field(None, description="Page height in pixels")
    rotation: float = Field(
        default=0.0, description="Detected rotation angle in degrees"
    )

    @field_validator("text")
    @classmethod
    def normalize_text(cls, v: str) -> str:
        """Normalize Japanese text (NFKC normalization)"""
        import unicodedata

        return unicodedata.normalize("NFKC", v.strip())

    model_config = {"from_attributes": True}


class OCROptions(BaseModel):
    """OCR processing options"""

    engine: str = Field(
        default="yomitoku",
        description="OCR engine: yomitoku or paddleocr",
    )
    language: str = Field(
        default="ja",
        description="Document language: ja, en, or multilingual",
    )
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for text",
    )
    enable_fallback: bool = Field(
        default=True,
        description="Enable fallback to secondary OCR engine",
    )
    fallback_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence threshold to trigger fallback",
    )
    preserve_layout: bool = Field(
        default=True,
        description="Preserve document layout (columns, tables)",
    )
    extract_tables: bool = Field(
        default=True,
        description="Extract table structures",
    )
    extract_images: bool = Field(
        default=False,
        description="Extract image descriptions (requires vision model)",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: markdown, text, json",
    )

    model_config = {"from_attributes": True}


class OCRResult(BaseModel):
    """Complete OCR result for a document"""

    document_id: str = Field(..., description="Document identifier")
    engine_used: str = Field(..., description="OCR engine used")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    pages: List[OCRPage] = Field(..., description="OCR results per page")
    markdown: str = Field(..., description="Full document in Markdown format")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    options: OCROptions = Field(..., description="Options used for OCR")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during processing",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Processing timestamp",
    )

    @field_validator("markdown")
    @classmethod
    def normalize_markdown(cls, v: str) -> str:
        """Normalize markdown output"""
        import unicodedata

        return unicodedata.normalize("NFKC", v)

    model_config = {"from_attributes": True}


class OCRError(Exception):
    """Base exception for OCR errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class OCREngineNotFoundError(OCRError):
    """Raised when OCR engine is not available"""


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails"""


class OCRConfidenceError(OCRError):
    """Raised when OCR confidence is below threshold"""
