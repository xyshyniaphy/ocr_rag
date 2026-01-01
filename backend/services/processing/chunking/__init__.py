"""
Text Chunking Service
Splits OCR results into searchable chunks with Japanese-aware processing
"""

import re
import time
import uuid
from typing import List, Optional

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.ocr.models import OCRResult, OCRPage
from backend.services.processing.chunking.models import (
    TextChunk,
    ChunkMetadata,
    ChunkingOptions,
    ChunkingResult,
)
from backend.services.processing.chunking.strategies import (
    BaseChunkingStrategy,
    get_strategy,
)

logger = get_logger(__name__)


class ChunkingService:
    """
    Text chunking service for RAG preprocessing

    Splits OCR results into optimally-sized chunks for:
    - Embedding generation (Sarashina max 512 tokens)
    - Vector search relevance
    - LLM context assembly
    """

    # Default chunking strategy
    DEFAULT_STRATEGY = "semantic"

    def __init__(
        self,
        options: Optional[ChunkingOptions] = None,
        strategy: str = DEFAULT_STRATEGY,
    ):
        """
        Initialize chunking service

        Args:
            options: Chunking options (uses defaults if None)
            strategy: Chunking strategy name
        """
        if options is None:
            options = ChunkingOptions(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                max_table_size=settings.CHUNK_MAX_TABLE_SIZE,
            )

        self.options = options
        self.strategy_name = strategy
        self._strategy: Optional[BaseChunkingStrategy] = None

    async def chunk_result(
        self,
        ocr_result: OCRResult,
        document_id: Optional[str] = None,
    ) -> ChunkingResult:
        """
        Chunk an entire OCR result (multi-page document)

        Args:
            ocr_result: OCR processing result
            document_id: Document UUID (generates new if None)

        Returns:
            ChunkingResult with all chunks
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        start_time = time.time()
        all_chunks = []
        warnings = []

        logger.info(
            f"Starting chunking for document {document_id}: "
            f"{ocr_result.total_pages} pages, "
            f"{len(ocr_result.pages)} pages with content"
        )

        # Get or create strategy
        strategy = self._get_strategy()

        # Process each page
        chunk_index = 0
        for page in ocr_result.pages:
            try:
                page_chunks = await strategy.chunk_page(
                    page,
                    document_id,
                    chunk_index,
                )

                # Filter chunks by quality
                valid_chunks = self._filter_chunks(page_chunks)
                all_chunks.extend(valid_chunks)
                chunk_index += len(page_chunks)

                # Warn if filtered chunks
                if len(valid_chunks) < len(page_chunks):
                    filtered = len(page_chunks) - len(valid_chunks)
                    warnings.append(
                        f"Filtered {filtered} low-quality chunks from page {page.page_number}"
                    )

            except Exception as e:
                logger.error(f"Failed to chunk page {page.page_number}: {e}")
                warnings.append(
                    f"Failed to chunk page {page.page_number}: {str(e)}"
                )

        # Calculate statistics
        total_chars = sum(len(c.text) for c in all_chunks)
        total_tokens = sum(c.token_count for c in all_chunks)
        avg_size = total_chars / len(all_chunks) if all_chunks else 0.0

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks, "
            f"{total_chars} chars, {total_tokens} tokens, "
            f"{processing_time}ms"
        )

        return ChunkingResult(
            document_id=document_id,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            total_characters=total_chars,
            total_tokens=total_tokens,
            avg_chunk_size=avg_size,
            processing_time_ms=processing_time,
            options=self.options,
            warnings=warnings,
        )

    async def chunk_page(
        self,
        page: OCRPage,
        document_id: str,
        chunk_index: int = 0,
    ) -> List[TextChunk]:
        """
        Chunk a single page

        Args:
            page: OCR page
            document_id: Document UUID
            chunk_index: Starting chunk index

        Returns:
            List of TextChunk objects
        """
        strategy = self._get_strategy()
        return await strategy.chunk_page(page, document_id, chunk_index)

    async def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
    ) -> List[TextChunk]:
        """
        Chunk raw text (without OCR metadata)

        Args:
            text: Raw text content
            document_id: Document UUID
            page_number: Page number

        Returns:
            List of TextChunk objects
        """
        # Create a fake OCR page for compatibility
        from backend.services.ocr.models import OCRPage

        page = OCRPage(
            page_number=page_number,
            text=text,
            confidence=1.0,  # Unknown confidence for raw text
            blocks=[],
        )

        return await self.chunk_page(page, document_id)

    def _get_strategy(self) -> BaseChunkingStrategy:
        """Get or create chunking strategy instance"""
        if self._strategy is None:
            self._strategy = get_strategy(self.strategy_name, self.options)
        return self._strategy

    def _filter_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Filter chunks by quality criteria

        Args:
            chunks: List of chunks to filter

        Returns:
            Filtered list of chunks
        """
        filtered = []

        for chunk in chunks:
            # Filter by minimum size
            if len(chunk.text) < self.options.min_chunk_size:
                continue

            # Filter by confidence if available
            if chunk.metadata.confidence is not None:
                if chunk.metadata.confidence < 0.3:
                    continue

            # Filter headers/footers if enabled
            if self.options.filter_headers:
                if self._is_header_footer(chunk.text):
                    continue

            filtered.append(chunk)

        return filtered

    def _is_header_footer(self, text: str) -> bool:
        """
        Detect if text is likely a header or footer

        Args:
            text: Text to check

        Returns:
            True if likely header/footer
        """
        # Page numbers
        if text.strip().isdigit() and len(text.strip()) <= 3:
            return True

        # Page number patterns (第X页, etc.)
        if re.match(r"^第\s*\d+\s*页?$", text.strip()):
            return True

        # URL patterns
        if re.match(r"^https?://", text.strip()):
            return True

        # Very short text with mostly symbols
        if len(text.strip()) < 10:
            symbol_ratio = sum(1 for c in text if not c.isalnum()) / max(len(text), 1)
            if symbol_ratio > 0.5:
                return True

        return False


# Convenience functions for common use cases

async def chunk_document(
    ocr_result: OCRResult,
    document_id: Optional[str] = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    strategy: str = "semantic",
) -> ChunkingResult:
    """
    Chunk a document from OCR results

    Convenience function for document chunking

    Args:
        ocr_result: OCR processing result
        document_id: Document UUID (auto-generated if None)
        chunk_size: Target chunk size in characters
        chunk_overlap: Character overlap between chunks
        strategy: Chunking strategy name

    Returns:
        ChunkingResult with all chunks
    """
    options = ChunkingOptions(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    service = ChunkingService(options=options, strategy=strategy)
    return await service.chunk_result(ocr_result, document_id)


async def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Chunk raw text into chunks

    Convenience function for simple text chunking

    Args:
        text: Raw text content
        document_id: Document UUID
        chunk_size: Target chunk size
        chunk_overlap: Overlap size

    Returns:
        List of TextChunk objects
    """
    options = ChunkingOptions(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    service = ChunkingService(options=options)
    return await service.chunk_text(text, document_id)
