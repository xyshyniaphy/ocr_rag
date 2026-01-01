"""
Text Chunking Strategies
Japanese-aware text splitting strategies for RAG
"""

import re
import uuid
import unicodedata
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from backend.core.logging import get_logger
from backend.services.ocr.models import OCRPage, OCRBlock
from backend.services.processing.chunking.models import (
    TextChunk,
    ChunkMetadata,
    ChunkingOptions,
)

logger = get_logger(__name__)


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies"""

    def __init__(self, options: ChunkingOptions):
        self.options = options

    @abstractmethod
    async def chunk_page(
        self,
        page: OCRPage,
        document_id: str,
        chunk_start_index: int = 0,
    ) -> List[TextChunk]:
        """Chunk a single OCR page"""
        pass


class JapaneseRecursiveSplitter(BaseChunkingStrategy):
    """
    Japanese-aware recursive text splitter

    Splits text hierarchically using separators in priority order:
    1. Paragraph breaks (\n\n)
    2. Line breaks (\n)
    3. Japanese sentence endings (。！？)
    4. Japanese commas (、)
    5. Character-level fallback
    """

    # Japanese punctuation patterns
    SENTENCE_ENDINGS = re.compile(r"[。！？]+")
    CLAUSAL_PAUSES = re.compile(r"[；、]+")
    WHITESPACE = re.compile(r"\s+")

    # Header/footer patterns (common in Japanese documents)
    HEADER_PATTERNS = [
        re.compile(r"^第\s*\d+\s*[章条]", re.MULTILINE),  # Chapter/section numbers
        re.compile(r"^\d+\s*/\s*\d+", re.MULTILINE),  # Page numbers
    ]

    def __init__(self, options: ChunkingOptions):
        super().__init__(options)
        self.separators = options.separators

    async def chunk_page(
        self,
        page: OCRPage,
        document_id: str,
        chunk_start_index: int = 0,
    ) -> List[TextChunk]:
        """
        Chunk a page using recursive splitting

        Args:
            page: OCR page with text content
            document_id: Document UUID
            chunk_start_index: Starting chunk index

        Returns:
            List of TextChunk objects
        """
        text = self._normalize_text(page.text)

        # Extract section headers if enabled
        section_headers = []
        if self.options.preserve_section_headers:
            section_headers = self._extract_headers(text)

        # Split text into chunks
        chunks_text = self._recursive_split(text)

        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            # Filter small chunks
            if len(chunk_text) < self.options.min_chunk_size:
                continue

            # Find section header for this chunk
            section_header = self._find_section_header(
                chunk_text, section_headers
            )

            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                page_number=page.page_number,
                chunk_index=chunk_start_index + i,
                document_id=document_id,
                chunk_type="text",
                section_header=section_header,
                confidence=page.confidence,
            )

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        logger.debug(
            f"JapaneseRecursiveSplitter: Created {len(chunks)} chunks "
            f"from page {page.page_number}"
        )

        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize Japanese text (NFKC)"""
        # NFKC normalization for Japanese
        text = unicodedata.normalize("NFKC", text)
        # Remove excessive whitespace
        text = self.WHITESPACE.sub(" ", text)
        return text.strip()

    def _extract_headers(self, text: str) -> List[str]:
        """Extract section headers from text"""
        headers = []
        lines = text.split("\n")

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue

            # Check if line matches header patterns
            for pattern in self.HEADER_PATTERNS:
                if pattern.match(line):
                    headers.append(line)
                    break

        return headers

    def _find_section_header(
        self,
        chunk_text: str,
        headers: List[str],
    ) -> Optional[str]:
        """Find the most relevant section header for a chunk"""
        if not headers:
            return None

        # Simple heuristic: return the last header seen
        # In production, you'd track position-based headers
        return headers[-1] if headers else None

    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using separators in priority order

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # If text fits in chunk size, return as single chunk
        if len(text) <= self.options.chunk_size:
            return [text] if text.strip() else []

        # Try each separator in order
        for separator in self.separators:
            if separator not in text:
                continue

            # Split by separator
            splits = text.split(separator)

            # Build chunks with overlap
            chunks = []
            current_chunk = ""

            for split in splits:
                split = split.strip()
                if not split:
                    continue

                # Check if adding split exceeds chunk size
                if len(current_chunk) + len(split) + len(separator) <= self.options.chunk_size:
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
                else:
                    # Save current chunk if not empty
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # Start new chunk
                    current_chunk = split

            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If we successfully created chunks, return them
            if len(chunks) > 1:
                # Add overlap between chunks
                return self._add_overlap(chunks)

        # Fallback: split by character if no separators worked
        return self._character_split(text)

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if not chunks or len(chunks) == 1:
            return chunks

        overlap_size = min(self.options.chunk_overlap, self.options.chunk_size // 4)

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap_size:] if len(prev_chunk) > overlap_size else prev_chunk
                overlapped.append(overlap_text + chunk)

        return overlapped

    def _character_split(self, text: str) -> List[str]:
        """Fallback character-based splitting"""
        chunks = []
        chunk_size = self.options.chunk_size
        overlap = self.options.chunk_overlap

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks


class SemanticChunker(BaseChunkingStrategy):
    """
    Semantic chunking using OCR block information

    Preserves semantic units like paragraphs, tables, and figures
    by leveraging OCR block layout information.
    """

    def __init__(self, options: ChunkingOptions):
        super().__init__(options)

    async def chunk_page(
        self,
        page: OCRPage,
        document_id: str,
        chunk_start_index: int = 0,
    ) -> List[TextChunk]:
        """
        Chunk a page using semantic block information

        Args:
            page: OCR page with blocks
            document_id: Document UUID
            chunk_start_index: Starting chunk index

        Returns:
            List of TextChunk objects
        """
        if not page.blocks:
            # Fallback to recursive splitting if no blocks
            fallback = JapaneseRecursiveSplitter(self.options)
            return await fallback.chunk_page(page, document_id, chunk_start_index)

        chunks = []
        current_chunk_text = ""
        current_chunk_blocks = []
        chunk_index = chunk_start_index

        for block in page.blocks:
            block_text = block.text.strip()

            # Check if adding block exceeds chunk size
            potential_size = len(current_chunk_text) + len(block_text) + 1

            if potential_size > self.options.chunk_size and current_chunk_text:
                # Finalize current chunk
                chunk = self._create_chunk_from_blocks(
                    current_chunk_blocks,
                    page,
                    document_id,
                    chunk_index,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk_text = block_text
                current_chunk_blocks = [block]
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n" + block_text
                else:
                    current_chunk_text = block_text
                current_chunk_blocks.append(block)

        # Add final chunk
        if current_chunk_blocks:
            chunk = self._create_chunk_from_blocks(
                current_chunk_blocks,
                page,
                document_id,
                chunk_index,
            )
            if chunk:
                chunks.append(chunk)

        logger.debug(
            f"SemanticChunker: Created {len(chunks)} chunks from page {page.page_number}"
        )

        return chunks

    def _create_chunk_from_blocks(
        self,
        blocks: List[OCRBlock],
        page: OCRPage,
        document_id: str,
        chunk_index: int,
    ) -> Optional[TextChunk]:
        """Create a TextChunk from OCR blocks"""
        if not blocks:
            return None

        # Combine block texts
        text = "\n".join(block.text for block in blocks)

        # Filter small chunks
        if len(text) < self.options.min_chunk_size:
            return None

        # Calculate average confidence
        avg_confidence = sum(b.confidence for b in blocks) / len(blocks)

        # Detect chunk type from blocks
        chunk_type = self._detect_chunk_type(blocks)

        chunk_id = str(uuid.uuid4())
        metadata = ChunkMetadata(
            page_number=page.page_number,
            chunk_index=chunk_index,
            document_id=document_id,
            chunk_type=chunk_type,
            confidence=avg_confidence,
            char_start=blocks[0].bbox.x0 if blocks else None,
            char_end=blocks[-1].bbox.x1 if blocks else None,
        )

        return TextChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
        )

    def _detect_chunk_type(self, blocks: List[OCRBlock]) -> str:
        """Detect chunk type from block types"""
        if not blocks:
            return "text"

        # Check if any block is a table
        if any(b.block_type == "table" for b in blocks):
            return "table"

        # Check if any block is a title/header
        if any(b.block_type in ["title", "header"] for b in blocks):
            return "title"

        return "text"


class TableAwareChunker(BaseChunkingStrategy):
    """
    Table-aware chunking that preserves table structure

    Detects tables and keeps them together as single chunks
    to maintain semantic integrity.
    """

    # Table detection patterns
    TABLE_PATTERNS = [
        re.compile(r"^\|.*\|$", re.MULTILINE),  # Markdown tables
        re.compile(r"^[\s\-\+|]{10,}$", re.MULTILINE),  # ASCII table borders
    ]

    def __init__(self, options: ChunkingOptions):
        super().__init__(options)

    async def chunk_page(
        self,
        page: OCRPage,
        document_id: str,
        chunk_start_index: int = 0,
    ) -> List[TextChunk]:
        """
        Chunk a page with table preservation

        Args:
            page: OCR page
            document_id: Document UUID
            chunk_start_index: Starting chunk index

        Returns:
            List of TextChunk objects
        """
        # Use OCR block information to detect tables
        has_tables = any(
            b.block_type == "table" for b in page.blocks
        )

        if has_tables and self.options.preserve_tables:
            return await self._chunk_with_tables(page, document_id, chunk_start_index)
        else:
            # Use semantic chunking as fallback
            semantic = SemanticChunker(self.options)
            return await semantic.chunk_page(page, document_id, chunk_start_index)

    async def _chunk_with_tables(
        self,
        page: OCRPage,
        document_id: str,
        chunk_start_index: int = 0,
    ) -> List[TextChunk]:
        """Chunk page while preserving table structures"""
        chunks = []
        chunk_index = chunk_start_index

        # Separate table and non-table blocks
        table_blocks = [b for b in page.blocks if b.block_type == "table"]
        text_blocks = [b for b in page.blocks if b.block_type != "table"]

        # Process text blocks with semantic chunking
        if text_blocks:
            # Create a fake page with only text blocks
            text_page = OCRPage(
                page_number=page.page_number,
                text="\n".join(b.text for b in text_blocks),
                confidence=page.confidence,
                blocks=text_blocks,
                width=page.width,
                height=page.height,
            )

            semantic = SemanticChunker(self.options)
            text_chunks = await semantic.chunk_page(text_page, document_id, chunk_index)
            chunks.extend(text_chunks)
            chunk_index += len(text_chunks)

        # Process tables as individual chunks
        for table_block in table_blocks:
            table_text = table_block.text.strip()

            if len(table_text) > self.options.min_chunk_size:
                chunk_id = str(uuid.uuid4())
                metadata = ChunkMetadata(
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    chunk_type="table",
                    confidence=table_block.confidence,
                )

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=table_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        logger.debug(
            f"TableAwareChunker: Created {len(chunks)} chunks "
            f"({len(table_blocks)} tables) from page {page.page_number}"
        )

        return chunks


def get_strategy(
    strategy_name: str,
    options: ChunkingOptions,
) -> BaseChunkingStrategy:
    """
    Get a chunking strategy by name

    Args:
        strategy_name: Strategy name (recursive, semantic, table_aware)
        options: Chunking options

    Returns:
        Chunking strategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "recursive": JapaneseRecursiveSplitter,
        "semantic": SemanticChunker,
        "table_aware": TableAwareChunker,
    }

    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        raise ValueError(
            f"Unknown chunking strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategy_class(options)
