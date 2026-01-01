"""
OCR Base Classes
Abstract base class and common functionality for OCR engines
"""

from abc import ABC, abstractmethod
from typing import List, Optional, BinaryIO
from pathlib import Path
import io

from backend.core.logging import get_logger
from backend.services.ocr.models import (
    OCRResult,
    OCRPage,
    OCROptions,
    OCREngineNotFoundError,
    OCRProcessingError,
)

logger = get_logger(__name__)


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engines

    All OCR engine implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, options: OCROptions):
        """
        Initialize the OCR engine

        Args:
            options: OCR processing options
        """
        self.options = options
        self._model = None
        self._processor = None
        self._is_loaded = False

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the name of this OCR engine"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the OCR engine/library"""
        pass

    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the OCR model into memory

        This method should load the model and set self._is_loaded = True
        """
        pass

    @abstractmethod
    async def process_page(
        self,
        image_bytes: bytes,
        page_number: int,
        dpi: int = 200,
    ) -> OCRPage:
        """
        Process a single page image

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)
            page_number: Page number (1-indexed)
            dpi: DPI for rendering PDF to image

        Returns:
            OCRPage with extracted text and metadata
        """
        pass

    @abstractmethod
    async def process_pdf(
        self,
        pdf_bytes: bytes,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> OCRResult:
        """
        Process a PDF document

        Args:
            pdf_bytes: Raw PDF bytes
            start_page: First page to process (1-indexed, None = first page)
            end_page: Last page to process (1-indexed, None = last page)

        Returns:
            OCRResult with all extracted text
        """
        pass

    async def is_available(self) -> bool:
        """
        Check if this OCR engine is available

        Returns:
            True if the engine can be used
        """
        try:
            await self.load_model()
            return self._is_loaded
        except Exception as e:
            logger.warning(f"{self.engine_name} not available: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._is_loaded

    async def unload(self) -> None:
        """
        Unload the model and free memory

        Override this method if the engine needs specific cleanup
        """
        self._model = None
        self._processor = None
        self._is_loaded = False
        logger.info(f"{self.engine_name} model unloaded")

    def _convert_pdf_to_images(
        self,
        pdf_bytes: bytes,
        dpi: int = 200,
    ) -> List[bytes]:
        """
        Convert PDF pages to images using pdf2image

        Args:
            pdf_bytes: Raw PDF bytes
            dpi: DPI for rendering

        Returns:
            List of image bytes (one per page)
        """
        try:
            from pdf2image import convert_from_bytes
            import io

            # Convert PDF to PIL Images
            pil_images = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                fmt="png",
                thread_count=4,
            )

            # Convert to bytes
            image_list = []
            for img in pil_images:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                image_list.append(img_bytes.read())

            logger.info(f"Converted PDF to {len(image_list)} images at {dpi} DPI")
            return image_list

        except ImportError:
            raise OCRProcessingError(
                "pdf2image library not available",
                details={"install": "pip install pdf2image"},
            )
        except Exception as e:
            raise OCRProcessingError(
                f"Failed to convert PDF to images: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    def _generate_markdown(self, pages: List[OCRPage], options: OCROptions) -> str:
        """
        Generate Markdown output from OCR pages

        Args:
            pages: List of OCRPage results
            options: OCR options

        Returns:
            Markdown formatted text
        """
        if options.output_format == "text":
            # Plain text without formatting
            return "\n\n".join(page.text for page in pages)

        # Markdown format
        md_lines = []

        for page in pages:
            # Page separator
            md_lines.append(f"\n---\n## Page {page.page_number}\n")

            if options.preserve_layout:
                # Preserve structure using blocks
                current_y = None
                for block in page.blocks:
                    # Detect column breaks by Y position
                    if current_y is not None and abs(block.bbox.y0 - current_y) > 50:
                        md_lines.append("")  # New paragraph

                    # Format based on block type
                    if block.block_type == "title" or block.block_type == "header":
                        md_lines.append(f"### {block.text}")
                    elif block.block_type == "table":
                        md_lines.append(f"\n{block.text}\n")
                    else:
                        md_lines.append(block.text)

                    current_y = block.bbox.y0
            else:
                # Just use the extracted text
                md_lines.append(page.text)

        return "\n".join(md_lines)

    def _calculate_confidence(self, pages: List[OCRPage]) -> float:
        """
        Calculate overall confidence from all pages

        Args:
            pages: List of OCRPage results

        Returns:
            Weighted average confidence
        """
        if not pages:
            return 0.0

        total_confidence = sum(page.confidence for page in pages)
        return total_confidence / len(pages)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.unload()
