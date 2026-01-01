"""
YomiToku OCR Engine
Japanese OCR engine from Preferred Networks (PFN)
https://github.com/PreferredNetworks/yomitoku

YomiToku is optimized for Japanese documents with:
- Multi-column layout detection
- Table recognition
- High accuracy on vertical text
- Reading order detection
"""

import time
import uuid
from typing import List, Optional
import io

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.ocr.base import BaseOCREngine
from backend.services.ocr.models import (
    OCRResult,
    OCRPage,
    OCRBlock,
    BoundingBox,
    OCROptions,
    OCRProcessingError,
)

logger = get_logger(__name__)


class YomiTokuOCREngine(BaseOCREngine):
    """
    YomiToku OCR Engine implementation

    YomiToku is a Japanese OCR library from Preferred Networks
    that provides excellent accuracy for Japanese documents.
    """

    # Model path configuration
    MODEL_PATH = "/app/models/yomitoku"

    def __init__(self, options: Optional[OCROptions] = None):
        """
        Initialize YomiToku OCR engine

        Args:
            options: OCR processing options (uses defaults if None)
        """
        if options is None:
            options = OCROptions(
                engine="yomitoku",
                confidence_threshold=settings.OCR_CONFIDENCE_THRESHOLD,
                fallback_threshold=settings.OCR_FALLBACK_THRESHOLD,
            )

        super().__init__(options)
        self._reader = None

    @property
    def engine_name(self) -> str:
        return "YomiToku"

    @property
    def version(self) -> str:
        """Return YomiToku version"""
        try:
            import yomitoku
            return yomitoku.__version__
        except (ImportError, AttributeError):
            return "unknown"

    async def load_model(self) -> None:
        """
        Load YomiToku model

        The model files are expected to be in /app/models/yomitoku/
        from the Docker base image.
        """
        if self._is_loaded:
            logger.debug("YomiToku model already loaded")
            return

        try:
            import yomitoku

            # Initialize the YomiToku reader
            # YomiToku automatically loads models from the default location
            self._reader = yomitoku.load_reader()

            self._is_loaded = True
            logger.info("YomiToku OCR model loaded successfully")

        except ImportError as e:
            raise OCRProcessingError(
                "YomiToku library not installed",
                details={"install": "pip install yomitoku"},
            )
        except Exception as e:
            raise OCRProcessingError(
                f"Failed to load YomiToku model: {str(e)}",
                details={"model_path": self.MODEL_PATH, "error_type": type(e).__name__},
            )

    async def process_page(
        self,
        image_bytes: bytes,
        page_number: int,
        dpi: int = 200,
    ) -> OCRPage:
        """
        Process a single page image with YomiToku

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)
            page_number: Page number (1-indexed)
            dpi: DPI for rendering

        Returns:
            OCRPage with extracted text and metadata
        """
        if not self._is_loaded:
            await self.load_model()

        try:
            from PIL import Image

            start_time = time.time()

            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))

            # Run YomiToku OCR
            # YomiToku returns structured data with text, boxes, and layout
            result = self._reader.readtext(
                image,
                detect_orientation=True,
                preserve_layout=self.options.preserve_layout,
                extract_tables=self.options.extract_tables,
            )

            processing_time = int((time.time() - start_time) * 1000)

            # Parse YomiToku result
            blocks = []
            total_confidence = 0.0

            for item in result:
                # YomiToku returns: (text, bbox, confidence)
                text = item.get("text", "")
                bbox_data = item.get("bbox", [])
                confidence = item.get("confidence", 0.0)

                if not text or confidence < self.options.confidence_threshold:
                    continue

                # Create bounding box
                if len(bbox_data) >= 4:
                    x_coords = [p[0] for p in bbox_data]
                    y_coords = [p[1] for p in bbox_data]
                    bbox = BoundingBox(
                        x0=min(x_coords),
                        y0=min(y_coords),
                        x1=max(x_coords),
                        y1=max(y_coords),
                        width=max(x_coords) - min(x_coords),
                        height=max(y_coords) - min(y_coords),
                    )
                else:
                    bbox = BoundingBox(
                        x0=0, y0=0, x1=100, y1=20, width=100, height=20
                    )

                # Determine block type
                block_type = item.get("type", "text")

                blocks.append(
                    OCRBlock(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        block_type=block_type,
                    )
                )
                total_confidence += confidence

            # Combine all text
            full_text = "\n".join(block.text for block in blocks)
            avg_confidence = total_confidence / len(blocks) if blocks else 0.0

            logger.debug(
                f"YomiToku processed page {page_number}: "
                f"{len(blocks)} blocks, {avg_confidence:.2%} confidence, "
                f"{processing_time}ms"
            )

            return OCRPage(
                page_number=page_number,
                text=full_text,
                confidence=avg_confidence,
                blocks=blocks,
                width=image.width,
                height=image.height,
            )

        except Exception as e:
            logger.error(f"YomiToku page processing failed: {e}")
            raise OCRProcessingError(
                f"Failed to process page {page_number}: {str(e)}",
                details={
                    "page_number": page_number,
                    "error_type": type(e).__name__,
                },
            )

    async def process_pdf(
        self,
        pdf_bytes: bytes,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> OCRResult:
        """
        Process a PDF document with YomiToku

        Args:
            pdf_bytes: Raw PDF bytes
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed)

        Returns:
            OCRResult with all extracted text
        """
        if not self._is_loaded:
            await self.load_model()

        start_time = time.time()
        document_id = str(uuid.uuid4())

        try:
            # Convert PDF to images
            all_images = self._convert_pdf_to_images(pdf_bytes, dpi=200)

            # Determine page range
            total_pages = len(all_images)
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = total_pages

            # Validate range
            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))

            # Process each page
            pages = []
            warnings = []

            for i, image_bytes in enumerate(all_images[start_page - 1 : end_page], start=start_page):
                try:
                    page = await self.process_page(image_bytes, i)

                    # Check confidence threshold
                    if page.confidence < self.options.fallback_threshold:
                        warnings.append(
                            f"Page {i} has low confidence ({page.confidence:.2%})"
                        )

                    pages.append(page)

                except Exception as e:
                    logger.error(f"Failed to process page {i}: {e}")
                    warnings.append(f"Page {i} processing failed: {str(e)}")
                    # Create a placeholder page
                    pages.append(
                        OCRPage(
                            page_number=i,
                            text="",
                            confidence=0.0,
                            blocks=[],
                        )
                    )

            # Generate markdown
            markdown = self._generate_markdown(pages, self.options)

            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(pages)

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"YomiToku processed {len(pages)} pages: "
                f"{overall_confidence:.2%} confidence, {processing_time}ms"
            )

            return OCRResult(
                document_id=document_id,
                engine_used=self.engine_name,
                total_pages=len(pages),
                pages=pages,
                markdown=markdown,
                confidence=overall_confidence,
                processing_time_ms=processing_time,
                options=self.options,
                metadata={
                    "start_page": start_page,
                    "end_page": end_page,
                    "dpi": 200,
                    "version": self.version,
                },
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"YomiToku PDF processing failed: {e}")
            raise OCRProcessingError(
                f"Failed to process PDF: {str(e)}",
                details={
                    "document_id": document_id,
                    "error_type": type(e).__name__,
                },
            )
