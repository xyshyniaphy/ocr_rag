"""
PaddleOCR OCR Engine
PaddleOCR-VL as fallback for Japanese document OCR

PaddleOCR is a multilingual OCR engine that supports:
- 80+ languages including Japanese
- Layout analysis
- Table recognition
- Lightweight and fast
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


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR-VL Engine implementation

    Fallback OCR engine when YomiToku is not available
    or produces low confidence results.
    """

    # Model path configuration
    MODEL_PATH = "/app/models/paddleocr"

    def __init__(self, options: Optional[OCROptions] = None):
        """
        Initialize PaddleOCR engine

        Args:
            options: OCR processing options (uses defaults if None)
        """
        if options is None:
            options = OCROptions(
                engine="paddleocr",
                confidence_threshold=settings.OCR_CONFIDENCE_THRESHOLD,
                fallback_threshold=settings.OCR_FALLBACK_THRESHOLD,
            )

        super().__init__(options)
        self._ocr = None

    @property
    def engine_name(self) -> str:
        return "PaddleOCR"

    @property
    def version(self) -> str:
        """Return PaddleOCR version"""
        try:
            import paddleocr
            return paddleocr.__version__
        except (ImportError, AttributeError):
            return "unknown"

    async def load_model(self) -> None:
        """
        Load PaddleOCR model

        The model files are expected to be in /app/models/paddleocr/
        from the Docker base image.
        """
        if self._is_loaded:
            logger.debug("PaddleOCR model already loaded")
            return

        try:
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR with Japanese language support
            # New API (paddleocr >= 2.7): use lang parameter for language
            self._ocr = PaddleOCR(
                lang="japan",  # Japanese language
                # Note: GPU is automatically detected in new versions
                # Old parameters like use_angle_cls, use_gpu, show_log are deprecated
            )

            self._is_loaded = True
            logger.info("PaddleOCR model loaded successfully")

        except ImportError as e:
            raise OCRProcessingError(
                "PaddleOCR library not installed",
                details={"install": "pip install paddleocr paddlepaddle-gpu"},
            )
        except Exception as e:
            raise OCRProcessingError(
                f"Failed to load PaddleOCR model: {str(e)}",
                details={"model_path": self.MODEL_PATH, "error_type": type(e).__name__},
            )

    async def process_page(
        self,
        image_bytes: bytes,
        page_number: int,
        dpi: int = 200,
    ) -> OCRPage:
        """
        Process a single page image with PaddleOCR

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
            import numpy as np
            from PIL import Image

            start_time = time.time()

            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            # Run PaddleOCR
            # Returns list of [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
            result = self._ocr.ocr(image_np, cls=True)

            processing_time = int((time.time() - start_time) * 1000)

            # Parse PaddleOCR result
            blocks = []
            total_confidence = 0.0

            if result and result[0]:
                for item in result[0]:
                    # item structure: [bbox, (text, confidence)]
                    bbox_data = item[0]
                    text_info = item[1]

                    if not text_info:
                        continue

                    text = text_info[0]
                    confidence = float(text_info[1])

                    # Filter by confidence threshold
                    if not text or confidence < self.options.confidence_threshold:
                        continue

                    # Create bounding box from 4 points
                    if len(bbox_data) == 4:
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

                    blocks.append(
                        OCRBlock(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            block_type="text",  # PaddleOCR doesn't distinguish block types
                        )
                    )
                    total_confidence += confidence

            # Combine all text
            full_text = "\n".join(block.text for block in blocks)
            avg_confidence = total_confidence / len(blocks) if blocks else 0.0

            logger.debug(
                f"PaddleOCR processed page {page_number}: "
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
            logger.error(f"PaddleOCR page processing failed: {e}")
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
        Process a PDF document with PaddleOCR

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
                f"PaddleOCR processed {len(pages)} pages: "
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
            logger.error(f"PaddleOCR PDF processing failed: {e}")
            raise OCRProcessingError(
                f"Failed to process PDF: {str(e)}",
                details={
                    "document_id": document_id,
                    "error_type": type(e).__name__,
                },
            )
