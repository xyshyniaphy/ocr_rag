"""
OCR Service
Main OCR service with engine management and fallback logic
"""

import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.ocr.models import (
    OCRResult,
    OCROptions,
    OCREngineNotFoundError,
    OCRConfidenceError,
)
from backend.services.ocr.base import BaseOCREngine

logger = get_logger(__name__)


class OCRService:
    """
    Main OCR service that manages OCR engines

    Features:
    - Engine selection (YomiToku, PaddleOCR)
    - Automatic fallback on low confidence
    - GPU memory management
    - Batch processing
    """

    # Engine registry
    _engines: Dict[str, type[BaseOCREngine]] = {}
    _instances: Dict[str, BaseOCREngine] = {}
    _lock = asyncio.Lock()

    @classmethod
    def register_engine(cls, name: str, engine_class: type[BaseOCREngine]) -> None:
        """
        Register an OCR engine

        Args:
            name: Engine name
            engine_class: Engine class (must inherit from BaseOCREngine)
        """
        cls._engines[name] = engine_class
        logger.info(f"Registered OCR engine: {name}")

    @classmethod
    async def get_engine(cls, name: str, options: Optional[OCROptions] = None) -> BaseOCREngine:
        """
        Get or create an OCR engine instance

        Args:
            name: Engine name ('yomitoku' or 'paddleocr')
            options: OCR options

        Returns:
            BaseOCREngine instance

        Raises:
            OCREngineNotFoundError: If engine is not registered
        """
        async with cls._lock:
            # Return existing instance if available
            if name in cls._instances:
                return cls._instances[name]

            # Check if engine is registered
            if name not in cls._engines:
                raise OCREngineNotFoundError(
                    f"OCR engine '{name}' not registered",
                    details={"available_engines": list(cls._engines.keys())},
                )

            # Create new instance
            engine_class = cls._engines[name]
            instance = engine_class(options)

            # Load the model
            await instance.load_model()

            # Cache the instance
            cls._instances[name] = instance
            logger.info(f"Created OCR engine instance: {name}")

            return instance

    @classmethod
    async def process_pdf(
        cls,
        pdf_bytes: bytes,
        engine: str = "yomitoku",
        fallback_engine: str = "paddleocr",
        options: Optional[OCROptions] = None,
        enable_fallback: bool = True,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> OCRResult:
        """
        Process a PDF with automatic fallback

        Args:
            pdf_bytes: Raw PDF bytes
            engine: Primary OCR engine
            fallback_engine: Fallback OCR engine
            options: OCR options
            enable_fallback: Enable fallback on low confidence
            start_page: First page to process
            end_page: Last page to process

        Returns:
            OCRResult with extracted text

        Raises:
            OCREngineNotFoundError: If no engines are available
        """
        if options is None:
            options = OCROptions(
                engine=engine,
                confidence_threshold=settings.OCR_CONFIDENCE_THRESHOLD,
                fallback_threshold=settings.OCR_FALLBACK_THRESHOLD,
                enable_fallback=enable_fallback,
            )

        # Try primary engine
        try:
            primary = await cls.get_engine(engine, options)
            result = await primary.process_pdf(pdf_bytes, start_page, end_page)

            # Check if fallback is needed
            if (
                enable_fallback
                and result.confidence < options.fallback_threshold
                and engine != fallback_engine
            ):
                logger.warning(
                    f"Primary engine {engine} confidence {result.confidence:.2%} "
                    f"below threshold {options.fallback_threshold:.2%}, "
                    f"trying fallback {fallback_engine}"
                )

                # Try fallback engine
                fallback = await cls.get_engine(fallback_engine, options)
                fallback_result = await fallback.process_pdf(pdf_bytes, start_page, end_page)

                # Use fallback if it has better confidence
                if fallback_result.confidence > result.confidence:
                    logger.info(
                        f"Using fallback engine {fallback_engine} "
                        f"({fallback_result.confidence:.2%} > {result.confidence:.2%})"
                    )
                    fallback_result.metadata["primary_engine"] = engine
                    fallback_result.metadata["primary_confidence"] = result.confidence
                    fallback_result.warnings.append(
                        f"Primary engine {engine} had low confidence, "
                        f"used {fallback_engine} instead"
                    )
                    return fallback_result

            return result

        except Exception as e:
            logger.error(f"Primary engine {engine} failed: {e}")

            # Try fallback as last resort
            if enable_fallback and engine != fallback_engine:
                try:
                    logger.info(f"Attempting fallback to {fallback_engine}")
                    fallback = await cls.get_engine(fallback_engine, options)
                    result = await fallback.process_pdf(pdf_bytes, start_page, end_page)
                    result.warnings.append(f"Primary engine {engine} failed, used fallback")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback engine {fallback_engine} also failed: {fallback_error}")

            raise

    @classmethod
    async def process_page(
        cls,
        image_bytes: bytes,
        page_number: int,
        engine: str = "yomitoku",
        options: Optional[OCROptions] = None,
    ) -> Any:
        """
        Process a single page image

        Args:
            image_bytes: Raw image bytes
            page_number: Page number
            engine: OCR engine to use
            options: OCR options

        Returns:
            OCRPage result
        """
        if options is None:
            options = OCROptions(
                engine=engine,
                confidence_threshold=settings.OCR_CONFIDENCE_THRESHOLD,
            )

        ocr_engine = await cls.get_engine(engine, options)
        return await ocr_engine.process_page(image_bytes, page_number)

    @classmethod
    async def unload_engine(cls, name: str) -> None:
        """
        Unload an OCR engine and free memory

        Args:
            name: Engine name
        """
        async with cls._lock:
            if name in cls._instances:
                await cls._instances[name].unload()
                del cls._instances[name]
                logger.info(f"Unloaded OCR engine: {name}")

    @classmethod
    async def unload_all(cls) -> None:
        """Unload all OCR engines and free memory"""
        async with cls._lock:
            for name, instance in cls._instances.items():
                try:
                    await instance.unload()
                except Exception as e:
                    logger.error(f"Error unloading {name}: {e}")
            cls._instances.clear()
            logger.info("Unloaded all OCR engines")

    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of registered engine names"""
        return list(cls._engines.keys())

    @classmethod
    def get_loaded_engines(cls) -> List[str]:
        """Get list of loaded engine names"""
        return list(cls._instances.keys())


# Register engines
from backend.services.ocr.yomitoku import YomiTokuOCREngine
from backend.services.ocr.paddleocr import PaddleOCREngine

OCRService.register_engine("yomitoku", YomiTokuOCREngine)
OCRService.register_engine("paddleocr", PaddleOCREngine)


# Convenience function for simple usage
async def ocr_pdf(
    pdf_bytes: bytes,
    engine: str = None,
    options: OCROptions = None,
) -> OCRResult:
    """
    Convenience function to OCR a PDF

    Args:
        pdf_bytes: Raw PDF bytes
        engine: OCR engine (uses default from config if None)
        options: OCR options

    Returns:
        OCRResult with extracted text
    """
    if engine is None:
        engine = settings.OCR_ENGINE

    return await OCRService.process_pdf(
        pdf_bytes=pdf_bytes,
        engine=engine,
        options=options,
    )


__all__ = [
    "OCRService",
    "ocr_pdf",
    "YomiTokuOCREngine",
    "PaddleOCREngine",
]
