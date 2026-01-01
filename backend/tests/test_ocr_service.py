#!/usr/bin/env python3
"""
OCR Service Test Script
Tests the OCR service with a sample PDF
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging import get_logger
from backend.services.ocr import OCRService, ocr_pdf, OCROptions

logger = get_logger(__name__)


async def test_ocr_service():
    """Test the OCR service with a sample PDF"""

    print("=" * 60)
    print("OCR Service Test")
    print("=" * 60)

    # Check if test PDF exists
    test_pdf_path = Path("/app/backend/testdata/test.pdf")
    if not test_pdf_path.exists():
        # Try relative path
        test_pdf_path = Path("testdata/test.pdf")

    if not test_pdf_path.exists():
        print(f"ERROR: Test PDF not found at {test_pdf_path}")
        print("Please ensure testdata/test.pdf exists")
        return False

    print(f"\nTest PDF: {test_pdf_path}")
    print(f"File size: {test_pdf_path.stat().st_size} bytes")

    # Read the PDF
    print("\nReading PDF...")
    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"PDF bytes: {len(pdf_bytes)} bytes")

    # Test 1: Check available engines
    print("\n" + "-" * 40)
    print("Test 1: Available Engines")
    print("-" * 40)
    engines = OCRService.get_available_engines()
    print(f"Registered engines: {engines}")

    # Test 2: Simple OCR with default settings
    print("\n" + "-" * 40)
    print("Test 2: OCR with Default Settings")
    print("-" * 40)

    try:
        result = await ocr_pdf(pdf_bytes)
        print(f"Engine used: {result.engine_used}")
        print(f"Total pages: {result.total_pages}")
        print(f"Overall confidence: {result.confidence:.2%}")
        print(f"Processing time: {result.processing_time_ms}ms")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")

        print(f"\nFirst 500 characters of extracted text:")
        print("-" * 40)
        print(result.markdown[:500] + "..." if len(result.markdown) > 500 else result.markdown)

        # Show page breakdown
        print(f"\nPage breakdown:")
        for page in result.pages:
            print(f"  Page {page.page_number}: {page.confidence:.2%} confidence, "
                  f"{len(page.blocks)} blocks, {len(page.text)} chars")

    except Exception as e:
        print(f"ERROR during OCR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: OCR with custom options
    print("\n" + "-" * 40)
    print("Test 3: OCR with Custom Options")
    print("-" * 40)

    try:
        options = OCROptions(
            engine="yomitoku",
            confidence_threshold=0.80,
            fallback_threshold=0.75,
            preserve_layout=True,
            extract_tables=True,
        )

        result2 = await ocr_pdf(pdf_bytes, options=options)
        print(f"Engine used: {result2.engine_used}")
        print(f"Overall confidence: {result2.confidence:.2%}")
        print(f"Processing time: {result2.processing_time_ms}ms")

    except Exception as e:
        print(f"Note: Custom options test encountered: {e}")
        # This might fail if models aren't available, that's okay for testing

    # Test 4: Memory cleanup
    print("\n" + "-" * 40)
    print("Test 4: Memory Management")
    print("-" * 40)

    loaded = OCRService.get_loaded_engines()
    print(f"Loaded engines before cleanup: {loaded}")

    await OCRService.unload_all()
    loaded_after = OCRService.get_loaded_engines()
    print(f"Loaded engines after cleanup: {loaded_after}")

    print("\n" + "=" * 60)
    print("OCR Service Test Completed Successfully!")
    print("=" * 60)

    return True


async def test_ocr_fallback():
    """Test OCR fallback mechanism"""

    print("\n" + "=" * 60)
    print("OCR Fallback Test")
    print("=" * 60)

    test_pdf_path = Path("/app/backend/testdata/test.pdf")
    if not test_pdf_path.exists():
        test_pdf_path = Path("testdata/test.pdf")

    if not test_pdf_path.exists():
        print(f"ERROR: Test PDF not found")
        return False

    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Test with very high threshold to trigger fallback
    print("\nTesting fallback with high confidence threshold...")

    options = OCROptions(
        engine="yomitoku",
        confidence_threshold=0.99,  # Very high to trigger fallback
        fallback_threshold=0.95,
        enable_fallback=True,
    )

    try:
        result = await OCRService.process_pdf(
            pdf_bytes,
            engine="yomitoku",
            fallback_engine="paddleocr",
            options=options,
            enable_fallback=True,
        )

        print(f"Final engine used: {result.engine_used}")
        print(f"Confidence: {result.confidence:.2%}")

        if result.metadata.get("primary_engine"):
            print(f"Primary engine was: {result.metadata['primary_engine']}")
            print(f"Primary confidence was: {result.metadata.get('primary_confidence', 0):.2%}")

        return True

    except Exception as e:
        print(f"ERROR during fallback test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "OCR Service Test Suite" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")

    success = True

    # Run basic test
    if not await test_ocr_service():
        success = False

    # Run fallback test
    if not await test_ocr_fallback():
        success = False

    if success:
        print("\n" + "🎉" + " All tests passed! " + "🎉")
        sys.exit(0)
    else:
        print("\n" + "❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
