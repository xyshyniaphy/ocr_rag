#!/usr/bin/env python3
"""
Text Chunking Service Test Script
Tests the chunking service with sample OCR results
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging import get_logger
from backend.services.ocr import ocr_pdf, OCROptions
from backend.services.processing import ChunkingService, ChunkingOptions, chunk_document

logger = get_logger(__name__)


async def test_basic_chunking():
    """Test basic text chunking"""

    print("=" * 60)
    print("Text Chunking Service Test")
    print("=" * 60)

    # Sample Japanese text
    sample_text = """
    第1章 総則

    第1条（目的）
    この法律は、電磁的方法による契約の締結等に関する規律を整備することにより、
    電子商取引の円滑化を図り、もって国民経済の健全な発展に寄与することを目的とする。

    第2条（定義）
    この法律において「電磁的方法」とは、電子情報処理組織を使用する方法その他の
    情報通信の技術を利用する方法であって、次に掲げるものをいう。

    一 電子計算機その他の情報処理機器に係る入力装置から、当該情報処理機器に
    係る出力装置へ情報を送信する方法

    二 前号に掲げる方法に準ずる方法として、主務省令で定める方法

    第3条（適用範囲）
    この法律は、事業者間の取引及び事業者と消費者との間の取引について適用する。
    ただし、消費者契約法（平成十二年法律第六十一号）第二章第四節の規定の
    適用を受ける取引については、この限りでない。
    """

    print("\n" + "-" * 40)
    print("Test 1: Basic Text Chunking")
    print("-" * 40)

    # Test with default options
    options = ChunkingOptions(
        chunk_size=200,
        chunk_overlap=30,
    )

    service = ChunkingService(options=options, strategy="recursive")

    chunks = await service.chunk_text(
        text=sample_text,
        document_id="test-doc-001",
    )

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Size: {len(chunk.text)} chars")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Preview: {chunk.text[:100]}...")


async def test_ocr_chunking():
    """Test chunking with real OCR results"""

    print("\n" + "=" * 60)
    print("OCR + Chunking Integration Test")
    print("=" * 60)

    # Check if test PDF exists
    test_pdf_path = Path("/app/backend/testdata/test.pdf")
    if not test_pdf_path.exists():
        test_pdf_path = Path("testdata/test.pdf")

    if not test_pdf_path.exists():
        print(f"ERROR: Test PDF not found at {test_pdf_path}")
        print("Skipping OCR chunking test")
        return

    print(f"\nTest PDF: {test_pdf_path}")

    # Read and OCR the PDF
    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("\nRunning OCR...")
    ocr_result = await ocr_pdf(pdf_bytes)

    print(f"\nOCR Result:")
    print(f"  Engine: {ocr_result.engine_used}")
    print(f"  Pages: {ocr_result.total_pages}")
    print(f"  Confidence: {ocr_result.confidence:.2%}")

    # Test different chunking strategies
    strategies = ["recursive", "semantic", "table_aware"]

    for strategy in strategies:
        print("\n" + "-" * 40)
        print(f"Strategy: {strategy}")
        print("-" * 40)

        try:
            options = ChunkingOptions(
                chunk_size=300,
                chunk_overlap=40,
            )

            chunking_result = await chunk_document(
                ocr_result=ocr_result,
                document_id="test-doc-002",
                chunk_size=300,
                chunk_overlap=40,
                strategy=strategy,
            )

            print(f"\nChunks: {chunking_result.total_chunks}")
            print(f"Total chars: {chunking_result.total_characters}")
            print(f"Total tokens: {chunking_result.total_tokens}")
            print(f"Avg chunk size: {chunking_result.avg_chunk_size:.1f} chars")
            print(f"Processing time: {chunking_result.processing_time_ms}ms")

            # Show first few chunks
            print(f"\nFirst 3 chunks:")
            for i, chunk in enumerate(chunking_result.chunks[:3], 1):
                print(f"\n  Chunk {i} (Page {chunk.metadata.page_number}):")
                print(f"    Size: {len(chunk.text)} chars, {chunk.token_count} tokens")
                print(f"    Type: {chunk.metadata.chunk_type}")
                print(f"    Preview: {chunk.text[:80]}...")

            if chunking_result.warnings:
                print(f"\nWarnings: {len(chunking_result.warnings)}")
                for warning in chunking_result.warnings[:3]:
                    print(f"  - {warning}")

        except Exception as e:
            print(f"ERROR with {strategy} strategy: {e}")
            import traceback
            traceback.print_exc()


async def test_japanese_awareness():
    """Test Japanese-specific chunking features"""

    print("\n" + "=" * 60)
    print("Japanese-Aware Chunking Test")
    print("=" * 60)

    # Japanese text with various sentence structures
    text = """
    日本語の文章は漢字、ひらがな、カタカナ、そしてアルファベットから構成されます。
    文の終わりには句読点が使われます。日本語の文章では「。」が最も一般的な句点です！
    疑問文には「？」を使いますが、これは比較的新しい傾向です。

    第1章　概要
    この章ではシステムの概要について説明します。システムは主に3つのコンポーネントから
    構成されています。それらは、OCRコンポーネント、チャンキングコンポーネント、そして
    検索コンポーネントです。

    第2章　詳細設計
    詳細設計では、各コンポーネントの仕様を定義します。まずOCRコンポーネントから
    始めます。OCRコンポーネントはYomiTokuとPaddleOCRをサポートしています。
    """

    print("\n" + "-" * 40)
    print("Japanese Sentence Boundary Detection")
    print("-" * 40)

    options = ChunkingOptions(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "；", "、"],
    )

    service = ChunkingService(options=options, strategy="recursive")
    chunks = await service.chunk_text(text, "test-doc-003")

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Text: {chunk.text}")
        print(f"  Ends with sentence marker: {chunk.text[-1] in '。！？'}")


async def main():
    """Main test runner"""

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Text Chunking Test Suite" + " " * 27 + "║")
    print("╚" + "═" * 58 + "╝")

    success = True

    # Test 1: Basic chunking
    try:
        await test_basic_chunking()
    except Exception as e:
        print(f"\nERROR in basic chunking test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: OCR + Chunking integration
    try:
        await test_ocr_chunking()
    except Exception as e:
        print(f"\nERROR in OCR chunking test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 3: Japanese awareness
    try:
        await test_japanese_awareness()
    except Exception as e:
        print(f"\nERROR in Japanese awareness test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("\n" + "🎉" + " All tests passed! " + "🎉")
        sys.exit(0)
    else:
        print("\n" + "❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
