"""Create minimal test PDF for E2E testing"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path

def create_test_pdf(output_path: Path):
    """Create a minimal test PDF with Japanese text"""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Test Document for OCR RAG System")
    
    # Add content
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "This is a test document for end-to-end testing.")
    c.drawString(100, 710, "It contains multiple lines of text.")
    c.drawString(100, 690, "The document will be processed by OCR,")
    c.drawString(100, 670, "chunked, embedded, and indexed in Milvus.")
    
    # Add page numbers
    c.drawString(100, 650, f"Page 1 of 1")
    
    c.save()
    print(f"Created test PDF: {output_path}")

if __name__ == "__main__":
    output_path = Path("tests/testdata/test.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_test_pdf(output_path)
