"""
Configuration for E2E tests
Fixtures for end-to-end testing
"""

import pytest
import pytest_asyncio
import uuid
import time
import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.db.session import init_db


@pytest_asyncio.fixture(autouse=True)
async def initialize_database():
    """Initialize database and storage for E2E tests"""
    await init_db()
    # Initialize MinIO storage client
    from backend.storage.client import init_minio
    try:
        await init_minio()
    except Exception as e:
        # MinIO might not be available in all test environments
        import warnings
        warnings.warn(f"MinIO initialization failed: {e}")


@pytest_asyncio.fixture
async def client():
    """Test HTTP client using ASGI transport"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient):
    """Authentication headers for API requests"""
    test_user = {
        "email": f"e2e_test_{uuid.uuid4().hex[:8]}@example.com",
        "password": "testpass123",
        "full_name": "E2E Test User"
    }

    # Register user
    response = await client.post(
        "/api/v1/auth/register",
        json=test_user
    )

    # If registration fails, try login
    if response.status_code not in [200, 201]:
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )

    if response.status_code in [200, 201]:
        token = response.json().get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}

    return {}


@pytest_asyncio.fixture
def test_pdf_bytes():
    """Generate minimal test PDF bytes"""
    # Minimal valid PDF (PDF 1.4 specification)
    pdf_template = """%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/MediaBox [0 0 612 792]
/Contents 5 0 R
>>
endobj
4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
5 0 obj
<<
/Length 200
>>
stream
BT
/F1 12 Tf
50 700 Td
(Test Document for OCR RAG System) Tj
0 -14 Td
(This is a test document for end-to-end testing.) Tj
0 -14 Td
(It contains multiple lines of text.) Tj
0 -14 Td
(The document will be processed by OCR.) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000345 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
456
%%EOF
"""
    return pdf_template.encode('latin-1')


async def wait_for_document_completion(
    client: AsyncClient,
    document_id: str,
    headers: dict,
    timeout: int = 300,
    poll_interval: int = 3
) -> dict:
    """
    Wait for document processing to complete

    Args:
        client: HTTP client
        document_id: Document ID to poll
        headers: Authentication headers
        timeout: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds

    Returns:
        Final status data

    Raises:
        TimeoutError: If processing doesn't complete in time
        Exception: If processing fails
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = await client.get(
            f"/api/v1/documents/{document_id}/status",
            headers=headers
        )

        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)

            if status == "completed":
                return status_data
            elif status == "failed":
                raise Exception(f"Document processing failed: {status_data}")

            # Still processing, log progress
            print(f"Document {document_id}: {status} ({progress}%)")

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Document processing timed out after {timeout}s")


async def clear_all_documents(client: AsyncClient, headers: dict):
    """Clear all documents from the system"""
    response = await client.delete(
        "/api/v1/documents/all",
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Cleared {data.get('deleted_count', 0)} documents")
    else:
        print(f"Clear documents returned {response.status_code}")


async def verify_document_in_query(
    client: AsyncClient,
    document_id: str,
    headers: dict
) -> bool:
    """Verify that document appears in query results"""
    response = await client.post(
        "/api/v1/query",
        headers=headers,
        json={
            "query": "test document",
            "top_k": 5,
            "rerank": False,
            "language": "en"
        }
    )

    if response.status_code == 200:
        result = response.json()
        sources = result.get("sources", [])
        # Check if any source is from our document
        return any(
            source.get("document_id") == document_id
            for source in sources
        )

    return False


async def verify_no_documents(client: AsyncClient, headers: dict) -> bool:
    """Verify that no documents exist in the system"""
    response = await client.get(
        "/api/v1/documents",
        headers=headers,
        params={"limit": 100}
    )

    if response.status_code == 200:
        data = response.json()
        return data.get("total", 0) == 0

    return False
