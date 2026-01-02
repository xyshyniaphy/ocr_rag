#!/usr/bin/env python3
"""
Integration Tests for Document Upload API
Tests for backend/api/v1/documents.py upload endpoint
"""

import pytest
import os
from pathlib import Path
from httpx import AsyncClient


# Helper to get test PDF path
def get_test_pdf_path() -> Path:
    """Get path to test PDF file"""
    # When running in container, tests are mounted at /app/tests
    if Path("/app/tests/testdata/test.pdf").exists():
        return Path("/app/tests/testdata/test.pdf")
    # Fallback for local development
    return Path(__file__).parent.parent.parent / "testdata" / "test.pdf"


@pytest.mark.integration
class TestDocumentUpload:
    """Test document upload endpoint"""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful PDF upload"""
        test_pdf_path = get_test_pdf_path()

        # Skip if test file doesn't exist
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        # Read PDF file
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Prepare upload data
        files = {
            "file": ("test.pdf", pdf_content, "application/pdf")
        }
        data = {
            "title": "Test Document",
            "author": "Test Author",
            "keywords": "test,upload,pdf"
        }

        # Upload document
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files,
            data=data
        )

        # Assert successful upload
        assert response.status_code == 200, f"Upload failed: {response.text}"

        # Verify response structure
        result = response.json()
        assert "document_id" in result, "Response missing document_id"
        assert "status" in result, "Response missing status"
        assert "filename" in result, "Response missing filename"
        assert "file_size_bytes" in result, "Response missing file_size_bytes"

        # Verify values
        assert result["filename"] == "test.pdf"
        assert result["status"] == "pending"
        assert result["file_size_bytes"] == len(pdf_content)
        assert isinstance(result["document_id"], str)

    @pytest.mark.asyncio
    async def test_upload_pdf_no_metadata(self, client: AsyncClient, auth_headers: dict):
        """Test PDF upload without optional metadata"""
        # Get test PDF path
        test_pdf_path = get_test_pdf_path()

        # Skip if test file doesn't exist
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        # Read PDF file
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Upload without optional metadata
        files = {
            "file": ("test.pdf", pdf_content, "application/pdf")
        }

        # Upload document
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )

        # Assert successful upload
        assert response.status_code == 200, f"Upload failed: {response.text}"

        # Verify response
        result = response.json()
        assert "document_id" in result
        assert result["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_upload_unauthorized(self, client: AsyncClient):
        """Test upload without authentication fails"""
        # Get test PDF path
        test_pdf_path = get_test_pdf_path()

        # Skip if test file doesn't exist
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        # Read PDF file
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Try upload without auth headers
        files = {
            "file": ("test.pdf", pdf_content, "application/pdf")
        }

        # Upload should fail without auth
        response = client.post(
            "/api/v1/documents/upload",
            files=files
        )

        # Assert unauthorized
        assert response.status_code == 401, "Upload should require authentication"

    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, client: AsyncClient, auth_headers: dict):
        """Test upload with invalid file type fails"""
        # Create a text file (not PDF)
        files = {
            "file": ("test.txt", b"This is not a PDF", "text/plain")
        }

        # Upload should fail for non-PDF
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )

        # Assert bad request or unprocessable entity
        assert response.status_code in [400, 422], \
            f"Non-PDF upload should fail: {response.status_code}"

    @pytest.mark.asyncio
    async def test_upload_empty_file(self, client: AsyncClient, auth_headers: dict):
        """Test upload with empty file fails"""
        # Create empty PDF
        files = {
            "file": ("empty.pdf", b"", "application/pdf")
        }

        # Upload should fail for empty file
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )

        # Assert bad request
        assert response.status_code == 400, \
            f"Empty file upload should fail: {response.status_code}"

    @pytest.mark.asyncio
    async def test_duplicate_file_handling(self, client: AsyncClient, auth_headers: dict):
        """Test duplicate file upload handling"""
        # Get test PDF path
        test_pdf_path = get_test_pdf_path()

        # Skip if test file doesn't exist
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        # Read PDF file
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Upload first time
        files = {
            "file": ("test.pdf", pdf_content, "application/pdf")
        }
        response1 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )

        assert response1.status_code == 200
        first_doc_id = response1.json().get("document_id")

        # Upload same file second time
        response2 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )

        # Should succeed (might return existing or create new)
        assert response2.status_code == 200
        second_doc_id = response2.json().get("document_id")

        # Both should have valid document IDs
        assert first_doc_id is not None
        assert second_doc_id is not None
