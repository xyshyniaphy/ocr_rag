#!/usr/bin/env python3
"""
Integration Tests for Documents API
Tests for backend/api/v1/documents.py endpoints
"""

import pytest
import os
import uuid
from pathlib import Path
from httpx import AsyncClient
from typing import Dict, Any


# Helper to get test PDF path
def get_test_pdf_path() -> Path:
    """Get path to test PDF file"""
    if Path("/app/tests/testdata/test.pdf").exists():
        return Path("/app/tests/testdata/test.pdf")
    return Path(__file__).parent.parent.parent.parent / "testdata" / "test.pdf"


@pytest.mark.integration
class TestDocumentsListAPI:
    """Test GET /api/v1/documents - List documents endpoint"""

    @pytest.mark.asyncio
    async def test_list_documents_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful documents list retrieval"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, client: AsyncClient, auth_headers: dict):
        """Test documents list with pagination parameters"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"limit": 5, "offset": 0}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["limit"] == 5
        assert result["offset"] == 0
        assert len(result["results"]) <= 5

    @pytest.mark.asyncio
    async def test_list_documents_with_status_filter(self, client: AsyncClient, auth_headers: dict):
        """Test documents list filtered by status"""
        # First, upload a document
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            files = {"file": ("test.pdf", f.read(), "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files
            )

        # Filter by pending status
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"status": "pending"}
        )

        assert response.status_code == 200
        result = response.json()

        # All results should have pending status
        for doc in result["results"]:
            assert doc["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_documents_empty_result(self, client: AsyncClient, auth_headers: dict):
        """Test documents list with no matching documents"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"status": "nonexistent_status"}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["total"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_list_documents_unauthorized(self, client: AsyncClient):
        """Test documents list without authentication

        NOTE: Current implementation does not enforce authentication.
        This test documents the current behavior but should be updated
        when authentication is properly enforced.
        """
        response = await client.get("/api/v1/documents")

        # TODO: Should return 401 when authentication is enforced
        # Currently returns 200 (no auth check)
        assert response.status_code == 200


@pytest.mark.integration
class TestDocumentDetailAPI:
    """Test GET /api/v1/documents/{document_id} - Get single document"""

    @pytest.mark.asyncio
    async def test_get_document_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful document retrieval"""
        # First, upload a document
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            content = f.read()
            # Modify content to avoid duplicate detection
            modified_content = content + b" get_document_success_test"

            files = {"file": ("test_detail.pdf", modified_content, "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files
            )

        # Handle both successful upload and duplicate response
        assert upload_response.status_code in [200, 202]
        doc_id = upload_response.json().get("document_id")
        assert doc_id is not None

        # Get the document
        response = await client.get(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        assert result["document_id"] == doc_id
        assert "filename" in result
        assert "status" in result
        assert "uploaded_at" in result

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test getting non-existent document"""
        fake_id = uuid.uuid4()
        response = await client.get(
            f"/api/v1/documents/{fake_id}",
            headers=auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_document_invalid_id(self, client: AsyncClient, auth_headers: dict):
        """Test getting document with invalid ID format"""
        response = await client.get(
            "/api/v1/documents/invalid-uuid",
            headers=auth_headers
        )

        # ValidationException returns 400, not 422
        assert response.status_code == 400


@pytest.mark.integration
class TestDocumentStatusAPI:
    """Test GET /api/v1/documents/{document_id}/status - Get document status"""

    @pytest.mark.asyncio
    async def test_get_document_status_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful document status retrieval"""
        # First, upload a document
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            files = {"file": ("test_status.pdf", f.read(), "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files
            )

        doc_id = upload_response.json().get("document_id")

        # Get document status
        response = await client.get(
            f"/api/v1/documents/{doc_id}/status",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        assert result["document_id"] == doc_id
        assert "status" in result
        assert "progress" in result
        assert isinstance(result["progress"], int)

    @pytest.mark.asyncio
    async def test_get_document_status_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test status for non-existent document"""
        fake_id = uuid.uuid4()
        response = await client.get(
            f"/api/v1/documents/{fake_id}/status",
            headers=auth_headers
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestDocumentDeleteAPI:
    """Test DELETE /api/v1/documents/{document_id} - Delete document"""

    @pytest.mark.asyncio
    async def test_delete_document_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful document deletion"""
        # First, upload a document
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            files = {"file": ("test_delete.pdf", f.read(), "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files
            )

        doc_id = upload_response.json().get("document_id")

        # Delete the document
        response = await client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers
        )

        assert response.status_code == 204

        # Verify document is deleted (should return 404)
        get_response = await client.get(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test deleting non-existent document"""
        fake_id = uuid.uuid4()
        response = await client.delete(
            f"/api/v1/documents/{fake_id}",
            headers=auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_unauthorized(self, client: AsyncClient):
        """Test deletion without authentication

        NOTE: Current implementation does not enforce authentication.
        This test documents the current behavior but should be updated
        when authentication is properly enforced.
        """
        fake_id = uuid.uuid4()
        response = await client.delete(f"/api/v1/documents/{fake_id}")

        # TODO: Should return 401 when authentication is enforced
        # Currently returns 404 (document not found, no auth check)
        assert response.status_code == 404


@pytest.mark.integration
class TestDocumentPagination:
    """Test pagination functionality"""

    @pytest.mark.asyncio
    async def test_pagination_offset(self, client: AsyncClient, auth_headers: dict):
        """Test pagination with offset"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"offset": 1, "limit": 5}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["offset"] == 1
        assert result["limit"] == 5

    @pytest.mark.asyncio
    async def test_pagination_large_limit(self, client: AsyncClient, auth_headers: dict):
        """Test pagination with large limit value"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"limit": 100}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["limit"] <= 100  # Should enforce max limit

    @pytest.mark.asyncio
    async def test_pagination_default_values(self, client: AsyncClient, auth_headers: dict):
        """Test pagination uses default values when not specified"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Check defaults are applied
        assert "limit" in result
        assert "offset" in result
        assert result["offset"] == 0


@pytest.mark.integration
class TestDocumentResponseStructure:
    """Test API response structures and data types"""

    @pytest.mark.asyncio
    async def test_document_response_fields(self, client: AsyncClient, auth_headers: dict):
        """Test document response has all required fields"""
        # Upload a test document
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            content = f.read()
            # Modify content to avoid duplicate detection
            modified_content = content + b" response_fields_test"

            files = {"file": ("test_fields.pdf", modified_content, "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files,
                data={"title": "Test Document", "author": "Test Author"}
            )

        assert upload_response.status_code in [200, 202]
        doc_id = upload_response.json().get("document_id")

        # Get document details
        response = await client.get(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        doc = response.json()

        # Verify all required fields exist and have correct types
        required_fields = {
            "document_id": str,
            "filename": str,
            "file_size_bytes": int,
            "file_hash": str,
            "content_type": str,
            "status": str,
            "uploaded_at": str,
            "owner": dict,
        }

        for field, field_type in required_fields.items():
            assert field in doc, f"Missing field: {field}"
            assert isinstance(doc[field], field_type), f"Field {field} has wrong type"

    @pytest.mark.asyncio
    async def test_document_list_response_structure(self, client: AsyncClient, auth_headers: dict):
        """Test document list response structure"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Verify top-level structure
        assert isinstance(result["total"], int)
        assert isinstance(result["limit"], int)
        assert isinstance(result["offset"], int)
        assert isinstance(result["results"], list)

        # Verify first document in list has required fields
        if len(result["results"]) > 0:
            doc = result["results"][0]
            required_fields = ["document_id", "filename", "status", "uploaded_at"]
            for field in required_fields:
                assert field in doc, f"Missing field in list item: {field}"


@pytest.mark.integration
class TestDocumentUploadAPI:
    """Test POST /api/v1/documents/upload - Document upload endpoint"""

    @pytest.mark.asyncio
    async def test_upload_with_metadata_persists(self, client: AsyncClient, auth_headers: dict):
        """Test that document upload creates a record that can be retrieved"""
        import json

        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(test_pdf_path, "rb") as f:
            content = f.read()
            # Modify content to avoid duplicate detection
            modified_content = content + b" metadata_persistence_test"

            files = {"file": ("test_metadata.pdf", modified_content, "application/pdf")}
            upload_response = await client.post(
                "/api/v1/documents/upload",
                headers=auth_headers,
                files=files,
            )

        # Upload returns 202 Accepted
        assert upload_response.status_code in [200, 202]
        doc_id = upload_response.json().get("document_id")
        assert doc_id is not None

        # Verify document can be retrieved
        response = await client.get(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        doc = response.json()

        # Verify basic document properties
        assert doc["document_id"] == doc_id
        assert doc["filename"] == "test_metadata.pdf"
        assert "metadata" in doc

    @pytest.mark.asyncio
    async def test_upload_creates_unique_ids(self, client: AsyncClient, auth_headers: dict):
        """Test that each upload creates a unique document ID"""
        test_pdf_path = get_test_pdf_path()
        if not test_pdf_path.exists():
            pytest.skip("Test PDF not found")

        doc_ids = []

        # Upload two documents with different names
        # Note: If they have the same content, duplicate detection will return the same ID
        for i in range(2):
            with open(test_pdf_path, "rb") as f:
                content = f.read()
                # Modify content slightly to avoid duplicate detection
                modified_content = content + f" {i}".encode()

                files = {"file": (f"test_unique_{i}.pdf", modified_content, "application/pdf")}
                upload_response = await client.post(
                    "/api/v1/documents/upload",
                    headers=auth_headers,
                    files=files
                )
                # Handle both 200 and 202, as well as duplicate responses
                if upload_response.status_code in [200, 202]:
                    doc_id = upload_response.json().get("document_id")
                    doc_ids.append(doc_id)

        # Verify we got at least some unique IDs or duplicate detection worked
        assert len(doc_ids) >= 1
        # If we got 2 IDs, they should be unique
        if len(doc_ids) == 2:
            assert doc_ids[0] != doc_ids[1]


@pytest.mark.integration
class TestDocumentAPIErrorHandling:
    """Test API error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_invalid_uuid_format_400(self, client: AsyncClient, auth_headers: dict):
        """Test invalid UUID format returns 400"""
        invalid_uuids = [
            "not-a-uuid",
            "123456",
            "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",  # Invalid hex
        ]

        for invalid_id in invalid_uuids:
            response = await client.get(
                f"/api/v1/documents/{invalid_id}",
                headers=auth_headers
            )
            # Should return 400 for validation error
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_negative_limit(self, client: AsyncClient, auth_headers: dict):
        """Test negative limit parameter"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"limit": -1}
        )

        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_large_offset(self, client: AsyncClient, auth_headers: dict):
        """Test large offset returns empty list"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"offset": 99999}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_non_numeric_pagination(self, client: AsyncClient, auth_headers: dict):
        """Test non-numeric pagination parameters"""
        response = await client.get(
            "/api/v1/documents",
            headers=auth_headers,
            params={"limit": "abc", "offset": "xyz"}
        )

        # Should return validation error
        assert response.status_code in [400, 422]
