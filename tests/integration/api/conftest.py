"""
Conftest for API integration tests
Defines fixtures specific to API testing

Note: These tests connect to a running server at http://localhost:8000
instead of using ASGI transport to avoid database initialization issues.
"""

import pytest
import pytest_asyncio
import uuid
import os
from httpx import AsyncClient

# Server URL from environment or default to localhost
SERVER_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Test HTTP client that connects to running server"""
    async with AsyncClient(base_url=SERVER_URL) as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict:
    """Authentication headers for API requests using admin user"""
    # Try to login with admin user
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "admin@example.com",
            "password": "admin123"
        }
    )

    if response.status_code == 200:
        token = response.json().get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}

    # Return empty headers if auth fails
    return {}
