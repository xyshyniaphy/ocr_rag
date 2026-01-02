"""
Conftest for API integration tests
Defines fixtures specific to API testing
"""

import pytest
import pytest_asyncio
import uuid
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Test HTTP client for FastAPI app (ASGI transport)"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
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
