#!/usr/bin/env python3
"""
Unit Tests for Cache Utilities
Tests for backend/core/cache.py
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from backend.core.cache import CacheManager, get_cache_manager


# Module-level fixture for cache manager (synchronous setup for reliability)
@pytest.fixture
def cache_manager():
    """Create a fresh cache manager for each test"""
    # Run the async setup in a new event loop
    async def setup_manager():
        manager = CacheManager()
        await manager.clear()
        return manager

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        manager = loop.run_until_complete(setup_manager())
        yield manager
    finally:
        loop.close()


class TestCacheManager:
    """Test CacheManager class"""

    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initializes correctly"""
        assert cache_manager is not None
        assert cache_manager._cache == {}
        assert cache_manager._lock is not None

    @pytest.mark.asyncio
    async def test_set_and_get_cache(self, cache_manager):
        """Test basic set and get operations"""
        key = "test_key"
        value = {"data": "test_value"}

        await cache_manager.set(key, value)
        retrieved = await cache_manager.get(key)

        assert retrieved == value

    @pytest.mark.asyncio
    async def test_get_nonexistent_key_returns_none(self, cache_manager):
        """Test getting a nonexistent key returns None"""
        result = await cache_manager.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager):
        """Test cache entries expire after TTL"""
        key = "expiring_key"
        value = "expiring_value"

        # Set with 1 second TTL
        await cache_manager.set(key, value, ttl=1)

        # Should be available immediately
        retrieved = await cache_manager.get(key)
        assert retrieved == value

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        retrieved = await cache_manager.get(key)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_existing_key(self, cache_manager):
        """Test deleting an existing cache entry"""
        key = "delete_key"
        value = "delete_value"

        await cache_manager.set(key, value)
        result = await cache_manager.delete(key)

        assert result is True
        retrieved = await cache_manager.get(key)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache_manager):
        """Test deleting a nonexistent key returns False"""
        result = await cache_manager.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_manager):
        """Test clearing all cache entries"""
        # Add multiple entries
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Clear all
        count = await cache_manager.clear()

        assert count == 3
        assert await cache_manager.get("key1") is None
        assert await cache_manager.get("key2") is None
        assert await cache_manager.get("key3") is None

    @pytest.mark.asyncio
    async def test_clear_empty_cache(self, cache_manager):
        """Test clearing an empty cache returns 0"""
        count = await cache_manager.clear()
        assert count == 0

    @pytest.mark.asyncio
    async def test_different_data_types(self, cache_manager):
        """Test caching different data types"""
        # String
        await cache_manager.set("string_key", "string_value")
        assert await cache_manager.get("string_key") == "string_value"

        # Integer
        await cache_manager.set("int_key", 42)
        assert await cache_manager.get("int_key") == 42

        # Float
        await cache_manager.set("float_key", 3.14)
        assert await cache_manager.get("float_key") == 3.14

        # Boolean
        await cache_manager.set("bool_key", True)
        assert await cache_manager.get("bool_key") is True

        # List
        await cache_manager.set("list_key", [1, 2, 3])
        assert await cache_manager.get("list_key") == [1, 2, 3]

        # Dict
        await cache_manager.set("dict_key", {"nested": "value"})
        assert await cache_manager.get("dict_key") == {"nested": "value"}

        # None
        await cache_manager.set("none_key", None)
        assert await cache_manager.get("none_key") is None

    @pytest.mark.asyncio
    async def test_default_ttl(self, cache_manager):
        """Test default TTL is applied"""
        key = "default_ttl_key"
        value = "default_ttl_value"

        await cache_manager.set(key, value)  # No TTL specified

        # Should be available immediately
        retrieved = await cache_manager.get(key)
        assert retrieved == value

        # Check default TTL is 3600 seconds (1 hour)
        # We can't wait that long, but we can verify it was set
        cache_entry = cache_manager._cache.get(key)
        assert cache_entry is not None

        # Check expiry is approximately 1 hour from now
        stored_value, expiry = cache_entry
        expected_expiry = datetime.now() + timedelta(seconds=3600)
        time_diff = abs((expiry - expected_expiry).total_seconds())
        assert time_diff < 1  # Less than 1 second difference

    @pytest.mark.asyncio
    async def test_cache_key_overwrite(self, cache_manager):
        """Test overwriting an existing cache key"""
        key = "overwrite_key"

        await cache_manager.set(key, "value1")
        assert await cache_manager.get(key) == "value1"

        await cache_manager.set(key, "value2")
        assert await cache_manager.get(key) == "value2"

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache_manager):
        """Test concurrent access to cache is thread-safe"""
        async def set_and_get(key, value):
            await cache_manager.set(key, value)
            return await cache_manager.get(key)

        # Run multiple concurrent operations
        tasks = [
            set_and_get(f"key_{i}", f"value_{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == f"value_{i}"

        # Verify all values are in cache
        for i in range(10):
            assert await cache_manager.get(f"key_{i}") == f"value_{i}"


class TestGetCacheManager:
    """Test get_cache_manager utility function"""

    @pytest.mark.asyncio
    async def test_get_cache_manager_returns_singleton(self):
        """Test get_cache_manager returns the same instance"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_get_cache_manager_returns_cache_manager(self):
        """Test get_cache_manager returns CacheManager instance"""
        manager = get_cache_manager()

        assert isinstance(manager, CacheManager)
