"""
Cache Manager
Simple in-memory cache for embedding results
"""

import json
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

from backend.core.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Simple in-memory cache manager with TTL support
    """

    def __init__(self):
        """Initialize the cache manager"""
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
        logger.info("CacheManager initialized")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]

            # Check if expired
            if datetime.now() > expiry:
                del self._cache[key]
                logger.debug(f"Cache expired: {key}")
                return None

            logger.debug(f"Cache hit: {key}")
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
    ) -> None:
        """
        Set a value in the cache

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time to live in seconds (default: 1 hour)
        """
        async with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self._cache[key] = (value, expiry)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache deleted: {key}")
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cache entries

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries")
            return count

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache

        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key
                for key, (_, expiry) in self._cache.items()
                if now > expiry
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        async with self._lock:
            now = datetime.now()
            active_count = sum(
                1
                for _, expiry in self._cache.values()
                if now <= expiry
            )
            expired_count = len(self._cache) - active_count

            return {
                "total_entries": len(self._cache),
                "active_entries": active_count,
                "expired_entries": expired_count,
            }


# Global singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get or create the global cache manager instance

    Returns:
        CacheManager singleton
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Export for convenience
cache_manager = get_cache_manager()
