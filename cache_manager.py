#!/usr/bin/env python3
"""
Simple and Efficient Cache Manager
Lightweight in-memory and disk caching with LRU eviction
"""

import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from collections import OrderedDict
import threading
import pickle

logger = logging.getLogger(__name__)


class MemoryCache:
    """Simple LRU memory cache with thread safety.

    Provides O(1) average case get/set operations with automatic
    eviction of least recently used items when capacity is reached.

    Attributes:
        max_size: Maximum number of items to cache
        ttl_seconds: Time-to-live for cached items in seconds
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = 3600):
        """Initialize memory cache.

        Args:
            max_size: Maximum cache size (default: 1000)
            ttl_seconds: TTL in seconds (default: 3600, None for no expiry)
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return value

    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)

    def delete(self, key: str):
        """Delete key from cache"""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        """Clear all cache"""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": round(hit_rate, 2)
            }


class DiskCache:
    """SQLite-based disk cache for persistence"""

    def __init__(self, cache_dir: str, ttl_seconds: Optional[int] = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    size INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
            conn.commit()

    def _hash_key(self, key: str) -> str:
        """Hash key for storage"""
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        hashed_key = self._hash_key(key)

        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT value, timestamp FROM cache WHERE key = ?",
                    (hashed_key,)
                ).fetchone()

                if not result:
                    return None

                value_blob, timestamp = result

                # Check TTL
                if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
                    conn.execute("DELETE FROM cache WHERE key = ?", (hashed_key,))
                    conn.commit()
                    return None

                return pickle.loads(value_blob)

        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            return None

    def set(self, key: str, value: Any):
        """Set value in disk cache"""
        hashed_key = self._hash_key(key)

        try:
            value_blob = pickle.dumps(value)
            size = len(value_blob)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache (key, value, timestamp, size)
                    VALUES (?, ?, ?, ?)
                """, (hashed_key, value_blob, time.time(), size))
                conn.commit()

        except Exception as e:
            logger.error(f"Disk cache set error: {e}")

    def delete(self, key: str):
        """Delete key from cache"""
        hashed_key = self._hash_key(key)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (hashed_key,))
                conn.commit()
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")

    def clear(self):
        """Clear all cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")

    def cleanup_expired(self):
        """Remove expired entries"""
        if not self.ttl_seconds:
            return 0

        cutoff_time = time.time() - self.ttl_seconds

        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "DELETE FROM cache WHERE timestamp < ?",
                    (cutoff_time,)
                )
                removed = result.rowcount
                conn.commit()
                return removed
        except Exception as e:
            logger.error(f"Disk cache cleanup error: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT COUNT(*), SUM(size) FROM cache
                """).fetchone()

                count, total_size = result
                total_size = total_size or 0

                return {
                    "entries": count,
                    "total_size_mb": round(total_size / 1024 / 1024, 2)
                }
        except Exception as e:
            logger.error(f"Disk cache stats error: {e}")
            return {"entries": 0, "total_size_mb": 0}


class CacheManager:
    """Unified cache manager with memory and disk tiers"""

    def __init__(
        self,
        memory_size: int = 1000,
        memory_ttl: int = 3600,
        cache_dir: str = "./cache",
        disk_ttl: int = 86400,
        use_disk: bool = True
    ):
        self.memory_cache = MemoryCache(max_size=memory_size, ttl_seconds=memory_ttl)
        self.disk_cache = DiskCache(cache_dir, ttl_seconds=disk_ttl) if use_disk else None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (memory first, then disk)"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try disk cache if available
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.set(key, value)
                return value

        return default

    def set(self, key: str, value: Any):
        """Set value in both caches"""
        self.memory_cache.set(key, value)
        if self.disk_cache:
            self.disk_cache.set(key, value)

    def delete(self, key: str):
        """Delete from both caches"""
        self.memory_cache.delete(key)
        if self.disk_cache:
            self.disk_cache.delete(key)

    def clear(self):
        """Clear both caches"""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()

    def cleanup(self):
        """Cleanup expired entries"""
        if self.disk_cache:
            removed = self.disk_cache.cleanup_expired()
            logger.info(f"Cleaned up {removed} expired cache entries")
            return removed
        return 0

    def get_stats(self) -> Dict:
        """Get combined statistics"""
        stats = {
            "memory": self.memory_cache.get_stats()
        }

        if self.disk_cache:
            stats["disk"] = self.disk_cache.get_stats()

        return stats


# Decorator for caching function results
def cached(cache_manager: CacheManager, ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            cache_key = hashlib.md5(json.dumps(key_parts).encode()).hexdigest()

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)

            return result

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create cache manager
    cache = CacheManager(memory_size=100, cache_dir="./test_cache")

    # Set and get
    cache.set("key1", {"data": "value1"})
    cache.set("key2", [1, 2, 3, 4, 5])

    print("Get key1:", cache.get("key1"))
    print("Get key2:", cache.get("key2"))

    # Statistics
    stats = cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")

    # Cleanup
    cache.cleanup()
