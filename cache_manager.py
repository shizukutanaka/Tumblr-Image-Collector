#!/usr/bin/env python3
"""
Unified Cache Manager - Production-Ready Caching System
Lightweight in-memory and disk caching with LRU eviction, TTL support, and statistics.

Features:
- Memory cache with LRU (Least Recently Used) eviction
- Optional disk-backed persistent cache
- TTL (time-to-live) support for automatic expiration
- Thread-safe operations
- Cache statistics and hit rate monitoring
- Automatic cleanup and garbage collection
"""

import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from collections import OrderedDict
import threading
import pickle
from datetime import datetime, timedelta

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


class AdaptiveCache(MemoryCache):
    """Adaptive cache with dynamic TTL and prefetching"""

    def __init__(self, max_size: int = 1000, base_ttl: int = 3600):
        super().__init__(max_size, base_ttl)
        self.access_patterns = OrderedDict()  # Track access patterns
        self.prefetch_candidates = set()  # Potential prefetch keys
        self.hit_rate_history = []  # Track hit rate over time
        self.max_pattern_history = 1000

    def get(self, key: str) -> Optional[Any]:
        """Enhanced get with adaptive behavior"""
        result = super().get(key)

        if result is not None:
            # Record successful access
            self._record_access_pattern(key, success=True)

            # Update TTL based on access frequency
            self._update_adaptive_ttl(key)

            # Check for prefetch opportunities
            self._check_prefetch_opportunities(key)

        else:
            # Record failed access
            self._record_access_pattern(key, success=False)

        # Update hit rate statistics
        self._update_hit_rate_stats()

        return result

    def _record_access_pattern(self, key: str, success: bool):
        """Record access pattern for analysis"""
        current_time = time.time()

        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'first_access': current_time,
                'last_access': current_time,
                'access_count': 0,
                'success_count': 0,
                'fail_count': 0,
                'avg_access_interval': 0
            }

        pattern = self.access_patterns[key]
        pattern['last_access'] = current_time
        pattern['access_count'] += 1

        if success:
            pattern['success_count'] += 1
        else:
            pattern['fail_count'] += 1

        # Calculate average access interval
        if pattern['access_count'] > 1:
            time_since_first = current_time - pattern['first_access']
            pattern['avg_access_interval'] = time_since_first / (pattern['access_count'] - 1)

        # Limit pattern history size
        if len(self.access_patterns) > self.max_pattern_history:
            self.access_patterns.popitem(last=False)

    def _update_adaptive_ttl(self, key: str):
        """Update TTL based on access frequency"""
        if key not in self.access_patterns:
            return

        pattern = self.access_patterns[key]

        # Calculate adaptive TTL based on access patterns
        base_ttl = self.ttl_seconds or 3600

        # More frequently accessed items get longer TTL
        frequency_factor = min(pattern['access_count'] / 10.0, 2.0)

        # Recent access gets bonus
        time_since_access = time.time() - pattern['last_access']
        recency_factor = max(0.5, 1.0 - (time_since_access / 3600))  # Decay over 1 hour

        # Success rate factor
        total_attempts = pattern['success_count'] + pattern['fail_count']
        success_rate = pattern['success_count'] / max(total_attempts, 1)
        success_factor = 0.8 + (success_rate * 0.4)  # 0.8 to 1.2 range

        adaptive_ttl = base_ttl * frequency_factor * recency_factor * success_factor

        # Update the cache entry with new TTL
        if key in self._cache:
            value, _ = self._cache[key]
            self._cache[key] = (value, time.time() - (self.ttl_seconds - adaptive_ttl))

    def _check_prefetch_opportunities(self, key: str):
        """Check for prefetch opportunities based on access patterns"""
        if key not in self.access_patterns:
            return

        pattern = self.access_patterns[key]

        # High frequency access suggests prefetching related keys
        if pattern['access_count'] > 5 and pattern['success_count'] / max(pattern['access_count'], 1) > 0.8:
            # Generate potential related keys (this would be application-specific)
            related_keys = self._generate_related_keys(key)
            self.prefetch_candidates.update(related_keys)

    def _generate_related_keys(self, key: str) -> List[str]:
        """Generate related cache keys (application-specific logic)"""
        related_keys = []

        # Example: if key is a blog URL, generate related post keys
        if 'tumblr.com' in key:
            # Extract blog name and generate related keys
            parts = key.split('/')
            if len(parts) >= 4:
                blog_name = parts[2].replace('.tumblr.com', '')
                # Generate sequential post keys (this is just an example)
                for i in range(1, 6):  # Prefetch next 5 posts
                    related_keys.append(f"{blog_name}_post_{i}")

        return related_keys

    def _update_hit_rate_stats(self):
        """Update hit rate statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]

        if total_requests > 0:
            current_hit_rate = (self._stats["hits"] / total_requests) * 100
            self.hit_rate_history.append(current_hit_rate)

            # Keep only recent history (last 1000 entries)
            if len(self.hit_rate_history) > 1000:
                self.hit_rate_history = self.hit_rate_history[-1000:]

    def get_cache_analytics(self) -> Dict:
        """Get detailed cache analytics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        avg_hit_rate = sum(self.hit_rate_history[-100:]) / min(len(self.hit_rate_history), 100) if self.hit_rate_history else 0

        # Find most popular keys
        popular_keys = []
        for key, pattern in self.access_patterns.items():
            if pattern['access_count'] > 3:  # Only keys accessed more than 3 times
                popular_keys.append({
                    'key': key,
                    'access_count': pattern['access_count'],
                    'success_rate': pattern['success_count'] / max(pattern['access_count'], 1),
                    'last_access': pattern['last_access']
                })

        popular_keys.sort(key=lambda x: x['access_count'], reverse=True)

        return {
            'current_stats': self.get_stats(),
            'avg_hit_rate_last_100': round(avg_hit_rate, 2),
            'total_patterns_tracked': len(self.access_patterns),
            'prefetch_candidates': len(self.prefetch_candidates),
            'popular_keys': popular_keys[:10],  # Top 10
            'recommendations': self._get_cache_recommendations()
        }

    def _get_cache_recommendations(self) -> List[str]:
        """Get cache optimization recommendations"""
        recommendations = []

        total_requests = self._stats["hits"] + self._stats["misses"]
        if total_requests > 100:
            hit_rate = (self._stats["hits"] / total_requests) * 100

            if hit_rate < 50:
                recommendations.append("キャッシュヒット率が低いです。キャッシュサイズを増やすか、TTLを調整してください。")
            elif hit_rate > 90:
                recommendations.append("キャッシュヒット率が高いです。キャッシュサイズをさらに増やせます。")

        if len(self.access_patterns) > self.max_size * 0.8:
            recommendations.append("アクセスパターンが多いです。キャッシュサイズを増やすことを検討してください。")

        if self._stats["evictions"] > self._stats["hits"]:
            recommendations.append("キャッシュサイズが不足しています。サイズを増やすことを推奨します。")

        return recommendations

    def optimize_cache_size(self) -> Dict:
        """Automatically optimize cache size based on usage patterns"""
        analytics = self.get_cache_analytics()

        current_size = analytics['current_stats']['size']
        current_max = analytics['current_stats']['max_size']
        hit_rate = analytics['current_stats']['hit_rate']
        evictions = analytics['current_stats']['evictions']

        recommendations = {}

        # Size optimization
        if hit_rate < 60 and evictions > current_size * 0.1:
            new_size = min(current_max * 2, 10000)
            recommendations['new_max_size'] = new_size
            recommendations['reason'] = '低ヒット率と高エビクション率のため'

        elif hit_rate > 90 and evictions < current_size * 0.01:
            new_size = max(current_max // 2, 100)
            recommendations['new_max_size'] = new_size
            recommendations['reason'] = '高ヒット率と低エビクション率のため'

        # TTL optimization
        if analytics['popular_keys']:
            # Analyze popular content patterns
            avg_interval = sum(
                time.time() - key['last_access']
                for key in analytics['popular_keys'][:5]
            ) / min(len(analytics['popular_keys']), 5)

            if avg_interval < 1800:  # Less than 30 minutes average
                recommendations['new_ttl'] = min(self.ttl_seconds * 2, 86400)
                recommendations['ttl_reason'] = '人気コンテンツのアクセス間隔が短いため'

        return recommendations

    def __init__(
        self,
        memory_size: int = 1000,
        memory_ttl: int = 3600,
        cache_dir: str = "./cache",
        disk_ttl: int = 86400,
        use_disk: bool = True
    ):
        # Use AdaptiveCache instead of regular MemoryCache for better performance
        self.memory_cache = AdaptiveCache(max_size=memory_size, base_ttl=memory_ttl)
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

    def get_cache_analytics(self) -> Dict:
        """Get comprehensive cache analytics"""
        memory_analytics = self.memory_cache.get_cache_analytics() if hasattr(self.memory_cache, 'get_cache_analytics') else self.memory_cache.get_stats()

        analytics = {
            'memory': memory_analytics,
            'cache_optimization': {
                'adaptive_cache_enabled': isinstance(self.memory_cache, AdaptiveCache),
                'total_requests': self.memory_cache.get_stats()['hits'] + self.memory_cache.get_stats()['misses'],
                'optimization_suggestions': self._get_optimization_suggestions()
            }
        }

        if self.disk_cache:
            analytics['disk'] = self.disk_cache.get_stats()

        return analytics

    def _get_optimization_suggestions(self) -> List[str]:
        """Get cache optimization suggestions"""
        suggestions = []
        stats = self.memory_cache.get_stats()

        total_requests = stats['hits'] + stats['misses']
        if total_requests > 100:
            hit_rate = stats['hit_rate']

            if hit_rate < 60:
                suggestions.append("キャッシュヒット率が低いです。キャッシュサイズを増やすか、TTLを調整してください。")
                suggestions.append("アクセスパターンを分析し、適切なキー設計を検討してください。")
            elif hit_rate > 90:
                suggestions.append("キャッシュヒット率が高いです。現在の設定で良好に動作しています。")

            if stats['evictions'] > stats['hits']:
                suggestions.append("エビクション率が高いです。キャッシュサイズを増やすことを推奨します。")

        # Adaptive cache specific suggestions
        if hasattr(self.memory_cache, 'get_cache_analytics'):
            analytics = self.memory_cache.get_cache_analytics()
            if 'recommendations' in analytics:
                suggestions.extend(analytics['recommendations'])

        return suggestions


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
