"""
Cache management system for Excel processing optimization.

This module provides intelligent caching with memory limits, disk-based
storage for large objects, and LRU eviction policies to improve performance
for repeated operations and large file processing.
"""

import hashlib
import pickle
import sqlite3
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Tuple
import threading
import weakref

import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    expires_at: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_entries: int = 0
    disk_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        """Get total cache size in MB."""
        return self.total_size_bytes / (1024 * 1024)


class CacheManager:
    """
    Intelligent cache manager with memory and disk storage.

    Provides LRU caching with configurable memory limits, automatic
    disk overflow for large objects, and comprehensive cache statistics.
    """

    def __init__(
        self,
        max_memory_mb: int = 100,
        max_disk_mb: int = 1000,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[float] = None,
        enable_disk_cache: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            max_memory_mb: Maximum memory cache size in MB
            max_disk_mb: Maximum disk cache size in MB
            cache_dir: Directory for disk cache (auto-generated if None)
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            enable_disk_cache: Whether to enable disk-based caching
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache

        # Initialize cache storage
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()

        # Initialize disk cache
        if self.enable_disk_cache:
            self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "exc-to-pdf-cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_dir / "cache.db"
            self._init_disk_cache()
        else:
            self.cache_dir = None
            self.db_path = None

        # Statistics
        self.stats = CacheStats()

        logger.info(
            "Cache manager initialized",
            max_memory_mb=max_memory_mb,
            max_disk_mb=max_disk_mb if enable_disk_cache else 0,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            ttl_seconds=ttl_seconds
        )

    def _init_disk_cache(self) -> None:
        """Initialize disk cache database."""
        if not self.enable_disk_cache:
            return

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        data BLOB,
                        size_bytes INTEGER,
                        created_at REAL,
                        last_accessed REAL,
                        access_count INTEGER,
                        expires_at REAL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed
                    ON cache_entries(last_accessed)
                """)
                conn.commit()

            logger.debug("Disk cache database initialized", db_path=str(self.db_path))

        except Exception as e:
            logger.error("Failed to initialize disk cache", error=str(e))
            self.enable_disk_cache = False

    def _calculate_size(self, obj: Any) -> int:
        """
        Calculate approximate size of an object.

        Args:
            obj: Object to size

        Returns:
            Size in bytes
        """
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback to string representation size
            return len(str(obj).encode('utf-8'))

    def _generate_key(self, key_parts: List[Any]) -> str:
        """
        Generate cache key from key components.

        Args:
            key_parts: List of key components

        Returns:
            Generated cache key
        """
        key_str = "|".join(str(part) for part in key_parts)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def _evict_memory_entries(self, target_size: Optional[int] = None) -> int:
        """
        Evict least recently used entries from memory cache.

        Args:
            target_size: Target size to reach (default: 80% of max)

        Returns:
            Number of entries evicted
        """
        target_size = target_size or int(self.max_memory_bytes * 0.8)
        evicted = 0

        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )

        current_size = sum(entry.size_bytes for entry in self._memory_cache.values())

        for key, entry in sorted_entries:
            if current_size <= target_size:
                break

            # Move to disk cache if enabled and entry is not too large
            if (self.enable_disk_cache and
                entry.size_bytes < self.max_disk_bytes * 0.1):  # Max 10% of disk cache per entry
                try:
                    self._store_to_disk(entry)
                except Exception as e:
                    logger.warning(f"Failed to move entry to disk: {e}")

            # Remove from memory
            del self._memory_cache[key]
            current_size -= entry.size_bytes
            evicted += 1
            self.stats.evictions += 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} memory entries", freed_bytes=current_size)

        return evicted

    def _store_to_disk(self, entry: CacheEntry) -> None:
        """
        Store cache entry to disk.

        Args:
            entry: Cache entry to store
        """
        if not self.enable_disk_cache:
            return

        try:
            data = pickle.dumps(entry.value)

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, data, size_bytes, created_at, last_accessed, access_count, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.key,
                        data,
                        entry.size_bytes,
                        entry.created_at,
                        entry.last_accessed,
                        entry.access_count,
                        entry.expires_at
                    )
                )
                conn.commit()

            self.stats.disk_entries += 1
            logger.debug(f"Stored entry to disk: {entry.key[:16]}...")

        except Exception as e:
            logger.error(f"Failed to store entry to disk: {e}")

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """
        Load cache entry from disk.

        Args:
            key: Cache key

        Returns:
            Cache entry if found and valid
        """
        if not self.enable_disk_cache:
            return None

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT data, size_bytes, created_at, last_accessed, access_count, expires_at
                    FROM cache_entries WHERE key = ?
                    """,
                    (key,)
                )
                row = cursor.fetchone()

                if not row:
                    return None

                data, size_bytes, created_at, last_accessed, access_count, expires_at = row
                value = pickle.loads(data)

                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count,
                    expires_at=expires_at
                )

                # Check if expired
                if entry.is_expired:
                    self._delete_from_disk(key)
                    return None

                # Update access statistics
                entry.touch()
                self._update_disk_access_stats(key, entry)

                logger.debug(f"Loaded entry from disk: {key[:16]}...")
                return entry

        except Exception as e:
            logger.error(f"Failed to load entry from disk: {e}")
            return None

    def _update_disk_access_stats(self, key: str, entry: CacheEntry) -> None:
        """Update access statistics for disk entry."""
        if not self.enable_disk_cache:
            return

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET last_accessed = ?, access_count = ?
                    WHERE key = ?
                    """,
                    (entry.last_accessed, entry.access_count, key)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update disk access stats: {e}")

    def _delete_from_disk(self, key: str) -> None:
        """Delete entry from disk cache."""
        if not self.enable_disk_cache:
            return

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                self.stats.disk_entries -= 1
        except Exception as e:
            logger.warning(f"Failed to delete entry from disk: {e}")

    def _cleanup_disk_cache(self) -> None:
        """Clean up expired and oversized entries from disk cache."""
        if not self.enable_disk_cache:
            return

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Delete expired entries
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
                expired_deleted = cursor.rowcount

                # Delete oldest entries if over size limit
                cursor = conn.execute("""
                    SELECT SUM(size_bytes) FROM cache_entries
                """)
                total_size = cursor.fetchone()[0] or 0

                if total_size > self.max_disk_bytes:
                    # Delete oldest entries
                    cursor = conn.execute("""
                        SELECT key FROM cache_entries
                        ORDER BY last_accessed ASC
                    """)

                    current_size = total_size
                    target_size = int(self.max_disk_bytes * 0.8)
                    size_deleted = 0

                    for (key,) in cursor:
                        if current_size <= target_size:
                            break

                        # Get entry size
                        size_cursor = conn.execute(
                            "SELECT size_bytes FROM cache_entries WHERE key = ?",
                            (key,)
                        )
                        entry_size = size_cursor.fetchone()[0]

                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        current_size -= entry_size
                        size_deleted += entry_size

                    logger.debug(f"Cleaned up disk cache", deleted_bytes=size_deleted)

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to cleanup disk cache: {e}")

    def get(self, key_parts: List[Any]) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key_parts: List of key components

        Returns:
            Cached value if found and valid
        """
        key = self._generate_key(key_parts)

        with self._cache_lock:
            # Try memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                # Check if expired
                if entry.is_expired:
                    del self._memory_cache[key]
                    self.stats.misses += 1
                    return None

                entry.touch()
                self.stats.hits += 1
                logger.debug(f"Cache hit (memory): {key[:16]}...")
                return entry.value

            # Try disk cache
            if self.enable_disk_cache:
                entry = self._load_from_disk(key)
                if entry:
                    # Move to memory if space allows
                    if entry.size_bytes <= self.max_memory_bytes * 0.1:  # Max 10% of memory cache per entry
                        self._evict_memory_entries()
                        self._memory_cache[key] = entry
                        self.stats.memory_entries += 1

                    self.stats.hits += 1
                    logger.debug(f"Cache hit (disk): {key[:16]}...")
                    return entry.value

            self.stats.misses += 1
            return None

    def put(self, key_parts: List[Any], value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """
        Store value in cache.

        Args:
            key_parts: List of key components
            value: Value to cache
            ttl_seconds: Custom TTL for this entry

        Returns:
            True if stored successfully
        """
        key = self._generate_key(key_parts)
        size_bytes = self._calculate_size(value)
        current_time = time.time()

        # Calculate expiration
        expires_at = None
        if ttl_seconds or self.ttl_seconds:
            ttl = ttl_seconds or self.ttl_seconds
            expires_at = current_time + ttl

        # Check if value is too large for any cache
        if size_bytes > self.max_memory_bytes and (not self.enable_disk_cache or size_bytes > self.max_disk_bytes):
            logger.warning(f"Value too large for cache: {size_bytes} bytes")
            return False

        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            expires_at=expires_at
        )

        with self._cache_lock:
            # Store in memory if small enough
            if size_bytes <= self.max_memory_bytes * 0.5:  # Max 50% of memory cache per entry
                # Evict entries if necessary
                current_size = sum(e.size_bytes for e in self._memory_cache.values())
                if current_size + size_bytes > self.max_memory_bytes:
                    self._evict_memory_entries()

                self._memory_cache[key] = entry
                self.stats.memory_entries += 1
                logger.debug(f"Stored in memory cache: {key[:16]}...")
            else:
                # Store directly to disk
                if self.enable_disk_cache:
                    self._store_to_disk(entry)
                else:
                    logger.warning(f"Value too large for memory cache and disk cache disabled")
                    return False

            self.stats.total_entries += 1
            self.stats.total_size_bytes += size_bytes

            return True

    def invalidate(self, key_parts: List[Any]) -> bool:
        """
        Invalidate cache entry.

        Args:
            key_parts: List of key components

        Returns:
            True if entry was invalidated
        """
        key = self._generate_key(key_parts)

        with self._cache_lock:
            invalidated = False

            # Remove from memory
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                del self._memory_cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.memory_entries -= 1
                invalidated = True

            # Remove from disk
            if self.enable_disk_cache:
                self._delete_from_disk(key)
                invalidated = True

            if invalidated:
                self.stats.total_entries -= 1
                logger.debug(f"Invalidated cache entry: {key[:16]}...")

            return invalidated

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            # Clear memory cache
            self._memory_cache.clear()
            self.stats.memory_entries = 0

            # Clear disk cache
            if self.enable_disk_cache:
                try:
                    with sqlite3.connect(str(self.db_path)) as conn:
                        conn.execute("DELETE FROM cache_entries")
                        conn.commit()
                    self.stats.disk_entries = 0
                except Exception as e:
                    logger.error(f"Failed to clear disk cache: {e}")

            # Reset statistics
            self.stats = CacheStats()

            logger.info("Cache cleared")

    def get_stats(self) -> CacheStats:
        """
        Get current cache statistics.

        Returns:
            CacheStats object with current statistics
        """
        with self._cache_lock:
            # Update current counts
            self.stats.memory_entries = len(self._memory_cache)
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self._memory_cache.values())

            # Get disk entry count
            if self.enable_disk_cache:
                try:
                    with sqlite3.connect(str(self.db_path)) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                        self.stats.disk_entries = cursor.fetchone()[0]
                except Exception:
                    self.stats.disk_entries = 0

            return self.stats

    def cleanup(self) -> None:
        """Clean up expired entries and optimize cache."""
        with self._cache_lock:
            # Clean up expired memory entries
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._memory_cache[key]

            # Clean up disk cache
            if self.enable_disk_cache:
                self._cleanup_disk_cache()

        logger.debug("Cache cleanup completed")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


# Global cache instance
_global_cache: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_global_cache() -> CacheManager:
    """
    Get or create global cache instance.

    Returns:
        Global CacheManager instance
    """
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = CacheManager()

    return _global_cache


def cache_with_ttl(ttl_seconds: float):
    """
    Decorator for caching function results with TTL.

    Args:
        ttl_seconds: Time-to-live in seconds

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()

            # Generate cache key
            key_parts = [
                func.__name__,
                str(args),
                str(sorted(kwargs.items()))
            ]

            # Try to get from cache
            result = cache.get(key_parts)
            if result is not None:
                return result

            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(key_parts, result, ttl_seconds)

            return result

        return wrapper
    return decorator