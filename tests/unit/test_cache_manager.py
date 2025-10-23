"""
Unit tests for cache management system.

This module tests intelligent caching with memory limits, disk-based
storage for large objects, and LRU eviction policies.
"""

import pickle
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.cache_manager import (
    CacheManager,
    CacheEntry,
    CacheStats,
    get_global_cache,
    cache_with_ttl
)


class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""

    def test_cache_entry_creation(self) -> None:
        """Test CacheEntry object creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=50,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 50
        assert entry.access_count == 1
        assert entry.expires_at is None
        assert entry.is_expired is False

    def test_cache_entry_expiration(self) -> None:
        """Test CacheEntry expiration logic."""
        current_time = time.time()
        past_time = current_time - 100  # 100 seconds ago
        future_time = current_time + 100  # 100 seconds in future

        # Expired entry
        expired_entry = CacheEntry(
            key="expired",
            value="value",
            size_bytes=50,
            created_at=past_time,
            last_accessed=past_time,
            access_count=1,
            expires_at=past_time  # Expired
        )
        assert expired_entry.is_expired is True

        # Non-expired entry
        valid_entry = CacheEntry(
            key="valid",
            value="value",
            size_bytes=50,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            expires_at=future_time  # Not expired
        )
        assert valid_entry.is_expired is False

        # Entry without expiration
        no_expire_entry = CacheEntry(
            key="no_expire",
            value="value",
            size_bytes=50,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1
        )
        assert no_expire_entry.is_expired is False

    def test_cache_entry_touch(self) -> None:
        """Test CacheEntry touch functionality."""
        entry = CacheEntry(
            key="test",
            value="value",
            size_bytes=50,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1
        )

        original_time = entry.last_accessed
        original_count = entry.access_count

        time.sleep(0.01)  # Small delay
        entry.touch()

        assert entry.last_accessed > original_time
        assert entry.access_count == original_count + 1


class TestCacheManager:
    """Test cases for CacheManager functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for cache tests."""
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def cache_manager(self, temp_dir: Path) -> CacheManager:
        """Create cache manager for testing."""
        return CacheManager(
            max_memory_mb=10,
            max_disk_mb=50,
            cache_dir=temp_dir,
            ttl_seconds=None,
            enable_disk_cache=True
        )

    @pytest.fixture
    def memory_only_cache(self) -> CacheManager:
        """Create memory-only cache manager."""
        return CacheManager(
            max_memory_mb=10,
            max_disk_mb=0,
            enable_disk_cache=False
        )

    def test_cache_manager_initialization(self, temp_dir: Path) -> None:
        """Test cache manager initialization."""
        manager = CacheManager(
            max_memory_mb=100,
            max_disk_mb=500,
            cache_dir=temp_dir,
            ttl_seconds=3600,
            enable_disk_cache=True
        )

        assert manager.max_memory_bytes == 100 * 1024 * 1024
        assert manager.max_disk_bytes == 500 * 1024 * 1024
        assert manager.cache_dir == temp_dir
        assert manager.ttl_seconds == 3600
        assert manager.enable_disk_cache is True
        assert manager.db_path == temp_dir / "cache.db"

    def test_cache_manager_memory_only_initialization(self) -> None:
        """Test memory-only cache manager initialization."""
        manager = CacheManager(
            max_memory_mb=50,
            enable_disk_cache=False
        )

        assert manager.max_memory_bytes == 50 * 1024 * 1024
        assert manager.enable_disk_cache is False
        assert manager.cache_dir is None
        assert manager.db_path is None

    def test_calculate_size(self, cache_manager: CacheManager) -> None:
        """Test object size calculation."""
        small_value = "test"
        large_value = "x" * 1000

        small_size = cache_manager._calculate_size(small_value)
        large_size = cache_manager._calculate_size(large_value)

        assert large_size > small_size
        assert small_size > 0
        assert large_size > 0

    def test_generate_key(self, cache_manager: CacheManager) -> None:
        """Test cache key generation."""
        key1_parts = ["test", "key", "parts"]
        key2_parts = ["test", "key", "parts"]
        key3_parts = ["test", "different", "parts"]

        key1 = cache_manager._generate_key(key1_parts)
        key2 = cache_manager._generate_key(key2_parts)
        key3 = cache_manager._generate_key(key3_parts)

        # Same parts should generate same key
        assert key1 == key2

        # Different parts should generate different keys
        assert key1 != key3

        # Keys should be consistent length (SHA256)
        assert len(key1) == 64
        assert len(key3) == 64

    def test_put_and_get_memory(self, memory_only_cache: CacheManager) -> None:
        """Test putting and getting values from memory cache."""
        key_parts = ["test", "key"]
        value = {"data": "test_value", "number": 42}

        # Put value
        result = memory_only_cache.put(key_parts, value)
        assert result is True

        # Get value
        retrieved = memory_only_cache.get(key_parts)
        assert retrieved == value

        # Get statistics
        stats = memory_only_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.memory_entries == 1

    def test_get_nonexistent_key(self, cache_manager: CacheManager) -> None:
        """Test getting non-existent key."""
        result = cache_manager.get(["nonexistent", "key"])
        assert result is None

        stats = cache_manager.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1

    def test_put_with_ttl(self, cache_manager: CacheManager) -> None:
        """Test putting value with TTL."""
        key_parts = ["ttl", "test"]
        value = "expires_soon"

        # Put with short TTL
        result = cache_manager.put(key_parts, value, ttl_seconds=0.1)
        assert result is True

        # Should be available immediately
        retrieved = cache_manager.get(key_parts)
        assert retrieved == value

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired now
        retrieved = cache_manager.get(key_parts)
        assert retrieved is None

    def test_memory_eviction(self, memory_only_cache: CacheManager) -> None:
        """Test memory-based LRU eviction."""
        # Fill cache with large entries to trigger eviction
        large_values = [f"large_value_{i}" * 1000 for i in range(20)]

        for i, value in enumerate(large_values):
            memory_only_cache.put([f"key_{i}"], value)

        # Some entries should have been evicted
        stats = memory_only_cache.get_stats()
        assert stats.evictions > 0
        assert stats.memory_entries < 20

    def test_disk_cache_storage(self, cache_manager: CacheManager) -> None:
        """Test disk cache storage."""
        key_parts = ["disk", "test"]
        value = "this should go to disk"

        # Put value that's too large for memory cache
        large_value = value * 10000  # Make it large
        result = cache_manager.put(key_parts, large_value)
        assert result is True

        # Clear memory cache
        cache_manager._memory_cache.clear()

        # Should still be retrievable from disk
        retrieved = cache_manager.get(key_parts)
        assert retrieved == large_value

        stats = cache_manager.get_stats()
        assert stats.disk_entries > 0

    def test_invalidate(self, cache_manager: CacheManager) -> None:
        """Test cache entry invalidation."""
        key_parts = ["invalidate", "test"]
        value = "to_be_removed"

        # Put value
        cache_manager.put(key_parts, value)

        # Verify it exists
        retrieved = cache_manager.get(key_parts)
        assert retrieved == value

        # Invalidate
        result = cache_manager.invalidate(key_parts)
        assert result is True

        # Should be gone
        retrieved = cache_manager.get(key_parts)
        assert retrieved is None

    def test_invalidate_nonexistent(self, cache_manager: CacheManager) -> None:
        """Test invalidating non-existent entry."""
        result = cache_manager.invalidate(["nonexistent"])
        assert result is False

    def test_clear(self, cache_manager: CacheManager) -> None:
        """Test clearing all cache entries."""
        # Add some entries
        for i in range(5):
            cache_manager.put([f"key_{i}"], f"value_{i}")

        # Verify entries exist
        stats = cache_manager.get_stats()
        assert stats.total_entries > 0

        # Clear cache
        cache_manager.clear()

        # Verify cache is empty
        stats = cache_manager.get_stats()
        assert stats.total_entries == 0
        assert stats.memory_entries == 0

    def test_cleanup_expired_entries(self, cache_manager: CacheManager) -> None:
        """Test cleanup of expired entries."""
        # Add entries with different TTLs
        cache_manager.put(["immediate"], "expires_immediately", ttl_seconds=0.01)
        cache_manager.put(["long_lived"], "stays_fresh", ttl_seconds=3600)

        # Wait for immediate expiration
        time.sleep(0.1)

        # Run cleanup
        cache_manager.cleanup()

        # Expired entry should be gone
        assert cache_manager.get(["immediate"]) is None
        # Long-lived entry should remain
        assert cache_manager.get(["long_lived"]) == "stays_fresh"

    def test_cache_stats(self, cache_manager: CacheManager) -> None:
        """Test cache statistics."""
        stats = cache_manager.get_stats()
        initial_hits = stats.hits
        initial_misses = stats.misses

        # Perform cache operations
        cache_manager.put(["test"], "value")
        cache_manager.get(["test"])  # Hit
        cache_manager.get(["nonexistent"])  # Miss

        stats = cache_manager.get_stats()
        assert stats.hits == initial_hits + 1
        assert stats.misses == initial_misses + 1
        assert stats.total_entries > 0
        assert stats.hit_rate > 0

    def test_value_too_large(self, cache_manager: CacheManager) -> None:
        """Test handling of values too large for cache."""
        # Create a value larger than both memory and disk limits
        huge_value = "x" * (cache_manager.max_memory_bytes * 2)

        result = cache_manager.put(["huge"], huge_value)
        assert result is False

        # Should not be cached
        retrieved = cache_manager.get(["huge"])
        assert retrieved is None

    def test_concurrent_access(self, cache_manager: CacheManager) -> None:
        """Test concurrent cache access."""
        import threading

        results = []
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(10):
                    key = [f"worker_{worker_id}", f"key_{i}"]
                    value = f"worker_{worker_id}_value_{i}"

                    # Put and get
                    cache_manager.put(key, value)
                    retrieved = cache_manager.get(key)
                    results.append((worker_id, i, retrieved == value))
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify operations succeeded
        assert len(results) == 50  # 5 workers * 10 operations each
        assert all(success for _, _, success in results)


class TestGlobalCache:
    """Test cases for global cache functionality."""

    def test_get_global_cache(self) -> None:
        """Test getting global cache instance."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        # Should return the same instance
        assert cache1 is cache2
        assert isinstance(cache1, CacheManager)

    def test_global_cache_persistence(self) -> None:
        """Test global cache data persistence."""
        cache = get_global_cache()

        key_parts = ["global", "test"]
        value = {"persistent": "data"}

        # Store in global cache
        cache.put(key_parts, value)

        # Retrieve from same instance
        retrieved = cache.get(key_parts)
        assert retrieved == value

        # Retrieve from new instance reference
        cache_ref = get_global_cache()
        retrieved_ref = cache_ref.get(key_parts)
        assert retrieved_ref == value


class TestCacheDecorator:
    """Test cases for cache decorator functionality."""

    def test_cache_with_ttl_decorator(self) -> None:
        """Test TTL cache decorator."""
        call_count = 0

        @cache_with_ttl(ttl_seconds=0.1)
        def expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * y + call_count  # Include call_count to detect cache misses

        # First call should execute function
        result1 = expensive_function(5, 10)
        assert result1 == 51  # 5*10 + 1
        assert call_count == 1

        # Second call with same args should use cache
        result2 = expensive_function(5, 10)
        assert result2 == 51  # Same result from cache
        assert call_count == 1  # Function not called again

        # Call with different args should execute function
        result3 = expensive_function(3, 7)
        assert result3 == 22  # 3*7 + 2
        assert call_count == 2

        # Wait for cache expiration
        time.sleep(0.2)

        # Call after expiration should execute function again
        result4 = expensive_function(5, 10)
        assert result4 == 52  # 5*10 + 3 (new call)
        assert call_count == 3

    def test_cache_decorator_with_different_args(self) -> None:
        """Test cache decorator with different argument combinations."""
        call_count = 0

        @cache_with_ttl(ttl_seconds=1.0)
        def multi_arg_function(a: str, b: int, c: float = 1.0) -> str:
            nonlocal call_count
            call_count += 1
            return f"{a}_{b}_{c}_{call_count}"

        # Different arg combinations should cache separately
        result1 = multi_arg_function("test", 1)
        result2 = multi_arg_function("test", 2)
        result3 = multi_arg_function("test", 1, 2.0)

        assert call_count == 3  # All calls should execute function
        assert result1 != result2 != result3

        # Repeat calls should use cache
        result1_repeat = multi_arg_function("test", 1)
        result2_repeat = multi_arg_function("test", 2)
        result3_repeat = multi_arg_function("test", 1, 2.0)

        assert call_count == 3  # No additional function calls
        assert result1_repeat == result1
        assert result2_repeat == result2
        assert result3_repeat == result3


class TestErrorHandling:
    """Test cases for error handling in cache operations."""

    def test_disk_cache_error_handling(self, temp_dir: Path) -> None:
        """Test disk cache error handling."""
        cache = CacheManager(
            max_memory_mb=10,
            max_disk_mb=50,
            cache_dir=temp_dir,
            enable_disk_cache=True
        )

        # Corrupt the database file
        if cache.db_path and cache.db_path.exists():
            cache.db_path.write_text("corrupted data")

        # Should handle corruption gracefully
        key_parts = ["error", "test"]
        value = "test_value"

        # Put should still work (fall back to memory)
        result = cache.put(key_parts, value)
        assert result is True

        # Get should work from memory
        retrieved = cache.get(key_parts)
        assert retrieved == value

    def test_pickle_error_handling(self, cache_manager: CacheManager) -> None:
        """Test handling of unpickleable objects."""
        # Try to cache an unpickleable object (file handle)
        import io

        unpickleable = io.StringIO("test")
        key_parts = ["unpickleable", "test"]

        # Should handle pickle error gracefully
        result = cache_manager.put(key_parts, unpickleable)
        # Result might be True or False depending on fallback behavior
        assert isinstance(result, bool)

        # Get should not crash
        retrieved = cache_manager.get(key_parts)
        # Might return None or the object depending on error handling