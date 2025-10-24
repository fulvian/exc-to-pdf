#!/usr/bin/env .devstream/bin/python3
"""
Test PERF-002 Atomic Fix for Blocking Operations in Event Loop
Tests the Context7-compliant atomic fixes for non-blocking operations.

Fixes Implemented:
1. time.sleep(delay) → await asyncio.sleep(delay)
2. Ollama embedding generation → background thread via run_in_executor()
3. Database operations → background thread via run_in_executor()
4. Database connection pooling for performance
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

# Test framework setup
import pytest
pytest_plugins = []

# Import the hook to test
sys.path.insert(0, str(Path(__file__).parent.parent / 'memory'))
from post_tool_use import PostToolUseHook


class TestPerf002AtomicFixes:
    """Test suite for PERF-002 atomic fixes."""

    @pytest.fixture
    def hook(self):
        """Create a PostToolUseHook instance for testing."""
        # Mock the dependencies to avoid actual database/API calls
        with patch('post_tool_use.get_unified_client') as mock_client, \
             patch('post_tool_use.OllamaEmbeddingClient') as mock_ollama, \
             patch('post_tool_use.get_real_time_capture') as mock_rtc:

            # Configure mocks
            mock_client.return_value = Mock()
            mock_ollama.return_value = Mock()
            mock_rtc.return_value = Mock(is_running=False)

            # Create hook instance
            hook = PostToolUseHook()

            # Mock the base class methods
            hook.base = Mock()
            hook.base.debug_log = Mock()
            hook.base.success_feedback = Mock()
            hook.base.warning_feedback = Mock()
            hook.base.user_feedback = Mock()
            hook.base.should_run = Mock(return_value=True)
            hook.base.is_memory_store_enabled = Mock(return_value=True)

            yield hook

    @pytest.mark.asyncio
    async def test_async_sleep_in_retry_with_backoff(self, hook):
        """Test that retry_with_backoff uses async sleep instead of blocking sleep."""

        # Mock operation that fails twice then succeeds
        call_count = 0
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary failure")
            return "success"

        # Measure time
        start_time = time.time()

        # Execute retry logic
        result = await hook.retry_with_backoff("test_operation", mock_operation)

        end_time = time.time()
        duration = end_time - start_time

        # Verify results
        assert result == "success"
        assert call_count == 3  # 2 failures + 1 success

        # Verify non-blocking behavior (should be roughly 1s + 2s = 3s of async sleep)
        # Allow some tolerance for test execution time
        assert 2.5 <= duration <= 4.0, f"Expected ~3s, got {duration:.2f}s"

        # Verify debug logs were called
        assert hook.base.debug_log.call_count >= 2

    @pytest.mark.asyncio
    async def test_non_blocking_embedding_generation(self, hook):
        """Test that embedding generation runs in background thread."""

        # Mock Ollama client
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        hook.ollama_client.generate_embedding = Mock(return_value=mock_embedding)

        # Mock the unified client store_memory method
        hook.unified_client.store_memory = AsyncMock(return_value={
            'memory_id': 'test_memory_123'
        })

        # Mock the update_memory_embedding method
        hook.update_memory_embedding = Mock(return_value=True)

        # Test content
        test_content = "def test_function(): return 'test'"

        # Measure execution time
        start_time = time.time()

        # Execute store_in_memory
        result = await hook.store_in_memory(
            file_path="/test/test.py",
            content=test_content,
            operation="Write",
            topics=["python", "testing"],
            entities=["pytest"],
            content_type="code"
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify results
        assert result == 'test_memory_123'

        # Verify embedding generation was called
        hook.ollama_client.generate_embedding.assert_called_once_with(test_content)

        # Verify embedding update was called in background thread
        hook.update_memory_embedding.assert_called_once_with('test_memory_123', mock_embedding)

        # Should complete quickly even with embedding generation (mock is fast)
        assert duration < 1.0, f"Expected <1s, got {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_non_blocking_database_operations(self, hook):
        """Test that database operations run in background thread."""

        # Mock the database operations
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Mock connection manager
        with patch('post_tool_use.get_connection_manager') as mock_cm:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.execute.return_value = mock_cursor
            mock_cursor.rowcount = 1
            mock_conn.cursor.return_value = mock_cursor
            mock_cm.return_value.get_connection.return_value.__enter__.return_value = mock_conn
            mock_cm.return_value.get_connection.return_value.__exit__.return_value = None

            # Test the update_memory_embedding method
            result = hook.update_memory_embedding('test_memory_123', mock_embedding)

            # Verify success
            assert result is True

            # Verify database operations were called
            mock_cm.assert_called_once()

    def test_database_connection_pool_initialization(self, hook):
        """Test that database connection pool is properly initialized."""

        # Verify pool was initialized during hook creation
        assert hasattr(hook, '_db_pool')
        assert hook._db_pool is not None

        # Note: Debug log might be called during __init__ before we mock it
        # So we just verify the pool exists

    @pytest.mark.asyncio
    async def test_database_connection_pool_usage(self, hook):
        """Test that connection pool is used for database operations."""

        # Mock aiosqlite for testing
        with patch('aiosqlite.connect') as mock_connect:
            mock_db = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = ('["task1", "task2"]',)
            mock_db.execute.return_value.__aenter__.return_value = mock_cursor
            mock_aiosqlite.connect.return_value = mock_db

            # Re-initialize pool with mock
            hook._init_db_pool()

            # Test _get_active_tasks method
            tasks = await hook._get_active_tasks('test_session_123')

            # Verify results
            assert tasks == ["task1", "task2"]

            # Verify pool was used
            mock_aiosqlite.connect.assert_called()

    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self, hook):
        """Test that connection pool can be properly closed."""

        # Mock the pool close method
        hook._db_pool = AsyncMock()

        # Test cleanup
        await hook._close_db_pool()

        # Verify close was called
        hook._db_pool.close.assert_called_once()

        # Verify debug log
        hook.base.debug_log.assert_any_call("Database connection pool closed")

    def test_destructor_cleanup(self, hook):
        """Test that destructor properly handles cleanup."""

        # Mock the pool
        hook._db_pool = AsyncMock()

        # Test destructor (should not raise exceptions)
        with patch('asyncio.run') as mock_run:
            hook.__del__()

            # Verify cleanup was attempted
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_pool_failure(self, hook):
        """Test graceful degradation when connection pool fails."""

        # Mock pool initialization failure
        with patch('aiosqlite.connect', side_effect=Exception("Pool init failed")):
            # Re-initialize pool (should fail gracefully)
            hook._init_db_pool()

            # Verify pool is None but no exception was raised
            assert hook._db_pool is None

            # Verify error was logged
            hook.base.debug_log.assert_any_call("Database pool init failed: Pool init failed")

    @pytest.mark.asyncio
    async def test_embedding_generation_failure_handling(self, hook):
        """Test graceful handling of embedding generation failures."""

        # Mock Ollama client to raise exception
        hook.ollama_client.generate_embedding = Mock(side_effect=Exception("Ollama unavailable"))

        # Mock the unified client store_memory method
        hook.unified_client.store_memory = AsyncMock(return_value={
            'memory_id': 'test_memory_123'
        })

        # Test embedding generation failure (should not crash)
        result = await hook.store_in_memory(
            file_path="/test/test.py",
            content="test content",
            operation="Write",
            topics=["test"],
            entities=["test"],
            content_type="code"
        )

        # Verify memory storage still succeeded
        assert result == 'test_memory_123'

        # Verify error was logged but didn't crash
        hook.base.debug_log.assert_any_call("Embedding generation failed (non-blocking): Ollama unavailable")

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, hook):
        """Test that multiple embedding generations can run concurrently."""

        # Mock Ollama client
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        hook.ollama_client.generate_embedding = Mock(return_value=mock_embedding)

        # Mock the unified client store_memory method
        hook.unified_client.store_memory = AsyncMock(return_value={
            'memory_id': 'test_memory_123'
        })

        # Mock the update_memory_embedding method
        hook.update_memory_embedding = Mock(return_value=True)

        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = hook.store_in_memory(
                file_path=f"/test/test_{i}.py",
                content=f"def test_function_{i}(): return 'test_{i}'",
                operation="Write",
                topics=["python", "testing"],
                entities=["pytest"],
                content_type="code"
            )
            tasks.append(task)

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        duration = end_time - start_time

        # Verify all tasks succeeded
        assert all(isinstance(result, str) for result in results)
        assert len(results) == 5

        # Should complete quickly due to concurrent execution
        assert duration < 2.0, f"Expected <2s for concurrent execution, got {duration:.2f}s"

        # Verify embedding generation was called for each task
        assert hook.ollama_client.generate_embedding.call_count == 5

    @pytest.mark.asyncio
    async def test_event_loop_blocking_prevention(self, hook):
        """Test that event loop is not blocked by long operations."""

        # Mock slow Ollama client (simulates 1 second delay)
        async def slow_embedding_generation(content):
            await asyncio.sleep(1.0)  # Simulate slow operation
            return [0.1, 0.2, 0.3]

        hook.ollama_client.generate_embedding = Mock(side_effect=slow_embedding_generation)

        # Mock the unified client store_memory method
        hook.unified_client.store_memory = AsyncMock(return_value={
            'memory_id': 'test_memory_123'
        })

        # Mock the update_memory_embedding method
        hook.update_memory_embedding = Mock(return_value=True)

        # Create a background task that should run while embedding is generated
        background_task_executed = False

        async def background_task():
            nonlocal background_task_executed
            await asyncio.sleep(0.5)  # Wait a bit
            background_task_executed = True

        # Start both tasks concurrently
        start_time = time.time()

        embedding_task = hook.store_in_memory(
            file_path="/test/test.py",
            content="test content",
            operation="Write",
            topics=["test"],
            entities=["test"],
            content_type="code"
        )

        # Execute both tasks
        results = await asyncio.gather(
            embedding_task,
            background_task(),
            return_exceptions=True
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify both tasks completed
        assert isinstance(results[0], str)  # Memory storage result
        assert background_task_executed is True  # Background task ran

        # Should complete in around 1 second, not 1.5 seconds (proving non-blocking)
        assert 0.8 <= duration <= 1.3, f"Expected ~1s, got {duration:.2f}s"


def main():
    """Run the PERF-002 test suite."""
    print("🧪 PERF-002 Atomic Fix Test Suite")
    print("=" * 50)
    print("Testing Context7-compliant non-blocking operations")
    print()

    # Run tests
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]

    result = pytest.main(pytest_args)

    # Print summary
    print()
    print("=" * 50)
    if result == 0:
        print("🎉 All PERF-002 atomic fix tests passed!")
        print("✅ Non-blocking operations verified")
        print("✅ Event loop blocking prevented")
        print("✅ Database connection pooling working")
        print("✅ Graceful degradation confirmed")
        print("✅ Concurrent execution validated")
    else:
        print("❌ Some PERF-002 tests failed - review implementation")

    return result == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)