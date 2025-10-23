"""
Unit tests for memory monitoring and resource management.

This module tests memory monitoring, garbage collection management,
and resource limiting capabilities for handling large Excel files efficiently.
"""

import gc
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from exc_to_pdf.memory_monitor import (
    MemoryMonitor,
    MemoryStats,
    ResourceLimits,
    get_system_memory_info,
    optimize_gc_settings,
)


class TestMemoryStats:
    """Test cases for MemoryStats dataclass."""

    def test_memory_stats_creation(self) -> None:
        """Test MemoryStats object creation."""
        stats = MemoryStats(
            current_mb=100.0,
            peak_mb=150.0,
            available_mb=800.0,
            percent_used=20.0,
            process_mb=100.0,
        )

        assert stats.current_mb == 100.0
        assert stats.peak_mb == 150.0
        assert stats.available_mb == 800.0
        assert stats.percent_used == 20.0
        assert stats.process_mb == 100.0


class TestResourceLimits:
    """Test cases for ResourceLimits dataclass."""

    def test_resource_limits_creation(self) -> None:
        """Test ResourceLimits object creation."""
        limits = ResourceLimits(
            max_memory_mb=2048, max_cpu_percent=80.0, gc_threshold_mb=100
        )

        assert limits.max_memory_mb == 2048
        assert limits.max_cpu_percent == 80.0
        assert limits.gc_threshold_mb == 100


class TestMemoryMonitor:
    """Test cases for MemoryMonitor functionality."""

    @pytest.fixture
    def resource_limits(self) -> ResourceLimits:
        """Create test resource limits."""
        return ResourceLimits(
            max_memory_mb=100, max_cpu_percent=80.0, gc_threshold_mb=10
        )

    @pytest.fixture
    def memory_monitor(self, resource_limits: ResourceLimits) -> MemoryMonitor:
        """Create memory monitor for testing."""
        return MemoryMonitor(resource_limits)

    def test_memory_monitor_initialization(
        self, resource_limits: ResourceLimits
    ) -> None:
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(resource_limits)

        assert monitor.limits == resource_limits
        assert monitor._peak_memory == 0.0
        assert monitor._last_gc_memory == 0.0
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None
        assert monitor._callbacks == {}

    def test_get_memory_stats(self, memory_monitor: MemoryMonitor) -> None:
        """Test memory statistics retrieval."""
        stats = memory_monitor.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.current_mb >= 0
        assert stats.peak_mb >= 0
        assert stats.available_mb >= 0
        assert stats.percent_used >= 0
        assert stats.process_mb >= 0

    @patch("src.memory_monitor.psutil")
    def test_get_memory_stats_with_mock(
        self, mock_psutil: Mock, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory statistics with mocked values."""
        # Mock system memory
        mock_memory_info = Mock()
        mock_memory_info.used = 8_000_000_000  # 8GB
        mock_memory_info.available = 8_000_000_000  # 8GB
        mock_memory_info.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory_info

        # Mock process memory
        mock_process_instance = Mock()
        mock_memory = Mock()
        mock_memory.rss = 500_000_000  # 500MB
        mock_process_instance.memory_info.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process_instance

        # Mock os.getpid to avoid actual process calls
        with patch("src.memory_monitor.os.getpid", return_value=12345):
            stats = memory_monitor.get_memory_stats()

        assert stats.current_mb == pytest.approx(500.0, rel=1e-1)
        assert stats.available_mb == pytest.approx(8000.0, rel=1e-1)
        assert stats.percent_used == pytest.approx(50.0, rel=1e-1)

    def test_check_memory_limits_within_limits(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory limit check when within limits."""
        with patch.object(memory_monitor, "get_memory_stats") as mock_stats:
            mock_stats.return_value = MemoryStats(
                current_mb=50.0,  # Well below 100MB limit
                peak_mb=60.0,
                available_mb=500.0,  # Well above 100MB buffer
                percent_used=20.0,
                process_mb=50.0,
            )

            assert memory_monitor.check_memory_limits() is True

    def test_check_memory_limits_exceeds_process_limit(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory limit check when process limit exceeded."""
        with patch.object(memory_monitor, "get_memory_stats") as mock_stats:
            mock_stats.return_value = MemoryStats(
                current_mb=150.0,  # Exceeds 100MB limit
                peak_mb=150.0,
                available_mb=500.0,
                percent_used=20.0,
                process_mb=150.0,
            )

            assert memory_monitor.check_memory_limits() is False

    def test_check_memory_limits_low_system_memory(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory limit check when system memory is low."""
        with patch.object(memory_monitor, "get_memory_stats") as mock_stats:
            mock_stats.return_value = MemoryStats(
                current_mb=50.0,  # Within process limit
                peak_mb=60.0,
                available_mb=50.0,  # Below 100MB system buffer
                percent_used=90.0,
                process_mb=50.0,
            )

            assert memory_monitor.check_memory_limits() is False

    def test_force_garbage_collection(self, memory_monitor: MemoryMonitor) -> None:
        """Test forced garbage collection."""
        # Add some objects to memory
        test_objects = [[] for _ in range(1000)]

        collected = memory_monitor.force_garbage_collection()

        assert isinstance(collected, int)
        assert collected >= 0

        # Verify GC was called
        assert memory_monitor._last_gc_memory >= 0

    def test_start_stop_monitoring(self, memory_monitor: MemoryMonitor) -> None:
        """Test starting and stopping memory monitoring."""
        assert memory_monitor._monitoring is False
        assert memory_monitor._monitor_thread is None

        # Start monitoring
        memory_monitor.start_monitoring(interval=0.1)  # Fast interval for testing

        assert memory_monitor._monitoring is True
        assert memory_monitor._monitor_thread is not None
        assert memory_monitor._monitor_thread.is_alive()

        # Stop monitoring
        memory_monitor.stop_monitoring()

        assert memory_monitor._monitoring is False

        # Wait for thread to finish
        if memory_monitor._monitor_thread:
            memory_monitor._monitor_thread.join(timeout=2.0)

    def test_start_monitoring_already_running(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test starting monitoring when already running."""
        memory_monitor.start_monitoring(interval=0.1)

        # Should not raise an exception
        memory_monitor.start_monitoring(interval=0.1)

        assert memory_monitor._monitoring is True

        memory_monitor.stop_monitoring()

    def test_memory_limit_callbacks(self, memory_monitor: MemoryMonitor) -> None:
        """Test memory limit exceeded callbacks."""
        callback_called = False

        def test_callback() -> None:
            nonlocal callback_called
            callback_called = True

        # Add callback
        memory_monitor.add_memory_limit_callback("test", test_callback)
        assert "test" in memory_monitor._callbacks

        # Trigger callback manually
        memory_monitor._trigger_memory_limit_callbacks()
        assert callback_called is True

        # Remove callback
        memory_monitor.remove_memory_limit_callback("test")
        assert "test" not in memory_monitor._callbacks

    def test_context_manager(self, resource_limits: ResourceLimits) -> None:
        """Test memory monitor as context manager."""
        with MemoryMonitor(resource_limits) as monitor:
            assert monitor._monitoring is True
            assert monitor._monitor_thread is not None

        # After context exit
        assert monitor._monitoring is False

    def test_monitoring_loop_error_handling(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test monitoring loop error handling."""
        with patch.object(memory_monitor, "get_memory_stats") as mock_stats:
            mock_stats.side_effect = Exception("Test error")

            # Start monitoring with fast interval
            memory_monitor.start_monitoring(interval=0.05)

            # Let monitoring run for a short time
            time.sleep(0.2)

            # Should handle errors gracefully and continue monitoring
            assert memory_monitor._monitoring is True

            memory_monitor.stop_monitoring()


class TestSystemMemoryInfo:
    """Test cases for system memory information functions."""

    def test_get_system_memory_info(self) -> None:
        """Test system memory information retrieval."""
        info = get_system_memory_info()

        assert isinstance(info, dict)
        assert "total_mb" in info
        assert "available_mb" in info
        assert "used_mb" in info
        assert "percent_used" in info

        # Verify values are reasonable
        assert info["total_mb"] > 0
        assert info["available_mb"] >= 0
        assert info["used_mb"] >= 0
        assert 0 <= info["percent_used"] <= 100

    @patch("src.memory_monitor.psutil.virtual_memory")
    def test_get_system_memory_info_with_mock(self, mock_virtual_memory: Mock) -> None:
        """Test system memory information with mocked values."""
        mock_memory = Mock()
        mock_memory.total = 16_000_000_000  # 16GB
        mock_memory.available = 8_000_000_000  # 8GB
        mock_memory.used = 8_000_000_000  # 8GB
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory

        info = get_system_memory_info()

        assert info["total_mb"] == pytest.approx(16000.0, rel=1e-1)
        assert info["available_mb"] == pytest.approx(8000.0, rel=1e-1)
        assert info["used_mb"] == pytest.approx(8000.0, rel=1e-1)
        assert info["percent_used"] == pytest.approx(50.0, rel=1e-1)

    @patch("src.memory_monitor.psutil.virtual_memory")
    def test_get_system_memory_info_error_handling(
        self, mock_virtual_memory: Mock
    ) -> None:
        """Test system memory information error handling."""
        mock_virtual_memory.side_effect = Exception("Test error")

        info = get_system_memory_info()

        # Should return empty dict on error
        assert info == {}


class TestGCOptimization:
    """Test cases for garbage collection optimization."""

    def test_optimize_gc_settings(self) -> None:
        """Test GC settings optimization."""
        original_threshold = gc.get_threshold()
        original_debug = gc.get_debug()

        try:
            optimize_gc_settings()

            # Verify thresholds were set (may be same as original)
            threshold = gc.get_threshold()
            assert len(threshold) == 3
            assert all(isinstance(t, int) for t in threshold)

        finally:
            # Restore original settings
            gc.set_threshold(*original_threshold)
            gc.set_debug(original_debug)

    @patch("src.memory_monitor.gc.set_threshold")
    @patch("src.memory_monitor.gc.set_debug")
    def test_optimize_gc_settings_with_mock(
        self, mock_set_debug: Mock, mock_set_threshold: Mock
    ) -> None:
        """Test GC settings optimization with mocked functions."""
        optimize_gc_settings()

        mock_set_threshold.assert_called_once_with(700, 10, 10)
        mock_set_debug.assert_called_once()  # Called with debug flags


class TestIntegration:
    """Integration tests for memory monitoring components."""

    def test_memory_monitor_with_real_memory_usage(self) -> None:
        """Test memory monitor with actual memory usage."""
        limits = ResourceLimits(
            max_memory_mb=1000, gc_threshold_mb=1  # Large limit for this test
        )

        monitor = MemoryMonitor(limits)

        # Create some memory pressure
        large_data = [[] for _ in range(1000)]
        for i in range(1000):
            large_data[i] = [j for j in range(1000)]

        # Get memory stats
        stats = monitor.get_memory_stats()
        assert stats.process_mb > 0

        # Force garbage collection
        collected = monitor.force_garbage_collection()
        assert isinstance(collected, int)

        # Clean up
        del large_data
        gc.collect()

    def test_memory_monitor_callback_integration(self) -> None:
        """Test memory monitor callback integration."""
        callback_data = {}

        def memory_limit_callback() -> None:
            callback_data["called"] = True
            # Just record that callback was called successfully

        limits = ResourceLimits(max_memory_mb=1)  # Very low limit
        monitor = MemoryMonitor(limits)
        monitor.add_memory_limit_callback("test", memory_limit_callback)

        with patch.object(monitor, "get_memory_stats") as mock_stats:
            mock_stats.return_value = MemoryStats(
                current_mb=10.0,  # Exceeds limit
                peak_mb=10.0,
                available_mb=1000.0,
                percent_used=10.0,
                process_mb=10.0,
            )

            # Trigger memory limit check
            result = monitor.check_memory_limits()
            assert result is False

            # Manually trigger callbacks
            monitor._trigger_memory_limit_callbacks()

            # Verify callback was called
            assert callback_data.get("called") is True
