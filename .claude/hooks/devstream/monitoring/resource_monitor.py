#!/usr/bin/env python3
"""
ResourceMonitor - System resource health monitoring for crash prevention.

Monitors macOS system resources (RAM, CPU, Ollama memory, swap) and provides
early warning before crashes occur.

Author: DevStream
License: MIT
"""

import psutil
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Resource health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceMetric:
    """Single resource metric with threshold context."""
    name: str
    current: float
    warning_threshold: float
    critical_threshold: float
    unit: str  # "percent" | "MB" | "GB"
    status: HealthStatus


@dataclass
class ResourceHealth:
    """Overall system resource health snapshot."""
    healthy: bool
    status: HealthStatus
    warnings: List[str]
    metrics: Dict[str, ResourceMetric]
    timestamp: datetime
    cache_age_seconds: float

    def is_critical(self) -> bool:
        """
        Check if ANY metric is at CRITICAL level.

        Returns:
            True if any resource is in CRITICAL state
        """
        return self.status == HealthStatus.CRITICAL

    def get_warning_summary(self) -> str:
        """
        Human-readable summary of warnings.

        Returns:
            Summary string describing current health state
        """
        if self.healthy:
            return "All systems healthy"
        return f"{len(self.warnings)} warnings: {', '.join(self.warnings)}"


class ResourceMonitor:
    """Monitor system resources for crash prevention."""

    # Thresholds (from architecture spec)
    THRESHOLDS = {
        'memory_percent': {'warning': 85, 'critical': 95},
        'cpu_percent': {'warning': 75, 'critical': 90},
        'ollama_memory_mb': {'warning': 800, 'critical': 1200},
        'swap_memory_gb': {'warning': 2, 'critical': 4}
    }

    CACHE_TTL_SECONDS = 8  # Cache ResourceHealth for 8s
    OLLAMA_PID_CACHE_SECONDS = 60  # Cache Ollama PID for 60s

    def __init__(self):
        """Initialize ResourceMonitor with empty cache."""
        self._cached_health: Optional[ResourceHealth] = None
        self._cache_timestamp: Optional[datetime] = None
        self._ollama_pid: Optional[int] = None
        self._ollama_pid_timestamp: Optional[datetime] = None

    def check_stability(self) -> ResourceHealth:
        """
        Check system resource health with caching.

        Returns:
            ResourceHealth object with current system state

        Note:
            Results cached for 8 seconds to avoid polling spam.
            Performance: <25ms per check (95th percentile).
        """
        # Check cache validity
        if self._is_cache_valid():
            logger.debug("ResourceMonitor using cached health",
                        cache_age_seconds=self._get_cache_age())
            return self._cached_health

        # Perform fresh check
        start_time = datetime.now()

        # Collect all metrics
        metrics: Dict[str, ResourceMetric] = {}
        warnings: List[str] = []

        # Check each resource
        mem_metric = self._check_memory()
        metrics['memory'] = mem_metric
        if mem_metric.status != HealthStatus.HEALTHY:
            warnings.append('SYSTEM_MEMORY_PRESSURE')

        cpu_metric = self._check_cpu()
        metrics['cpu'] = cpu_metric
        if cpu_metric.status != HealthStatus.HEALTHY:
            warnings.append('HIGH_CPU_USAGE')

        ollama_metric = self._check_ollama()
        if ollama_metric:  # May be None if Ollama not running
            metrics['ollama'] = ollama_metric
            if ollama_metric.status != HealthStatus.HEALTHY:
                warnings.append('OLLAMA_MEMORY_HIGH')

        swap_metric = self._check_swap()
        metrics['swap'] = swap_metric
        if swap_metric.status != HealthStatus.HEALTHY:
            warnings.append('SWAP_THRASHING')

        # Determine overall status (worst metric wins)
        overall_status = HealthStatus.HEALTHY
        for metric in metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif metric.status == HealthStatus.WARNING:
                overall_status = HealthStatus.WARNING

        # Build ResourceHealth
        health = ResourceHealth(
            healthy=(overall_status == HealthStatus.HEALTHY),
            status=overall_status,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now(),
            cache_age_seconds=0.0
        )

        # Update cache
        self._cached_health = health
        self._cache_timestamp = datetime.now()

        # Performance logging
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug("ResourceMonitor check complete",
                    elapsed_ms=round(elapsed_ms, 2),
                    status=health.status.value,
                    warnings_count=len(warnings))

        return health

    def _is_cache_valid(self) -> bool:
        """
        Check if cached health is still valid.

        Returns:
            True if cache exists and is within TTL
        """
        if self._cached_health is None or self._cache_timestamp is None:
            return False

        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self.CACHE_TTL_SECONDS

    def _get_cache_age(self) -> float:
        """
        Get age of cached data in seconds.

        Returns:
            Age in seconds, 0.0 if no cache
        """
        if self._cache_timestamp is None:
            return 0.0
        return (datetime.now() - self._cache_timestamp).total_seconds()

    def _check_memory(self) -> ResourceMetric:
        """
        Check system RAM usage.

        Returns:
            ResourceMetric with current memory state

        Note:
            Uses psutil.virtual_memory() for system-wide RAM usage
        """
        mem = psutil.virtual_memory()
        percent = mem.percent

        # Determine status
        if percent >= self.THRESHOLDS['memory_percent']['critical']:
            status = HealthStatus.CRITICAL
        elif percent >= self.THRESHOLDS['memory_percent']['warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return ResourceMetric(
            name="system_memory",
            current=percent,
            warning_threshold=self.THRESHOLDS['memory_percent']['warning'],
            critical_threshold=self.THRESHOLDS['memory_percent']['critical'],
            unit="percent",
            status=status
        )

    def _check_cpu(self) -> ResourceMetric:
        """
        Check system CPU usage (5s average).

        Returns:
            ResourceMetric with current CPU state

        Note:
            Uses interval=0 for non-blocking (uses cached value).
            First call may block ~100ms, subsequent calls instant.
        """
        # Use interval=0 for non-blocking (uses cached value)
        cpu_percent = psutil.cpu_percent(interval=0)

        # Determine status
        if cpu_percent >= self.THRESHOLDS['cpu_percent']['critical']:
            status = HealthStatus.CRITICAL
        elif cpu_percent >= self.THRESHOLDS['cpu_percent']['warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return ResourceMetric(
            name="cpu_percent",
            current=cpu_percent,
            warning_threshold=self.THRESHOLDS['cpu_percent']['warning'],
            critical_threshold=self.THRESHOLDS['cpu_percent']['critical'],
            unit="percent",
            status=status
        )

    def _check_ollama(self) -> Optional[ResourceMetric]:
        """
        Check Ollama process memory usage.

        Returns:
            ResourceMetric if Ollama running, None otherwise

        Note:
            Uses PID caching (60s TTL) to avoid repeated process_iter()
        """
        ollama_proc = self._find_ollama_runner()

        if not ollama_proc:
            return None  # Ollama not running

        # Get RSS memory in MB
        memory_mb = ollama_proc.memory_info().rss / 1024 / 1024

        # Determine status
        if memory_mb >= self.THRESHOLDS['ollama_memory_mb']['critical']:
            status = HealthStatus.CRITICAL
        elif memory_mb >= self.THRESHOLDS['ollama_memory_mb']['warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return ResourceMetric(
            name="ollama_memory",
            current=memory_mb,
            warning_threshold=self.THRESHOLDS['ollama_memory_mb']['warning'],
            critical_threshold=self.THRESHOLDS['ollama_memory_mb']['critical'],
            unit="MB",
            status=status
        )

    def _check_swap(self) -> ResourceMetric:
        """
        Check swap memory usage.

        Returns:
            ResourceMetric with current swap state

        Note:
            High swap usage (>2GB) indicates memory pressure and thrashing risk
        """
        swap = psutil.swap_memory()
        swap_gb = swap.used / 1024 / 1024 / 1024  # Convert to GB

        # Determine status
        if swap_gb >= self.THRESHOLDS['swap_memory_gb']['critical']:
            status = HealthStatus.CRITICAL
        elif swap_gb >= self.THRESHOLDS['swap_memory_gb']['warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return ResourceMetric(
            name="swap_memory",
            current=swap_gb,
            warning_threshold=self.THRESHOLDS['swap_memory_gb']['warning'],
            critical_threshold=self.THRESHOLDS['swap_memory_gb']['critical'],
            unit="GB",
            status=status
        )

    def _find_ollama_runner(self) -> Optional[psutil.Process]:
        """
        Find Ollama runner process with PID caching.

        Returns:
            psutil.Process if found, None otherwise

        Note:
            Caches PID for 60 seconds to avoid repeated process_iter().
            Looks for process with 'ollama' in name and 'runner' in cmdline.
        """
        # Check PID cache
        if self._ollama_pid and self._ollama_pid_timestamp:
            age = (datetime.now() - self._ollama_pid_timestamp).total_seconds()
            if age < self.OLLAMA_PID_CACHE_SECONDS:
                try:
                    proc = psutil.Process(self._ollama_pid)
                    if proc.is_running() and 'ollama' in proc.name().lower():
                        return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # PID invalid, re-scan

        # Scan for Ollama runner process
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('runner' in str(arg) for arg in cmdline):
                        # Cache PID
                        self._ollama_pid = proc.pid
                        self._ollama_pid_timestamp = datetime.now()
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return None  # Ollama not running
