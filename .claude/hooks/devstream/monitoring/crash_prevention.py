#!/usr/bin/env python3
"""
Crash Prevention Monitor - System stability monitoring for DevStream.

Prevents kernel panics by monitoring:
- File descriptor usage (ulimit monitoring)
- Memory pressure patterns
- File system event storms
- Resource exhaustion indicators

Based on Context7 best practices for watchdog on macOS.

Author: DevStream Crash Prevention Team
License: MIT
"""

import os
import psutil
import platform
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

logger = structlog.get_logger()


class CrashRiskLevel(Enum):
    """Crash risk assessment levels."""
    LOW = "low"           # Normal operation
    MEDIUM = "medium"     # Caution advised
    HIGH = "high"         # Risk detected
    CRITICAL = "critical" # Immediate action required


@dataclass
class SystemRiskMetrics:
    """System metrics that indicate crash risk."""
    file_descriptors_used: int
    file_descriptors_limit: int
    fd_usage_percent: float
    memory_pressure: float
    cpu_usage: float
    io_operations: int
    risk_level: CrashRiskLevel
    warnings: List[str]
    timestamp: datetime


class CrashPreventionMonitor:
    """Monitor system conditions that could lead to kernel panics."""

    # Risk thresholds based on macOS crash patterns
    FD_WARNING_THRESHOLD = 80.0    # % of ulimit
    FD_CRITICAL_THRESHOLD = 95.0   # % of ulimit
    MEMORY_WARNING_THRESHOLD = 85.0  # % memory usage
    CPU_WARNING_THRESHOLD = 90.0    # % CPU usage
    IO_STORM_THRESHOLD = 1000      # I/O ops per second

    def __init__(self):
        self.system = platform.system()
        self.risk_history: List[SystemRiskMetrics] = []
        self.max_history = 100  # Keep last 100 assessments

        logger.info("CrashPreventionMonitor initialized",
                   platform=self.system)

    def get_file_descriptor_usage(self) -> tuple[int, int]:
        """Get current file descriptor usage and limits."""
        try:
            if self.system == 'Darwin':  # macOS
                # macOS-specific: get process fd count
                pid = os.getpid()
                proc_path = f"/proc/{pid}/fdinfo" if os.path.exists("/proc") else None

                if proc_path and os.path.exists(proc_path):
                    fd_count = len(os.listdir(proc_path))
                else:
                    # Fallback: estimate based on lsof
                    import subprocess
                    result = subprocess.run(['lsof', '-p', str(pid)],
                                          capture_output=True, text=True)
                    fd_count = len(result.stdout.splitlines()) - 1  # Subtract header

                # Get ulimit
                import resource
                fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]

                return fd_count, fd_limit
            else:
                # Linux/Unix systems
                pid = os.getpid()
                fd_count = len(os.listdir(f"/proc/{pid}/fd"))
                import resource
                fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
                return fd_count, fd_limit

        except Exception as e:
            logger.warning("Failed to get file descriptor usage", error=str(e))
            return 0, 1024  # Conservative fallback

    def assess_crash_risk(self) -> SystemRiskMetrics:
        """Assess current system risk for kernel panics."""
        warnings = []
        risk_level = CrashRiskLevel.LOW

        # File descriptor usage (primary indicator for FSEvents issues)
        fd_used, fd_limit = self.get_file_descriptor_usage()
        fd_usage_percent = (fd_used / fd_limit) * 100 if fd_limit > 0 else 0

        if fd_usage_percent >= self.FD_CRITICAL_THRESHOLD:
            risk_level = CrashRiskLevel.CRITICAL
            warnings.append(f"Critical file descriptor usage: {fd_usage_percent:.1f}%")
        elif fd_usage_percent >= self.FD_WARNING_THRESHOLD:
            risk_level = CrashRiskLevel.HIGH
            warnings.append(f"High file descriptor usage: {fd_usage_percent:.1f}%")
        elif fd_usage_percent >= 60.0:
            risk_level = CrashRiskLevel.MEDIUM
            warnings.append(f"Moderate file descriptor usage: {fd_usage_percent:.1f}%")

        # Memory pressure
        memory = psutil.virtual_memory()
        memory_pressure = memory.percent

        if memory_pressure >= self.MEMORY_WARNING_THRESHOLD:
            risk_level = max(risk_level, CrashRiskLevel.HIGH)
            warnings.append(f"High memory usage: {memory_pressure:.1f}%")

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage >= self.CPU_WARNING_THRESHOLD:
            risk_level = max(risk_level, CrashRiskLevel.HIGH)
            warnings.append(f"High CPU usage: {cpu_usage:.1f}%")

        # I/O operations (potential file system event storm)
        try:
            io_stats = psutil.disk_io_counters()
            io_operations = (io_stats.read_count + io_stats.write_count) if io_stats else 0

            # Check for I/O storm (simple heuristic)
            if len(self.risk_history) > 0:
                last_metrics = self.risk_history[-1]
                time_diff = (datetime.now() - last_metrics.timestamp).total_seconds()
                if time_diff > 0:
                    io_rate = (io_operations - last_metrics.io_operations) / time_diff
                    if io_rate > self.IO_STORM_THRESHOLD:
                        risk_level = max(risk_level, CrashRiskLevel.CRITICAL)
                        warnings.append(f"I/O storm detected: {io_rate:.0f} ops/sec")
        except Exception:
            io_operations = 0

        # Create metrics
        metrics = SystemRiskMetrics(
            file_descriptors_used=fd_used,
            file_descriptors_limit=fd_limit,
            fd_usage_percent=fd_usage_percent,
            memory_pressure=memory_pressure,
            cpu_usage=cpu_usage,
            io_operations=io_operations,
            risk_level=risk_level,
            warnings=warnings,
            timestamp=datetime.now()
        )

        # Store in history
        self.risk_history.append(metrics)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)

        # Log assessment
        logger.info("Crash risk assessment",
                   risk_level=risk_level.value,
                   fd_usage=f"{fd_usage_percent:.1f}%",
                   memory=f"{memory_pressure:.1f}%",
                   cpu=f"{cpu_usage:.1f}%",
                   warnings=len(warnings))

        if warnings:
            logger.warning("Crash risk warnings detected", warnings=warnings)

        return metrics

    def get_safety_recommendations(self, metrics: SystemRiskMetrics) -> List[str]:
        """Get safety recommendations based on current metrics."""
        recommendations = []

        if metrics.fd_usage_percent >= self.FD_WARNING_THRESHOLD:
            recommendations.extend([
                "Reduce file system monitoring intensity",
                "Consider increasing ulimit: ulimit -n 2048",
                "Use PollingObserver instead of native FSEvents",
                "Close unused file handles"
            ])

        if metrics.memory_pressure >= self.MEMORY_WARNING_THRESHOLD:
            recommendations.extend([
                "Reduce memory-intensive operations",
                "Monitor for memory leaks",
                "Consider restarting DevStream session"
            ])

        if metrics.cpu_usage >= self.CPU_WARNING_THRESHOLD:
            recommendations.extend([
                "Reduce concurrent operations",
                "Check for infinite loops or blocking operations",
                "Monitor process priority"
            ])

        if metrics.risk_level in [CrashRiskLevel.HIGH, CrashRiskLevel.CRITICAL]:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Risk of kernel panic")

        return recommendations

    def should_disable_monitoring(self, metrics: SystemRiskMetrics) -> bool:
        """Determine if file monitoring should be disabled for safety."""
        return (
            metrics.risk_level == CrashRiskLevel.CRITICAL or
            metrics.fd_usage_percent >= self.FD_CRITICAL_THRESHOLD or
            metrics.memory_pressure >= 95.0
        )

    def get_system_summary(self) -> Dict[str, any]:
        """Get summary of system state and recent risks."""
        if not self.risk_history:
            return {"status": "no_data", "message": "No risk assessments available"}

        latest = self.risk_history[-1]
        recent_high_risk = sum(1 for m in self.risk_history[-10:]
                             if m.risk_level in [CrashRiskLevel.HIGH, CrashRiskLevel.CRITICAL])

        return {
            "current_risk": latest.risk_level.value,
            "fd_usage": f"{latest.fd_usage_percent:.1f}%",
            "memory_pressure": f"{latest.memory_pressure:.1f}%",
            "cpu_usage": f"{latest.cpu_usage:.1f}%",
            "recent_high_risk_events": recent_high_risk,
            "warnings": latest.warnings,
            "recommendations": self.get_safety_recommendations(latest),
            "last_assessment": latest.timestamp.isoformat()
        }


# Singleton instance for global access
_monitor_instance: Optional[CrashPreventionMonitor] = None

def get_crash_monitor() -> CrashPreventionMonitor:
    """Get global crash prevention monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CrashPreventionMonitor()
    return _monitor_instance

def assess_current_risk() -> SystemRiskMetrics:
    """Convenience function to assess current crash risk."""
    return get_crash_monitor().assess_crash_risk()

def should_disable_file_monitoring() -> bool:
    """Check if file monitoring should be disabled for safety."""
    metrics = assess_current_risk()
    should_disable = get_crash_monitor().should_disable_monitoring(metrics)

    if should_disable:
        logger.critical("File monitoring disabled for crash prevention",
                       risk_level=metrics.risk_level.value,
                       fd_usage=metrics.fd_usage_percent)

    return should_disable