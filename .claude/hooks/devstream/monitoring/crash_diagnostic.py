#!/usr/bin/env python3
"""
Crash Diagnostic Tool - Analyzes kernel panics and system crashes for DevStream.

Provides comprehensive analysis of:
- Kernel panic reports (macOS)
- System resource usage patterns
- File system event correlations
- DevStream operational history

Based on Context7 best practices for debugging and monitoring.

Author: DevStream Diagnostic Team
License: MIT
"""

import re
import json
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class CrashAnalysis:
    """Analysis of a system crash event."""
    crash_type: str
    timestamp: datetime
    process_involved: str
    memory_corruption: bool
    filesystem_involvement: bool
    devstream_correlation: float  # 0.0 to 1.0
    indicators: List[str]
    recommendations: List[str]
    raw_report: str


@dataclass
class SystemSnapshot:
    """Snapshot of system state at time of analysis."""
    timestamp: datetime
    platform: str
    devstream_version: str
    python_version: str
    file_descriptors: int
    memory_usage: float
    cpu_usage: float
    running_processes: List[str]
    devstream_hooks_active: List[str]


class CrashDiagnosticTool:
    """Comprehensive crash diagnostic and analysis tool."""

    def __init__(self, devstream_root: Optional[Path] = None):
        self.devstream_root = devstream_root or Path.cwd()
        self.platform = platform.system()
        self.analysis_history: List[CrashAnalysis] = []

        logger.info("CrashDiagnosticTool initialized",
                   platform=self.platform,
                   devstream_root=str(self.devstream_root))

    def parse_macos_panic_report(self, report_text: str) -> Optional[CrashAnalysis]:
        """
        Parse macOS kernel panic report and analyze for DevStream correlation.

        Args:
            report_text: Full text of kernel panic report

        Returns:
            CrashAnalysis with correlation assessment
        """
        try:
            # Extract panic type and process
            panic_match = re.search(r'panic\((.*?)\): (.+)', report_text)
            if not panic_match:
                return None

            panic_type, panic_description = panic_match.groups()

            # Extract timestamp
            timestamp_match = re.search(r'Epoch Time:\s*sec\s*usec\s*\n\s*(\w+)\s+(\w+)', report_text)
            if timestamp_match:
                timestamp = datetime.fromtimestamp(int(timestamp_match.group(1), 16))
            else:
                timestamp = datetime.now()

            # Extract panicked task
            task_match = re.search(r'Panicked task.*?(\d+): (.+?)\s*\n', report_text)
            process_involved = task_match.group(2) if task_match else "unknown"

            # Check for memory corruption indicators
            memory_correlation_keywords = [
                'use-after-free',
                'memory abort',
                'kernel data abort',
                'double free',
                'buffer overflow',
                'malloc'
            ]

            memory_corruption = any(keyword in report_text.lower()
                                  for keyword in memory_correlation_keywords)

            # Check for file system involvement
            fs_correlation_keywords = [
                'fseventsd',
                'filesystem',
                'watchdog',
                'file system',
                'observer',
                'kalloc'
            ]

            filesystem_involvement = any(keyword in report_text.lower()
                                        for keyword in fs_correlation_keywords)

            # Calculate DevStream correlation score
            correlation_score = self._calculate_devstream_correlation(
                report_text, process_involved, memory_corruption, filesystem_involvement
            )

            # Generate indicators
            indicators = self._extract_crash_indicators(report_text)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                correlation_score, memory_corruption, filesystem_involvement, process_involved
            )

            analysis = CrashAnalysis(
                crash_type=f"{panic_type}: {panic_description}",
                timestamp=timestamp,
                process_involved=process_involved,
                memory_corruption=memory_corruption,
                filesystem_involvement=filesystem_involvement,
                devstream_correlation=correlation_score,
                indicators=indicators,
                recommendations=recommendations,
                raw_report=report_text
            )

            self.analysis_history.append(analysis)
            return analysis

        except Exception as e:
            logger.error("Failed to parse macOS panic report", error=str(e))
            return None

    def _calculate_devstream_correlation(self, report_text: str, process: str,
                                       memory_corruption: bool, filesystem_involvement: bool) -> float:
        """Calculate correlation score between crash and DevStream."""
        score = 0.0

        # High correlation indicators
        if 'fseventsd' in process:
            score += 0.6  # Direct FSEvents involvement
        if 'use-after-free' in report_text.lower():
            score += 0.4  # Memory corruption pattern
        if 'kalloc' in report_text.lower():
            score += 0.3  # Kernel allocator involvement

        # Medium correlation indicators
        if filesystem_involvement:
            score += 0.3
        if memory_corruption:
            score += 0.2

        # Check for DevStream-specific patterns
        if any(keyword in report_text.lower() for keyword in ['claude', 'devstream', 'hook']):
            score += 0.2

        # File descriptor pressure indicators
        if 'file' in report_text.lower() and memory_corruption:
            score += 0.1

        return min(score, 1.0)

    def _extract_crash_indicators(self, report_text: str) -> List[str]:
        """Extract key indicators from crash report."""
        indicators = []

        # Memory corruption indicators
        if 'use-after-free' in report_text.lower():
            indicators.append("Memory corruption: use-after-free detected")
        if 'kernel data abort' in report_text.lower():
            indicators.append("Kernel memory access violation")
        if 'malloc' in report_text.lower():
            indicators.append("Memory allocator involvement")

        # File system indicators
        if 'fseventsd' in report_text.lower():
            indicators.append("File system events daemon involved")
        if 'kalloc' in report_text.lower():
            indicators.append("Kernel allocator pressure")

        # Process indicators
        if 'Panicked task' in report_text:
            task_match = re.search(r'Panicked task.*?(\d+): (.+?)\s*\n', report_text)
            if task_match:
                indicators.append(f"Panicked process: {task_match.group(2)}")

        # Register dump indicators
        if 'x0:' in report_text and 'x1:' in report_text:
            indicators.append("Register dump available for analysis")

        return indicators

    def _generate_recommendations(self, correlation_score: float,
                                memory_corruption: bool, filesystem_involvement: bool,
                                process: str) -> List[str]:
        """Generate crash-specific recommendations."""
        recommendations = []

        if correlation_score >= 0.8:
            recommendations.extend([
                "🚨 HIGH CORRELATION: DevStream likely contributed to this crash",
                "IMMEDIATE: Disable real-time file monitoring",
                "Use PollingObserver instead of native FSEvents",
                "Review file descriptor usage patterns"
            ])
        elif correlation_score >= 0.5:
            recommendations.extend([
                "⚠️ MEDIUM CORRELATION: DevStream may have contributed",
                "Consider disabling intensive file monitoring",
                "Monitor system resource usage",
                "Review recent DevStream activity"
            ])
        else:
            recommendations.extend([
                "ℹ️ LOW CORRELATION: DevStream unlikely to be primary cause",
                "Monitor for patterns in future crashes",
                "Consider other running applications"
            ])

        if memory_corruption:
            recommendations.extend([
                "🧠 MEMORY CORRUPTION: Check for memory leaks",
                "Monitor memory usage of DevStream processes",
                "Consider reducing memory-intensive operations"
            ])

        if filesystem_involvement:
            recommendations.extend([
                "📁 FILESYSTEM INVOLVEMENT: Review file monitoring setup",
                "Check for excessive file handle usage",
                "Consider reducing file system monitoring frequency"
            ])

        if 'fseventsd' in process:
            recommendations.extend([
                "🎯 FSEVENTS CRASH: Use PollingObserver on macOS",
                "This is a known issue with watchdog library",
                "See: https://github.com/gorakhargosh/watchdog"
            ])

        return recommendations

    def collect_system_snapshot(self) -> SystemSnapshot:
        """Collect comprehensive system state snapshot."""
        try:
            # Basic system info
            platform_info = platform.platform()
            python_version = platform.python_version()

            # DevStream version (try to get from package or git)
            devstream_version = "unknown"
            try:
                result = subprocess.run(['git', 'describe', '--tags'],
                                      capture_output=True, text=True,
                                      cwd=self.devstream_root)
                if result.returncode == 0:
                    devstream_version = result.stdout.strip()
            except:
                pass

            # Resource usage
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=1)

            # File descriptors
            import resource
            fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if platform.system() == 'Darwin':
                pid = psutil.Process().pid
                try:
                    result = subprocess.run(['lsof', '-p', str(pid)],
                                          capture_output=True, text=True)
                    fd_count = len(result.stdout.splitlines()) - 1
                except:
                    fd_count = 0
            else:
                fd_count = len([f for f in Path(f"/proc/{psutil.Process().pid}/fd").iterdir()])

            # Running processes
            running_processes = []
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        running_processes.append(f"{proc.info['name']} (pid: {proc.info['pid']})")
                    except:
                        pass
            except:
                running_processes = ["Unable to enumerate processes"]

            # DevStream hooks
            devstream_hooks = []
            hooks_dir = self.devstream_root / '.claude' / 'hooks' / 'devstream'
            if hooks_dir.exists():
                for hook_file in hooks_dir.rglob('*.py'):
                    if hook_file.name != '__init__.py':
                        devstream_hooks.append(str(hook_file.relative_to(self.devstream_root)))

            snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                platform=platform_info,
                devstream_version=devstream_version,
                python_version=python_version,
                file_descriptors=fd_count,
                memory_usage=memory_info.percent,
                cpu_usage=cpu_usage,
                running_processes=running_processes[:50],  # Limit to first 50
                devstream_hooks_active=devstream_hooks
            )

            return snapshot

        except Exception as e:
            logger.error("Failed to collect system snapshot", error=str(e))
            return SystemSnapshot(
                timestamp=datetime.now(),
                platform=platform.platform(),
                devstream_version="error",
                python_version=platform.python_version(),
                file_descriptors=0,
                memory_usage=0.0,
                cpu_usage=0.0,
                running_processes=[f"Error collecting snapshot: {str(e)}"],
                devstream_hooks_active=[]
            )

    def generate_diagnostic_report(self, include_snapshot: bool = True) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform,
            "devstream_root": str(self.devstream_root),
            "analyses_performed": len(self.analysis_history),
            "crash_analyses": []
        }

        # Include crash analyses
        for analysis in self.analysis_history:
            report["crash_analyses"].append({
                "crash_type": analysis.crash_type,
                "timestamp": analysis.timestamp.isoformat(),
                "process_involved": analysis.process_involved,
                "devstream_correlation": analysis.devstream_correlation,
                "memory_corruption": analysis.memory_corruption,
                "filesystem_involvement": analysis.filesystem_involvement,
                "indicators": analysis.indicators,
                "recommendations": analysis.recommendations
            })

        # Include system snapshot
        if include_snapshot:
            snapshot = self.collect_system_snapshot()
            report["system_snapshot"] = {
                "timestamp": snapshot.timestamp.isoformat(),
                "platform": snapshot.platform,
                "devstream_version": snapshot.devstream_version,
                "python_version": snapshot.python_version,
                "file_descriptors": snapshot.file_descriptors,
                "memory_usage_percent": snapshot.memory_usage,
                "cpu_usage_percent": snapshot.cpu_usage,
                "running_processes_count": len(snapshot.running_processes),
                "devstream_hooks_active": snapshot.devstream_hooks_active
            }

        # Generate summary and recommendations
        if self.analysis_history:
            max_correlation = max(a.devstream_correlation for a in self.analysis_history)
            report["summary"] = {
                "highest_devstream_correlation": max_correlation,
                "total_crashes_analyzed": len(self.analysis_history),
                "memory_correlation_present": any(a.memory_corruption for a in self.analysis_history),
                "filesystem_correlation_present": any(a.filesystem_involvement for a in self.analysis_history)
            }

            if max_correlation >= 0.8:
                report["summary"]["overall_assessment"] = "High likelihood DevStream is contributing to system crashes"
            elif max_correlation >= 0.5:
                report["summary"]["overall_assessment"] = "DevStream may be contributing to system crashes"
            else:
                report["summary"]["overall_assessment"] = "Low correlation between DevStream and crashes"

        return report

    def save_diagnostic_report(self, output_path: Optional[Path] = None) -> Path:
        """Save diagnostic report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.devstream_root / f"crash_diagnostic_{timestamp}.json"

        report = self.generate_diagnostic_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Diagnostic report saved", path=str(output_path))
        return output_path


# Convenience functions
def analyze_macos_panic_report(report_text: str, devstream_root: Optional[Path] = None) -> Optional[CrashAnalysis]:
    """Analyze macOS kernel panic report for DevStream correlation."""
    tool = CrashDiagnosticTool(devstream_root)
    return tool.parse_macos_panic_report(report_text)

def generate_crash_diagnostic(devstream_root: Optional[Path] = None) -> Dict[str, Any]:
    """Generate comprehensive crash diagnostic report."""
    tool = CrashDiagnosticTool(devstream_root)
    return tool.generate_diagnostic_report()

def save_crash_diagnostic(devstream_root: Optional[Path] = None) -> Path:
    """Save crash diagnostic report to file."""
    tool = CrashDiagnosticTool(devstream_root)
    return tool.save_diagnostic_report()