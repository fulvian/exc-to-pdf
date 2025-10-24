#!/usr/bin/env python3
"""
MCP Process Monitor - Detects and alerts on multiple MCP server instances.

This script monitors MCP server processes and detects zombie instances that can
cause database lock contention and timeout issues.

Usage:
    .devstream/bin/python .claude/hooks/devstream/monitoring/mcp_process_monitor.py
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import structlog

# Setup structured logging
logger = structlog.get_logger(__name__)


class MCPProcessMonitor:
    """Monitor MCP server processes for zombie detection."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.mcp_server_path = self.project_root / "mcp-devstream-server" / "dist" / "index.js"
        self.max_instances = 2  # Allow max 2 instances (1 active + 1 starting/stopping)

    async def get_mcp_processes(self) -> List[Dict[str, Any]]:
        """
        Get all running MCP server processes.

        Returns:
            List of process info dictionaries with pid, cpu, mem, start_time
        """
        try:
            # Use pgrep to find processes matching MCP server path
            result = subprocess.run(
                ["pgrep", "-fl", str(self.mcp_server_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                # No processes found (returncode 1) or error
                return []

            processes = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                # Parse: "PID command args..."
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

                pid = int(parts[0])

                # Get detailed process info using ps
                ps_result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "pid,pcpu,pmem,lstart"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if ps_result.returncode == 0:
                    # Skip header line
                    lines = ps_result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        ps_line = lines[1].strip()
                        ps_parts = ps_line.split()
                        if len(ps_parts) >= 8:
                            processes.append({
                                "pid": pid,
                                "cpu": float(ps_parts[1]),
                                "mem": float(ps_parts[2]),
                                "start_time": " ".join(ps_parts[3:])
                            })

            return processes

        except subprocess.TimeoutExpired:
            logger.error("Process check timed out")
            return []
        except Exception as e:
            logger.error("Failed to get MCP processes", error=str(e))
            return []

    async def kill_zombie_processes(self, processes: List[Dict[str, Any]]) -> int:
        """
        Kill zombie MCP processes, keeping only the most recent one.

        Args:
            processes: List of process info dictionaries

        Returns:
            Number of processes killed
        """
        if len(processes) <= self.max_instances:
            return 0

        # Sort by start time (oldest first)
        sorted_processes = sorted(processes, key=lambda p: p["start_time"])

        # Keep the newest instances, kill the rest
        to_keep = sorted_processes[-self.max_instances:]
        to_kill = sorted_processes[:-self.max_instances]

        killed_count = 0
        for process in to_kill:
            try:
                subprocess.run(
                    ["kill", "-9", str(process["pid"])],
                    check=True,
                    timeout=5
                )
                logger.warning(
                    "Killed zombie MCP process",
                    pid=process["pid"],
                    cpu=process["cpu"],
                    mem=process["mem"],
                    start_time=process["start_time"]
                )
                killed_count += 1
            except Exception as e:
                logger.error(
                    "Failed to kill zombie process",
                    pid=process["pid"],
                    error=str(e)
                )

        return killed_count

    async def check_and_alert(self) -> Dict[str, Any]:
        """
        Check for zombie processes and return alert status.

        Returns:
            Dictionary with status, process_count, killed_count, alert_level
        """
        processes = await self.get_mcp_processes()
        process_count = len(processes)

        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "process_count": process_count,
            "processes": processes,
            "killed_count": 0,
            "alert_level": "ok"
        }

        if process_count == 0:
            result["alert_level"] = "warning"
            result["message"] = "No MCP server processes running"
            logger.warning("No MCP server processes detected")
        elif process_count > self.max_instances:
            result["alert_level"] = "critical"
            result["message"] = f"Zombie processes detected: {process_count} instances running"
            logger.critical(
                "Zombie MCP processes detected",
                count=process_count,
                max_allowed=self.max_instances
            )

            # Kill zombie processes
            killed_count = await self.kill_zombie_processes(processes)
            result["killed_count"] = killed_count

            if killed_count > 0:
                result["message"] += f" (killed {killed_count})"
        else:
            logger.info("MCP server health OK", process_count=process_count)

        return result

    async def monitor_loop(self, interval_seconds: int = 60):
        """
        Continuous monitoring loop.

        Args:
            interval_seconds: Check interval in seconds (default: 60)
        """
        logger.info("Starting MCP process monitor", interval=interval_seconds)

        while True:
            try:
                result = await self.check_and_alert()

                # Log to structured log file
                log_file = self.project_root / ".claude" / "logs" / "devstream" / "mcp_process_monitor.jsonl"
                log_file.parent.mkdir(parents=True, exist_ok=True)

                import json
                with open(log_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

                # Alert on critical issues
                if result["alert_level"] == "critical":
                    print(f"⚠️  ALERT: {result['message']}", file=sys.stderr)

            except Exception as e:
                logger.error("Monitor loop error", error=str(e))

            await asyncio.sleep(interval_seconds)


async def main():
    """Main entry point for monitoring script."""
    monitor = MCPProcessMonitor()

    # Run single check (for manual/cron execution)
    if "--once" in sys.argv:
        result = await monitor.check_and_alert()
        import json
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["alert_level"] == "ok" else 1)

    # Continuous monitoring mode
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
