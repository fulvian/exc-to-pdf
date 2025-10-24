#!/usr/bin/env python3
"""
MCP Health Check - Periodic ping and timeout detection for MCP server.

This script performs health checks on the DevStream MCP server to detect
timeouts, crashes, and unresponsive states.

Usage:
    .devstream/bin/python .claude/hooks/devstream/monitoring/mcp_health_check.py --interval 30
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from mcp_client import get_mcp_client

# Setup structured logging
logger = structlog.get_logger(__name__)


class MCPHealthCheck:
    """Health check monitor for DevStream MCP server."""

    def __init__(self, timeout_threshold: int = 10):
        """
        Initialize health check monitor.

        Args:
            timeout_threshold: Timeout threshold in seconds (default: 10)
        """
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.log_file = self.project_root / ".claude" / "logs" / "devstream" / "mcp_health_check.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.timeout_threshold = timeout_threshold
        self.mcp_client = get_mcp_client()

        # Health check state
        self.consecutive_failures = 0
        self.last_success_time: Optional[datetime] = None
        self.health_status = "unknown"  # unknown, healthy, degraded, critical

    async def ping_mcp_server(self) -> Dict[str, Any]:
        """
        Ping MCP server with lightweight operation.

        Uses devstream_list_tasks (no parameters) as ping operation.
        Falls back to process check if MCP client unavailable.

        Returns:
            Dictionary with ping status, latency_ms, success
        """
        start_time = time.time()

        try:
            # Primary: Try MCP client ping
            result = await asyncio.wait_for(
                self.mcp_client.list_tasks(),
                timeout=self.timeout_threshold
            )

            latency_ms = (time.time() - start_time) * 1000

            # Verify response is valid (has tasks array)
            if result and isinstance(result, dict):
                return {
                    "success": True,
                    "latency_ms": round(latency_ms, 2),
                    "error": None,
                    "method": "mcp_client"
                }
            else:
                return {
                    "success": False,
                    "latency_ms": round(latency_ms, 2),
                    "error": "invalid_response",
                    "method": "mcp_client"
                }

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency_ms": round(latency_ms, 2),
                "error": "timeout",
                "method": "mcp_client"
            }

        except Exception as e:
            # Fallback: Process-based health check (stdio-only mode)
            try:
                import subprocess
                result = subprocess.run(
                    ["pgrep", "-f", "devstream-mcp-server"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                latency_ms = (time.time() - start_time) * 1000

                if result.returncode == 0 and result.stdout.strip():
                    return {
                        "success": True,
                        "latency_ms": round(latency_ms, 2),
                        "error": None,
                        "method": "process_check",
                        "pid": result.stdout.strip().split('\n')[0]
                    }
                else:
                    return {
                        "success": False,
                        "latency_ms": round(latency_ms, 2),
                        "error": "no_process",
                        "method": "process_check"
                    }

            except subprocess.TimeoutError:
                return {
                    "success": False,
                    "latency_ms": round(latency_ms, 2),
                    "error": "process_timeout",
                    "method": "process_check"
                }
            except Exception as fallback_error:
                return {
                    "success": False,
                    "latency_ms": round(latency_ms, 2),
                    "error": f"mcp_error:{str(e)[:50]}|process_error:{str(fallback_error)[:50]}",
                    "method": "both_failed"
                }

    def _update_health_status(self, ping_result: Dict[str, Any]) -> str:
        """
        Update health status based on ping result.

        Args:
            ping_result: Ping operation result

        Returns:
            Updated health status (healthy, degraded, critical)
        """
        if ping_result["success"]:
            self.consecutive_failures = 0
            self.last_success_time = datetime.utcnow()

            # Check latency for degraded status
            if ping_result["latency_ms"] > (self.timeout_threshold * 1000 * 0.5):
                self.health_status = "degraded"
                return "degraded"
            else:
                self.health_status = "healthy"
                return "healthy"

        else:
            self.consecutive_failures += 1

            # Critical: 3+ consecutive failures
            if self.consecutive_failures >= 3:
                self.health_status = "critical"
                return "critical"

            # Degraded: 1-2 failures
            self.health_status = "degraded"
            return "degraded"

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary with health check results
        """
        timestamp = datetime.utcnow()

        # Ping MCP server
        ping_result = await self.ping_mcp_server()

        # Update health status
        health_status = self._update_health_status(ping_result)

        # Calculate uptime
        uptime_seconds = None
        if self.last_success_time:
            uptime = timestamp - self.last_success_time
            uptime_seconds = int(uptime.total_seconds())

        result = {
            "timestamp": timestamp.isoformat() + "Z",
            "health_status": health_status,
            "ping_success": ping_result["success"],
            "ping_latency_ms": ping_result["latency_ms"],
            "ping_error": ping_result["error"],
            "ping_method": ping_result.get("method", "unknown"),
            "process_id": ping_result.get("pid"),
            "consecutive_failures": self.consecutive_failures,
            "uptime_seconds": uptime_seconds
        }

        # Log to file
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Log to structured logger
        if health_status == "healthy":
            logger.info(
                "MCP health check: HEALTHY",
                latency_ms=ping_result["latency_ms"]
            )
        elif health_status == "degraded":
            logger.warning(
                "MCP health check: DEGRADED",
                latency_ms=ping_result["latency_ms"],
                failures=self.consecutive_failures,
                error=ping_result.get("error")
            )
        elif health_status == "critical":
            logger.error(
                "MCP health check: CRITICAL",
                failures=self.consecutive_failures,
                error=ping_result.get("error")
            )

        return result

    async def monitor_loop(self, interval_seconds: int = 30):
        """
        Continuous health monitoring loop.

        Args:
            interval_seconds: Check interval in seconds (default: 30)
        """
        logger.info("Starting MCP health monitor", interval=interval_seconds)

        while True:
            try:
                result = await self.check_health()

                # Alert on critical status
                if result["health_status"] == "critical":
                    print(
                        f"🚨 CRITICAL: MCP server unresponsive "
                        f"({result['consecutive_failures']} consecutive failures)",
                        file=sys.stderr
                    )

            except Exception as e:
                logger.error("Health check loop error", error=str(e))

            await asyncio.sleep(interval_seconds)


async def main():
    """Main entry point for health check."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Health Check Monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout threshold in seconds (default: 10)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single check and exit"
    )

    args = parser.parse_args()

    health_check = MCPHealthCheck(timeout_threshold=args.timeout)

    # Single check mode
    if args.once:
        result = await health_check.check_health()
        import json
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["health_status"] == "healthy" else 1)

    # Continuous monitoring mode
    await health_check.monitor_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
