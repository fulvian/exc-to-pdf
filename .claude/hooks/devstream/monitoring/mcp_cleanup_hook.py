#!/usr/bin/env python3
"""
MCP Cleanup Hook - Kills zombie MCP processes before PreToolUse execution.

This hook runs before EVERY tool execution to ensure no zombie MCP processes
exist that could cause database lock contention.

Integrated into PreToolUse hook chain for automatic cleanup.

Usage:
    .devstream/bin/python .claude/hooks/devstream/monitoring/mcp_cleanup_hook.py
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import structlog

# Setup structured logging
logger = structlog.get_logger("mcp_cleanup_hook")

# Context7 Pattern: Dynamic path resolution
def get_project_root():
    """Get project root using Context7/Dynaconf patterns."""
    # Priority 1: DEVSTREAM_PROJECT_ROOT (multi-project mode)
    project_root = os.getenv("DEVSTREAM_PROJECT_ROOT")

    # Priority 2: Current working directory (single-project mode)
    if project_root is None:
        project_root = os.getcwd()

    return Path(project_root)

# Dynamic path configuration (Context7 best practice)
PROJECT_ROOT = get_project_root()
CORRECT_DB_PATH = str(PROJECT_ROOT / "data" / "devstream.db")
WRONG_DB_PATH_PATTERN = "mcp-devstream-server/data/devstream.db"


class MCPCleanupHook:
    """Context7-compliant cleanup hook to prevent zombie MCP processes."""

    def __init__(self):
        # Context7 Pattern: Dynamic project root detection
        self.project_root = PROJECT_ROOT
        self.devstream_root = self._detect_devstream_root()
        self.mcp_server_path = self.devstream_root / "mcp-devstream-server" / "dist" / "index.js"
        self.max_instances = 2  # Allow max 2 instances

        # Multi-project log paths
        self.log_file = self.project_root / ".claude" / "logs" / "devstream" / "mcp_cleanup_hook.jsonl"
        self.wrong_path_log = self.project_root / ".claude" / "logs" / "devstream" / "wrong-path-detections.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.wrong_path_log.parent.mkdir(parents=True, exist_ok=True)

    def _detect_devstream_root(self):
        """Detect DevStream installation root using Context7 patterns."""
        # Priority 1: DEVSTREAM_ROOT environment variable
        devstream_root = os.getenv("DEVSTREAM_ROOT")

        # Priority 2: Try to locate based on hook script location
        if devstream_root is None:
            hook_file = Path(__file__)
            # Hook is in: /Users/fulvioventura/devstream/.claude/hooks/devstream/monitoring/
            # DevStream root is 4 levels up
            potential_root = hook_file.parent.parent.parent.parent
            if (potential_root / ".claude" / "hooks" / "devstream").exists():
                devstream_root = str(potential_root)

        # Priority 3: Fallback to project root (single-project mode)
        if devstream_root is None:
            devstream_root = str(self.project_root)

        return Path(devstream_root)

    async def get_mcp_process_count(self) -> int:
        """
        Get count of running MCP server processes.

        Returns:
            Number of running MCP processes
        """
        try:
            result = subprocess.run(
                ["pgrep", "-f", str(self.mcp_server_path)],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return 0

            # Count non-empty lines
            return len([line for line in result.stdout.strip().split("\n") if line])

        except subprocess.TimeoutExpired:
            logger.error("Process count check timed out")
            return 0
        except Exception as e:
            logger.error("Failed to count MCP processes", error=str(e))
            return 0

    async def kill_excess_processes(self) -> Dict[str, Any]:
        """
        Kill excess MCP processes, keeping only the most recent ones.

        Returns:
            Dictionary with cleanup status
        """
        try:
            # Get all MCP process PIDs sorted by start time
            result = subprocess.run(
                ["pgrep", "-f", str(self.mcp_server_path)],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return {"killed_count": 0, "remaining_count": 0, "status": "no_processes"}

            pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid]

            if len(pids) <= self.max_instances:
                return {
                    "killed_count": 0,
                    "remaining_count": len(pids),
                    "status": "ok"
                }

            # Get process start times
            process_info = []
            for pid in pids:
                ps_result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "lstart="],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if ps_result.returncode == 0:
                    start_time = ps_result.stdout.strip()
                    process_info.append({"pid": pid, "start_time": start_time})

            # Sort by start time (oldest first)
            process_info.sort(key=lambda p: p["start_time"])

            # Kill oldest processes, keep newest
            to_kill = process_info[:-self.max_instances]
            killed_count = 0

            for proc in to_kill:
                try:
                    subprocess.run(
                        ["kill", "-9", str(proc["pid"])],
                        check=True,
                        timeout=2
                    )
                    logger.warning(
                        "Cleanup killed zombie MCP process",
                        pid=proc["pid"],
                        start_time=proc["start_time"]
                    )
                    killed_count += 1
                except Exception as e:
                    logger.error(
                        "Failed to kill process",
                        pid=proc["pid"],
                        error=str(e)
                    )

            return {
                "killed_count": killed_count,
                "remaining_count": len(pids) - killed_count,
                "status": "cleaned" if killed_count > 0 else "ok"
            }

        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            return {"killed_count": 0, "remaining_count": 0, "status": "error", "error": str(e)}

    def detect_wrong_path_usage(self, cmd_line: str) -> bool:
        """
        Detect if MCP process is using wrong database path.

        Args:
            cmd_line: Full command line of the process

        Returns:
            True if wrong path detected, False otherwise
        """
        return WRONG_DB_PATH_PATTERN in cmd_line

    def log_wrong_path_detection(self, pid: int, cmd_line: str) -> None:
        """
        Log detection of wrong path usage for monitoring.

        Args:
            pid: Process ID
            cmd_line: Full command line of the process
        """
        logger.warning(
            "Wrong database path detected in MCP process",
            pid=pid,
            command_line=cmd_line,
            wrong_path=WRONG_DB_PATH_PATTERN,
            correct_path=CORRECT_DB_PATH,
            detection_time=datetime.now().isoformat()
        )

        # Append to dedicated wrong-path log
        with open(self.wrong_path_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} | PID {pid} | WRONG PATH | {cmd_line}\n")

    async def get_mcp_processes_with_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about running MCP processes.

        Returns:
            List of dictionaries with process details
        """
        try:
            # Get all MCP processes with full command lines
            result = subprocess.run(
                ["pgrep", "-af", str(self.mcp_server_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return []

            processes = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        pid, cmd = parts
                        processes.append({
                            "pid": int(pid),
                            "command_line": cmd,
                            "wrong_path": self.detect_wrong_path_usage(cmd)
                        })

            return processes

        except Exception as e:
            logger.error("Failed to get MCP process details", error=str(e))
            return []

    async def run_cleanup(self) -> Dict[str, Any]:
        """
        Run cleanup check and log results.

        Returns:
            Cleanup result dictionary
        """
        start_time = datetime.utcnow()

        # Get detailed process information
        processes = await self.get_mcp_processes_with_details()
        initial_count = len(processes)

        # Check for wrong path usage
        wrong_path_processes = [p for p in processes if p["wrong_path"]]
        wrong_path_count = len(wrong_path_processes)

        result = {
            "timestamp": start_time.isoformat() + "Z",
            "initial_process_count": initial_count,
            "wrong_path_count": wrong_path_count,
            "cleanup_required": initial_count > self.max_instances or wrong_path_count > 0
        }

        # Log wrong path detections
        for proc in wrong_path_processes:
            self.log_wrong_path_detection(proc["pid"], proc["command_line"])

        # Kill wrong path processes immediately (high priority)
        wrong_path_killed = 0
        for proc in wrong_path_processes:
            try:
                subprocess.run(
                    ["kill", "-9", str(proc["pid"])],
                    check=True,
                    timeout=2
                )
                logger.critical(
                    "Killed MCP process using wrong database path",
                    pid=proc["pid"],
                    command_line=proc["command_line"]
                )
                wrong_path_killed += 1
            except Exception as e:
                logger.error(
                    "Failed to kill wrong-path process",
                    pid=proc["pid"],
                    error=str(e)
                )

        # Update process count after wrong-path kills
        if wrong_path_killed > 0:
            await asyncio.sleep(1)  # Give processes time to terminate
            processes = await self.get_mcp_processes_with_details()
            current_count = len(processes)
        else:
            current_count = initial_count

        # Continue with normal cleanup if still too many processes
        if current_count > self.max_instances:
            cleanup_result = await self.kill_excess_processes()
            result.update(cleanup_result)
        else:
            result.update({
                "killed_count": wrong_path_killed,
                "remaining_count": current_count,
                "status": "wrong_paths_cleaned" if wrong_path_killed > 0 else "ok"
            })

        # Add wrong path cleanup info to result
        result["wrong_path_killed"] = wrong_path_killed

        # Log to file
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        logger.info(
            "MCP cleanup executed",
            initial_count=initial_count,
            wrong_path_count=wrong_path_count,
            wrong_path_killed=wrong_path_killed,
            killed=result.get("killed_count", 0),
            remaining=result.get("remaining_count", 0)
        )

        return result


async def main():
    """Main entry point for cleanup hook."""
    cleanup = MCPCleanupHook()
    result = await cleanup.run_cleanup()

    # Exit with non-zero if cleanup failed
    if result.get("status") == "error":
        sys.exit(1)

    # Silent success (don't pollute hook output)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
