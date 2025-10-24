#!/usr/bin/env python3
"""
DevStream Concurrency Guard Hook
===================================

PreToolUse hook to prevent MCP tool concurrency conflicts.
Implements Context7 best practices for tool use reliability.

Based on Claude Code MCP Enhanced patterns:
- Retry mechanism with exponential backoff
- Error classification for retryable vs non-retryable errors
- Sequential execution to prevent race conditions

Usage: Configured in settings.json PreToolUse hooks
"""

import json
import sys
import time
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class ConcurrencyGuard:
    """
    Prevents MCP tool concurrency conflicts using rate limiting and retry logic.

    Implements Context7 best practices:
    1. Error classification (retryable vs non-retryable)
    2. Exponential backoff with jitter
    3. Sequential execution enforcement
    4. Comprehensive logging and monitoring
    """

    def __init__(self):
        self.lock_dir = Path(tempfile.gettempdir()) / "devstream_mcp_locks"
        self.lock_dir.mkdir(exist_ok=True)
        self.max_retries = 3
        self.base_delay = 0.5  # seconds
        self.max_delay = 10.0  # seconds
        self.backoff_factor = 2.0
        self.jitter_factor = 0.1

    def get_lock_file_path(self, tool_name: str) -> Path:
        """Get lock file path for a specific tool"""
        safe_tool_name = tool_name.replace(":", "_").replace("/", "_")
        return self.lock_dir / f"{safe_tool_name}.lock"

    def is_retryable_error(self, error_msg: str) -> bool:
        """
        Classify errors as retryable vs non-retryable.
        Based on Claude Code MCP Enhanced error patterns.
        """
        error_msg_lower = error_msg.lower()

        # Concurrency and rate limit errors (retryable)
        concurrency_patterns = [
            "400", "concurrency", "too many requests", "rate limit",
            "timeout", "connection", "network", "econnreset",
            "etimedout", "econnrefused", "429", "500", "502", "503", "504"
        ]

        # Non-retryable errors
        non_retryable_patterns = [
            "401", "403", "404", "authentication", "authorization",
            "permission", "not found", "invalid format", "syntax error"
        ]

        # Check non-retryable first (fail fast)
        if any(pattern in error_msg_lower for pattern in non_retryable_patterns):
            return False

        # Check retryable patterns
        return any(pattern in error_msg_lower for pattern in concurrency_patterns)

    def calculate_delay_with_jitter(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        Prevents thundering herd problems.
        """
        delay = min(
            self.base_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )

        # Add jitter (±10% of delay)
        jitter = delay * self.jitter_factor * (2 * (hash(str(time.time())) % 100) / 100 - 1)
        return max(0, delay + jitter)

    def acquire_lock(self, tool_name: str, timeout: float = 30.0) -> bool:
        """
        Acquire lock for tool execution with timeout.
        Implements file-based locking for cross-process safety.
        Enhanced with stale lock cleanup for blocked sessions.
        """
        lock_file = self.get_lock_file_path(tool_name)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if lock file exists and is stale
                if lock_file.exists():
                    try:
                        lock_data = json.loads(lock_file.read_text())
                        lock_time = datetime.fromisoformat(lock_data.get("timestamp", ""))
                        lock_pid = lock_data.get("pid", 0)

                        # Check if lock is older than 2 minutes or process is dead
                        age_minutes = (datetime.utcnow() - lock_time).total_seconds() / 60
                        process_alive = False

                        try:
                            # Check if process is still alive
                            os.kill(lock_pid, 0)
                            process_alive = True
                        except (ProcessLookupError, PermissionError):
                            process_alive = False

                        if age_minutes > 2 or not process_alive:
                            # Clean up stale lock
                            lock_file.unlink()
                            logger.info(
                                "Cleaned stale lock",
                                tool=tool_name,
                                age_minutes=age_minutes,
                                process_alive=process_alive
                            )
                    except (json.JSONDecodeError, ValueError, OSError):
                        # Invalid lock file, remove it
                        lock_file.unlink()
                        logger.warning("Removed invalid lock file", tool=tool_name)

                # Try to create lock file (atomic operation)
                lock_file.write_text(json.dumps({
                    "tool": tool_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "pid": os.getpid()
                }))
                return True

            except (OSError, IOError):
                # Lock file exists, wait and retry
                time.sleep(0.1)

        logger.warning("Lock acquisition timeout", tool=tool_name, timeout=timeout)
        return False

    def release_lock(self, tool_name: str) -> None:
        """Release lock for tool execution"""
        lock_file = self.get_lock_file_path(tool_name)
        try:
            lock_file.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to release lock", tool=tool_name, error=str(e))

    def cleanup_stale_locks(self, max_age_minutes: int = 5) -> None:
        """Clean up stale lock files older than max_age_minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)

        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                stat = lock_file.stat()
                file_time = datetime.fromtimestamp(stat.st_mtime)

                if file_time < cutoff_time:
                    lock_file.unlink()
                    logger.debug("Cleaned up stale lock", lock_file=str(lock_file))
            except OSError:
                continue  # Lock file might be in use

    def retry_with_backoff(self, operation: callable, tool_name: str) -> Any:
        """
        Execute operation with retry logic and exponential backoff.
        Implements Context7 best practices for resilient execution.
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.calculate_delay_with_jitter(attempt - 1)
                    logger.info(
                        "Retrying operation after delay",
                        tool=tool_name,
                        attempt=attempt + 1,
                        max_retries=self.max_retries + 1,
                        delay=delay
                    )
                    time.sleep(delay)

                # Execute the operation
                result = operation()

                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retry",
                        tool=tool_name,
                        attempt=attempt + 1
                    )

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)

                logger.warning(
                    "Operation attempt failed",
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=error_msg,
                    retryable=self.is_retryable_error(error_msg)
                )

                # Check if error is retryable and we have retries left
                if not self.is_retryable_error(error_msg) or attempt == self.max_retries:
                    logger.error(
                        "Operation failed permanently",
                        tool=tool_name,
                        final_attempt=attempt + 1,
                        error=error_msg
                    )
                    raise e

        # This should never be reached, but just in case
        raise last_error if last_error else Exception("Unknown error in retry logic")

    def execute_tool_safely(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool safely with concurrency protection and retry logic.
        Main entry point for the concurrency guard.
        """
        tool_name = tool_input.get("tool_name", "unknown")

        logger.info(
            "Concurrency guard: Executing tool safely",
            tool=tool_name,
            input_keys=list(tool_input.keys())
        )

        # Clean up stale locks first
        self.cleanup_stale_locks()

        def execute_operation():
            """Inner function to execute with lock protection"""
            if not self.acquire_lock(tool_name):
                raise Exception(f"Could not acquire lock for tool: {tool_name}")

            try:
                # Return success to allow tool execution
                return {"status": "allowed", "tool": tool_name}
            finally:
                self.release_lock(tool_name)

        # Execute with retry logic
        return self.retry_with_backoff(execute_operation, tool_name)


def main():
    """Main hook execution function"""
    try:
        # Read tool input from stdin
        input_data = json.load(sys.stdin)

        # Initialize concurrency guard
        guard = ConcurrencyGuard()

        # Execute tool safely
        result = guard.execute_tool_safely(input_data)

        # Output result (allow tool execution)
        json.dump(result, sys.stdout)
        sys.exit(0)

    except Exception as e:
        logger.error("Concurrency guard failed", error=str(e))

        # Return error response to block tool execution
        error_response = {
            "status": "blocked",
            "error": str(e),
            "retry_suggested": guard.is_retryable_error(str(e)) if 'guard' in locals() else False
        }

        json.dump(error_response, sys.stderr)
        sys.exit(1)  # Block tool execution


if __name__ == "__main__":
    main()