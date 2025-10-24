#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "structlog>=23.0.0",
#     "python-json-logger>=2.0.0",
# ]
# ///

"""
DevStream Structured Logging System - Context7 Compliant
Enhanced logging per DevStream hook system con structured output.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
import os
import tempfile

class DevStreamLogger:
    """
    Structured logger per DevStream hooks following Context7 patterns.
    Provides JSON logging con context tracking.
    """

    def __init__(self, hook_name: str, log_level: str = "INFO"):
        """
        Initialize DevStream logger.

        Args:
            hook_name: Nome del hook (e.g., 'user_prompt_submit')
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.hook_name = hook_name

        log_dir_env = os.getenv('DEVSTREAM_LOG_DIR')
        if log_dir_env:
            candidate_dir = Path(log_dir_env).expanduser()
        else:
            project_root = os.getenv('DEVSTREAM_PROJECT_ROOT')
            if project_root:
                candidate_dir = Path(project_root) / '.claude' / 'logs' / 'devstream'
            else:
                candidate_dir = Path.home() / '.claude' / 'logs' / 'devstream'

        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = candidate_dir
        except PermissionError:
            fallback_dir = Path(tempfile.gettempdir()) / 'devstream' / 'logs'
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = fallback_dir

        # Setup structured logging
        self._setup_structlog(log_level)

        # Get logger instance
        self.logger = structlog.get_logger(hook_name)

    def _setup_structlog(self, log_level: str) -> None:
        """
        Configure structlog per DevStream needs.

        Context7 Best Practice: Use structlog.configure with cache_logger_on_first_use
        for thread-safe, performance-optimized logging without logging.basicConfig issues.

        CRITICAL FIX: Check if structlog is already configured to prevent "threads can only be started once" error.
        structlog.configure() can only be called once per process.

        Args:
            log_level: Logging level string
        """
        # CRITICAL FIX: Check if structlog is already configured (Context7 best practice)
        # This prevents the "threads can only be started once" error
        if structlog.is_configured():
            # Structlog already configured, just setup our specific handlers
            self._setup_handlers_only(log_level)
            return

        # Configure structlog ONLY if not already configured (Context7 pattern)
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
            cache_logger_on_first_use=True,  # Performance optimization + thread safety
        )

        # Setup standard logging handlers manually (Context7 pattern - avoid basicConfig)
        self._setup_handlers_only(log_level)

    def _setup_handlers_only(self, log_level: str) -> None:
        """
        Setup logging handlers only, without reconfiguring structlog.

        Called when structlog is already configured to prevent duplicate configuration.

        Args:
            log_level: Logging level string
        """
        # Setup standard logging handlers manually (Context7 pattern - avoid basicConfig)
        root_logger = logging.getLogger()

        hook_handler_name = f'devstream_{self.hook_name}_handler'
        expected_path = self.log_dir / f'{self.hook_name}.jsonl'
        for handler in list(root_logger.handlers):
            if getattr(handler, 'name', None) == hook_handler_name:
                current_path = Path(getattr(handler, 'baseFilename', expected_path))
                if current_path != expected_path:
                    root_logger.removeHandler(handler)
                    handler.close()
                else:
                    return  # Handler already configured with correct path

        # Set level if not already set
        if not root_logger.level or root_logger.level == logging.NOTSET:
            root_logger.setLevel(getattr(logging, log_level.upper()))

        # Add file handler with unique name
        log_file_path = self.log_dir / f'{self.hook_name}.jsonl'
        try:
            file_handler = logging.FileHandler(log_file_path)
        except PermissionError:
            fallback_dir = Path(tempfile.gettempdir()) / 'devstream' / 'logs'
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = fallback_dir
            log_file_path = self.log_dir / f'{self.hook_name}.jsonl'
            file_handler = logging.FileHandler(log_file_path)

        file_handler.setFormatter(logging.Formatter('%(message)s'))
        file_handler.name = hook_handler_name  # Name to track and avoid duplicates
        root_logger.addHandler(file_handler)

        # Add stderr handler (only add once)
        stderr_handler_name = 'devstream_stderr_handler'
        has_stderr = any(getattr(h, 'name', None) == stderr_handler_name for h in root_logger.handlers)
        if not has_stderr:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            stream_handler.name = stderr_handler_name
            root_logger.addHandler(stream_handler)

    def log_hook_start(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log hook start with input data.

        Args:
            input_data: Input JSON data from Claude Code
            context: Additional context information
        """
        self.logger.info(
            "Hook execution started",
            hook_event="hook_start",
            hook=self.hook_name,
            input_size=len(json.dumps(input_data)),
            session_id=input_data.get('session_id', 'unknown'),
            hook_context=context or {}
        )

    def log_hook_success(
        self,
        result: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log successful hook completion.

        Args:
            result: Hook result data
            metrics: Performance metrics
        """
        self.logger.info(
            "Hook execution completed successfully",
            hook_event="hook_success",
            hook=self.hook_name,
            result=result or {},
            metrics=metrics or {}
        )

    def log_hook_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log hook error with context.

        Args:
            error: Exception that occurred
            context: Error context information
        """
        self.logger.error(
            "Hook execution failed",
            hook_event="hook_error",
            hook=self.hook_name,
            error=str(error),
            error_type=type(error).__name__,
            hook_context=context or {}
        )

    def log_mcp_call(
        self,
        tool: str,
        parameters: Dict[str, Any],
        success: bool,
        response: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log MCP tool call.

        Args:
            tool: MCP tool name
            parameters: Tool parameters
            success: Call success status
            response: MCP response data
            error: Error message if failed
        """
        self.logger.info(
            "MCP tool call",
            hook_event="mcp_call",
            hook=self.hook_name,
            tool=tool,
            parameters=parameters,
            success=success,
            response=response or {},
            error=error
        )

    def log_context_injection(
        self,
        context_type: str,
        content_size: int,
        keywords: Optional[list] = None
    ) -> None:
        """
        Log context injection event.

        Args:
            context_type: Type of context injected
            content_size: Size of injected content
            keywords: Extracted keywords
        """
        self.logger.info(
            "Context injected",
            hook_event="context_injection",
            hook=self.hook_name,
            context_type=context_type,
            content_size=content_size,
            keywords=keywords or []
        )

    def log_memory_operation(
        self,
        operation: str,
        content_type: str,
        content_size: int,
        memory_id: Optional[str] = None,
        keywords: Optional[list] = None
    ) -> None:
        """
        Log memory storage/retrieval operations.

        Args:
            operation: Operation type (store/retrieve/search)
            content_type: DevStream content type
            content_size: Content size in characters
            memory_id: Memory entry ID
            keywords: Associated keywords
        """
        self.logger.info(
            "Memory operation",
            hook_event="memory_operation",
            hook=self.hook_name,
            operation=operation,
            content_type=content_type,
            content_size=content_size,
            memory_id=memory_id,
            keywords=keywords or []
        )

    def log_task_operation(
        self,
        operation: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        priority: Optional[int] = None,
        status: Optional[str] = None
    ) -> None:
        """
        Log task management operations.

        Args:
            operation: Operation type (create/update/complete)
            task_id: Task identifier
            task_type: Type of task
            priority: Task priority
            status: Task status
        """
        self.logger.info(
            "Task operation",
            hook_event="task_operation",
            hook=self.hook_name,
            operation=operation,
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            status=status
        )

    def log_performance_metrics(
        self,
        execution_time_ms: float,
        memory_usage_mb: Optional[float] = None,
        api_calls: Optional[int] = None
    ) -> None:
        """
        Log performance metrics.

        Args:
            execution_time_ms: Execution time in milliseconds
            memory_usage_mb: Memory usage in MB
            api_calls: Number of API calls made
        """
        self.logger.info(
            "Performance metrics",
            hook_event="performance",
            hook=self.hook_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            api_calls=api_calls
        )

    def log_direct_call(
        self,
        operation: str,
        parameters: Dict[str, Any],
        success: bool,
        duration_ms: float,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log direct database call (replacing MCP calls).

        Args:
            operation: Database operation performed
            parameters: Operation parameters
            success: Operation success status
            duration_ms: Duration in milliseconds
            result: Operation result if successful
            error: Error message if failed
        """
        self.logger.info(
            "Direct database call",
            hook_event="direct_call",
            hook=self.hook_name,
            operation=operation,
            parameters=parameters,
            success=success,
            duration_ms=duration_ms,
            result=result or {},
            error=error
        )

    # Delegate methods for standard logging (Context7 pattern)
    def info(self, event: str, **kwargs) -> None:
        """Log info message with structured context."""
        self.logger.info(event, hook=self.hook_name, **kwargs)

    def debug(self, event: str, **kwargs) -> None:
        """Log debug message with structured context."""
        self.logger.debug(event, hook=self.hook_name, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """Log warning message with structured context."""
        self.logger.warning(event, hook=self.hook_name, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        """Log error message with structured context."""
        self.logger.error(event, hook=self.hook_name, **kwargs)

def get_devstream_logger(hook_name: str, log_level: str = "INFO") -> DevStreamLogger:
    """
    Get DevStream logger instance.

    Args:
        hook_name: Hook name
        log_level: Logging level

    Returns:
        DevStreamLogger instance
    """
    return DevStreamLogger(hook_name, log_level)

# Example usage for testing
if __name__ == "__main__":
    # Test logger
    logger = get_devstream_logger("test_hook")

    logger.log_hook_start(
        input_data={"session_id": "test-123", "prompt": "test prompt"},
        context={"project": "DevStream"}
    )

    logger.log_memory_operation(
        operation="store",
        content_type="context",
        content_size=100,
        memory_id="test-memory-123",
        keywords=["test", "devstream"]
    )

    logger.log_hook_success(
        result={"status": "completed"},
        metrics={"execution_time_ms": 150.5}
    )

    print("✅ DevStream logging system test completed")
    print(f"📁 Logs written to: {Path.home() / '.claude' / 'logs' / 'devstream'}")
