#!/usr/bin/env python3
"""
DevStream Logger Adapter - Context7 Compliant
Adapts the existing DevStreamLogger to work with the service locator pattern.

Context7 Pattern: Adapter Pattern
- Wraps existing implementation to match interface
- Enables dependency injection without breaking existing code
- Maintains backward compatibility
"""

from typing import Dict, Any, Optional
from service_interfaces import LoggerInterface


class LoggerAdapter(LoggerInterface):
    """
    Adapter for DevStreamLogger to implement LoggerInterface.

    This adapter wraps the existing DevStreamLogger implementation
    to work with the dependency injection system.
    """

    def __init__(self, devstream_logger=None):
        """
        Initialize logger adapter.

        Args:
            devstream_logger: Existing DevStreamLogger instance
        """
        if devstream_logger is None:
            # Fallback: create basic logger
            from logger import get_devstream_logger
            self._logger = get_devstream_logger('adapter_fallback')
        else:
            self._logger = devstream_logger

    def info(self, event: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(event, **kwargs)

    def debug(self, event: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(event, **kwargs)

    def log_memory_operation(
        self,
        operation: str,
        content_type: str,
        content_size: int,
        memory_id: Optional[str] = None,
        keywords: Optional[list] = None
    ) -> None:
        """Log memory operation."""
        self._logger.log_memory_operation(
            operation=operation,
            content_type=content_type,
            content_size=content_size,
            memory_id=memory_id,
            keywords=keywords
        )

    def log_performance_metrics(
        self,
        execution_time_ms: float,
        memory_usage_mb: Optional[float] = None,
        api_calls: Optional[int] = None
    ) -> None:
        """Log performance metrics."""
        self._logger.log_performance_metrics(
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
        """Log direct database call with Context7 structured logging."""
        if hasattr(self._logger, 'log_direct_call'):
            self._logger.log_direct_call(
                operation=operation,
                parameters=parameters,
                success=success,
                duration_ms=duration_ms,
                result=result,
                error=error
            )
        else:
            # Fallback to standard logging
            if success:
                self.info(f"Direct DB operation {operation} completed in {duration_ms:.2f}ms",
                        extra={"operation": operation, "duration_ms": duration_ms, **parameters})
            else:
                self.error(f"Direct DB operation {operation} failed in {duration_ms:.2f}ms: {error}",
                          extra={"operation": operation, "duration_ms": duration_ms, "error": error, **parameters})

    @property
    def logger(self):
        """Access to underlying logger for compatibility."""
        return self._logger


class PathValidatorAdapter:
    """
    Adapter for path validation to work with dependency injection.
    """

    def __init__(self):
        """Initialize path validator adapter."""
        from path_validator import validate_db_path
        self._validate_db_path = validate_db_path

    def validate_db_path(self, path: str) -> str:
        """
        Validate database path using existing implementation.

        Args:
            path: Path to validate

        Returns:
            Validated path
        """
        return self._validate_db_path(path)


def initialize_service_locator():
    """
    Initialize the service locator with adapter implementations.

    This function should be called once during application startup
    to register all services with their implementations.
    """
    from service_interfaces import service_locator

    # Register logger service
    try:
        from logger import get_devstream_logger
        logger_instance = get_devstream_logger('service_locator')
        service_locator.register_service('logger', LoggerAdapter(logger_instance))
    except Exception as e:
        # Fallback to basic logger
        import logging
        logging.warning(f"Failed to initialize DevStream logger: {e}")
        service_locator.register_service('logger', LoggerAdapter())

    # Register path validator service
    try:
        service_locator.register_service('path_validator', PathValidatorAdapter())
    except Exception as e:
        import logging
        logging.error(f"Failed to initialize path validator: {e}")
        raise


if __name__ == "__main__":
    # Test logger adapter
    print("🧪 Testing Logger Adapter")

    try:
        initialize_service_locator()

        from service_interfaces import get_logger
        logger = get_logger()

        logger.info("Logger adapter working correctly")
        logger.debug("Debug message test")
        logger.warning("Warning message test")

        # Test memory operation logging
        logger.log_memory_operation(
            operation="test",
            content_type="context",
            content_size=100,
            memory_id="test-123",
            keywords=["test", "adapter"]
        )

        # Test performance logging
        logger.log_performance_metrics(
            execution_time_ms=150.5,
            memory_usage_mb=25.3,
            api_calls=3
        )

        print("✅ Logger Adapter: IMPLEMENTED")
        print("✅ Service Locator: INITIALIZED")
        print("✅ Circular Import Prevention: ACTIVE")

    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()