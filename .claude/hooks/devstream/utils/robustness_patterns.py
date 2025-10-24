#!/usr/bin/env python3
"""
DevStream Robustness Patterns

Context7-inspired robustness patterns for database operations.
Provides interrupt handling, async context managers, and error recovery patterns.

Key Features:
- Interrupt handling for graceful cancellation
- Async context managers for proper resource management
- Connection retry logic with exponential backoff
- Health monitoring and circuit breaker patterns
- Timeout management with configurable defaults
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime, timedelta


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RobustnessConfig:
    """Configuration for robustness patterns."""

    # Timeout configuration
    default_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_base_delay: float = 1.0

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_success_threshold: int = 3

    # Health check configuration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5

    # Resource limits
    max_concurrent_operations: int = 10
    operation_queue_timeout_seconds: int = 60


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.

    Prevents repeated calls to failing services by implementing
    a three-state pattern: CLOSED → OPEN → HALF_OPEN → CLOSED.
    """

    def __init__(self, config: RobustnessConfig):
        """
        Initialize circuit breaker.

        Args:
            config: Robustness configuration
        """
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker_{id(self)}")

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    def _should_allow_request(self) -> bool:
        """
        Check if request should be allowed based on current state.

        Returns:
            True if request should proceed, False otherwise
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                # Check if timeout has elapsed
                if (self._last_failure_time and
                    (datetime.now() - self._last_failure_time).total_seconds() >
                    self.config.circuit_breaker_timeout_seconds):
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                return True

    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.circuit_breaker_success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self.logger.info("Circuit breaker transitioning to CLOSED")
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        """Handle failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.circuit_breaker_failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    self.logger.warning(
                        f"Circuit breaker transitioning to OPEN "
                        f"(failures: {self._failure_count})"
                    )
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._state = CircuitBreakerState.OPEN
                self.logger.warning("Circuit breaker returning to OPEN from HALF_OPEN")

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with circuit breaker protection.

        Args:
            operation: Async callable to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Result of operation if successful

        Raises:
            Exception: If operation fails or circuit breaker is open
        """
        if not self._should_allow_request():
            raise ConnectionError("Circuit breaker is OPEN - service unavailable")

        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise


class OperationLimiter:
    """
    Limits concurrent operations to prevent resource exhaustion.

    Uses asyncio.Semaphore to control the number of concurrent operations
    and provides timeout handling for queued operations.
    """

    def __init__(self, config: RobustnessConfig):
        """
        Initialize operation limiter.

        Args:
            config: Robustness configuration
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        self.logger = logging.getLogger(f"operation_limiter_{id(self)}")

    @asynccontextmanager
    async def acquire(self, timeout_seconds: Optional[float] = None):
        """
        Acquire operation slot with timeout.

        Args:
            timeout_seconds: Custom timeout (uses config default if None)

        Yields:
            None when slot is acquired

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        timeout = timeout_seconds or self.config.operation_queue_timeout_seconds

        try:
            # Acquire semaphore with timeout
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )
            yield
        finally:
            self._semaphore.release()

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with concurrency limiting.

        Args:
            operation: Async callable to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Result of operation
        """
        async with self.acquire():
            return await operation(*args, **kwargs)


async def with_interrupt_handling(
    operation: Callable,
    timeout_seconds: int = 30
) -> Any:
    """
    Execute operation with interrupt handling capability.

    Uses APSW-style interrupt patterns for graceful cancellation.
    Implements timeout management and proper cleanup.

    Args:
        operation: Async callable to execute
        timeout_seconds: Maximum execution time

    Example:
        result = await with_interrupt_handling(
            lambda: conn.execute("SELECT ..."),
            timeout_seconds=30
        )
    """
    try:
        # Execute with timeout
        result = await asyncio.wait_for(
            operation(),
            timeout=timeout_seconds
        )
        return result

    except asyncio.TimeoutError:
        # Log timeout with context
        logger = logging.getLogger("interrupt_handler")
        logger.warning(
            f"Operation timed out after {timeout_seconds} seconds",
            extra={"operation": operation.__name__ if hasattr(operation, '__name__') else str(operation)}
        )
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    except Exception as e:
        # Log and re-raise other exceptions
        logger = logging.getLogger("interrupt_handler")
        logger.error(
            f"Operation failed with interrupt handling",
            extra={
                "error": str(e),
                "operation": operation.__name__ if hasattr(operation, '__name__') else str(operation)
            }
        )
        raise


@asynccontextmanager
async def database_connection_context(
    connection_factory: Callable,
    timeout_seconds: int = 30
):
    """
    Async context manager for database connections with proper cleanup.

    Args:
        connection_factory: Callable that returns database connection
        timeout_seconds: Connection timeout

    Yields:
        Database connection object
    """
    conn = None
    try:
        # Acquire connection with timeout
        conn = await asyncio.wait_for(
            connection_factory(),
            timeout=timeout_seconds
        )

        # Verify connection is healthy
        if hasattr(conn, 'ping'):
            await conn.ping()
        elif hasattr(conn, 'execute'):
            # Simple health check
            await conn.execute("SELECT 1")

        yield conn

    except Exception as e:
        logger = logging.getLogger("db_context")
        logger.error(
            f"Database connection context failed",
            extra={"error": str(e), "timeout": timeout_seconds}
        )
        raise

    finally:
        # Ensure connection is properly closed
        if conn:
            try:
                if hasattr(conn, 'close'):
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
            except Exception as close_error:
                logger = logging.getLogger("db_context")
                logger.warning(
                    f"Failed to close database connection",
                    extra={"error": str(close_error)}
                )


class RetryPolicy:
    """
    Configurable retry policy for resilient operations.

    Supports exponential backoff, jitter, and different retry strategies
    based on exception types.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries
            backoff_factor: Multiplier for exponential backoff
            max_delay: Maximum delay between retries
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.logger = logging.getLogger(f"retry_policy_{id(self)}")

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.

        Args:
            attempt: Retry attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )

        if self.jitter:
            # Add random jitter (±25%)
            import random
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried based on exception.

        Args:
            exception: Exception that occurred
            attempt: Current retry attempt

        Returns:
            True if operation should be retried
        """
        if attempt >= self.max_retries:
            return False

        # Don't retry certain exception types
        non_retryable_exceptions = (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            PermissionError
        )

        if isinstance(exception, non_retryable_exceptions):
            return False

        # Retry on network and temporary failures
        retryable_patterns = [
            'connection',
            'timeout',
            'temporary',
            'busy',
            'locked',
            'unavailable',
            'rate limit',
            'overloaded'
        ]

        error_msg = str(exception).lower()
        return any(pattern in error_msg for pattern in retryable_patterns)

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation: Async callable to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Result of operation

        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self._should_retry(e, attempt):
                    self.logger.debug(
                        f"Exception not retryable: {e}",
                        extra={"attempt": attempt, "exception_type": type(e).__name__}
                    )
                    raise

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(
                        f"Retrying operation in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})",
                        extra={
                            "attempt": attempt,
                            "delay": delay,
                            "exception": str(e)[:200],
                            "exception_type": type(e).__name__
                        }
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"Operation failed after {self.max_retries + 1} attempts",
                        extra={
                            "attempts": attempt + 1,
                            "final_exception": str(e)[:200],
                            "exception_type": type(e).__name__
                        }
                    )

        # Raise the last exception if all retries failed
        raise last_exception


class HealthMonitor:
    """
    Health monitoring for database connections and operations.

    Tracks health metrics, detects degradation, and provides
    recommendations for recovery actions.
    """

    def __init__(self, config: RobustnessConfig):
        """
        Initialize health monitor.

        Args:
            config: Robustness configuration
        """
        self.config = config
        self._health_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_response_time': 0.0,
            'last_health_check': None,
            'consecutive_failures': 0,
            'last_error': None
        }
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"health_monitor_{id(self)}")

    def record_operation(self, success: bool, duration_ms: float, error: Optional[Exception] = None) -> None:
        """
        Record operation outcome for health tracking.

        Args:
            success: Whether operation was successful
            duration_ms: Operation duration in milliseconds
            error: Exception if operation failed
        """
        with self._lock:
            self._health_stats['total_operations'] += 1

            if success:
                self._health_stats['successful_operations'] += 1
                self._health_stats['consecutive_failures'] = 0
            else:
                self._health_stats['failed_operations'] += 1
                self._health_stats['consecutive_failures'] += 1
                self._health_stats['last_error'] = str(error)[:200] if error else None

            # Update average response time
            total_ops = self._health_stats['total_operations']
            current_avg = self._health_stats['average_response_time']
            self._health_stats['average_response_time'] = (
                (current_avg * (total_ops - 1) + duration_ms) / total_ops
            )

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.

        Returns:
            Dictionary with health metrics
        """
        with self._lock:
            stats = self._health_stats.copy()

            # Calculate success rate
            if stats['total_operations'] > 0:
                stats['success_rate'] = (
                    stats['successful_operations'] / stats['total_operations']
                )
            else:
                stats['success_rate'] = 1.0

            # Determine health status
            if stats['success_rate'] >= 0.95 and stats['consecutive_failures'] == 0:
                stats['health'] = 'healthy'
            elif stats['success_rate'] >= 0.8 and stats['consecutive_failures'] <= 3:
                stats['health'] = 'degraded'
            else:
                stats['health'] = 'unhealthy'

            # Add recommendations
            stats['recommendations'] = self._get_recommendations(stats)

            return stats

    def _get_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """
        Generate health recommendations based on current stats.

        Args:
            stats: Current health statistics

        Returns:
            List of recommendations
        """
        recommendations = []

        if stats['success_rate'] < 0.8:
            recommendations.append("Success rate below 80% - investigate underlying issues")

        if stats['consecutive_failures'] >= 5:
            recommendations.append("Multiple consecutive failures - check service availability")

        if stats['average_response_time'] > 5000:  # 5 seconds
            recommendations.append("High response time detected - consider optimization")

        if stats['total_operations'] == 0:
            recommendations.append("No operations recorded - check monitoring configuration")

        return recommendations


# Factory function for creating configured robustness components
def create_robustness_components(config: Optional[RobustnessConfig] = None) -> Dict[str, Any]:
    """
    Create a complete set of robustness components.

    Args:
        config: Optional configuration (uses defaults if None)

    Returns:
        Dictionary with all robustness components
    """
    if config is None:
        config = RobustnessConfig()

    return {
        'circuit_breaker': CircuitBreaker(config),
        'operation_limiter': OperationLimiter(config),
        'retry_policy': RetryPolicy(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            backoff_factor=config.retry_backoff_factor
        ),
        'health_monitor': HealthMonitor(config),
        'config': config
    }


# Example usage
async def example_robust_operation():
    """Example showing how to use robustness patterns."""
    # Create robustness components
    components = create_robustness_components()

    # Define operation to execute
    async def database_operation():
        # Simulate database operation
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "success", "data": "example"}

    try:
        # Execute with all robustness patterns
        result = await components['circuit_breaker'].call(
            components['retry_policy'].execute,
            components['operation_limiter'].call,
            database_operation
        )

        # Record success
        components['health_monitor'].record_operation(
            success=True,
            duration_ms=100
        )

        return result

    except Exception as e:
        # Record failure
        components['health_monitor'].record_operation(
            success=False,
            duration_ms=0,
            error=e
        )

        # Get health status
        health = components['health_monitor'].get_health_status()
        print(f"Operation failed. Health status: {health['health']}")

        raise


if __name__ == "__main__":
    # Test robustness patterns
    print("🧪 Testing robustness patterns...")

    async def test():
        try:
            result = await example_robust_operation()
            print(f"✅ Test successful: {result}")
        except Exception as e:
            print(f"❌ Test failed: {e}")

    asyncio.run(test())