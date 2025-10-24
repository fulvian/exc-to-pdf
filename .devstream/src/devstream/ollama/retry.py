"""
Advanced retry mechanisms for Ollama operations.

Based on Context7 research of exponential backoff patterns and best practices.
"""

import asyncio
import random
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar, Tuple
from functools import wraps, partial
import httpx
import ollama

from .config import RetryConfig
from .exceptions import (
    OllamaError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaRetryExhaustedError,
    OllamaServerError,
    map_httpx_error_to_ollama_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
RetryableException = Union[Type[Exception], Tuple[Type[Exception], ...]]


class RetryStatistics:
    """Statistics tracking for retry operations."""

    def __init__(self) -> None:
        self.total_attempts: int = 0
        self.successful_attempts: int = 0
        self.failed_attempts: int = 0
        self.total_delay: float = 0.0
        self.last_attempt_time: Optional[float] = None
        self.exception_counts: Dict[str, int] = {}

    def record_attempt(self, success: bool, delay: float = 0.0, exception: Optional[Exception] = None) -> None:
        """Record an attempt result."""
        self.total_attempts += 1
        self.total_delay += delay
        self.last_attempt_time = time.time()

        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if exception:
                exc_name = type(exception).__name__
                self.exception_counts[exc_name] = self.exception_counts.get(exc_name, 0) + 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100.0

    @property
    def average_delay(self) -> float:
        """Calculate average delay per attempt."""
        failed_attempts = max(self.failed_attempts, 1)  # Avoid division by zero
        return self.total_delay / failed_attempts

    def reset(self) -> None:
        """Reset all statistics."""
        self.__init__()


class ExponentialBackoff:
    """
    Exponential backoff implementation based on Context7 best practices.

    Features:
    - Configurable exponential base and max delay
    - Optional jitter to prevent thundering herd
    - Backoff statistics tracking
    - Circuit breaker pattern support
    """

    def __init__(self, config: RetryConfig) -> None:
        self.config = config
        self.statistics = RetryStatistics()

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate backoff delay for given attempt number.

        Uses exponential backoff with optional jitter.
        """
        if attempt <= 0:
            return 0.0

        # Calculate exponential delay
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled (±25% random variation)
        if self.config.jitter:
            jitter_range = delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay + jitter)  # Ensure minimum delay

        return delay

    async def sleep(self, delay: float) -> None:
        """Sleep for specified delay with cancellation support."""
        if delay > 0:
            logger.debug(f"Backing off for {delay:.2f} seconds")
            await asyncio.sleep(delay)


class RetryHandler:
    """
    Advanced retry handler with Context7-validated patterns.

    Supports:
    - Multiple exception types
    - Custom give-up conditions
    - Async operations
    - Detailed logging and metrics
    """

    def __init__(
        self,
        config: RetryConfig,
        exceptions: RetryableException = (OllamaError, httpx.HTTPError),
        giveup_condition: Optional[Callable[[Exception], bool]] = None,
        on_backoff: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_giveup: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_success: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config
        self.exceptions = exceptions if isinstance(exceptions, tuple) else (exceptions,)
        self.giveup_condition = giveup_condition
        self.on_backoff = on_backoff
        self.on_giveup = on_giveup
        self.on_success = on_success
        self.backoff = ExponentialBackoff(config)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried based on exception and attempt count.
        """
        # Check max retries
        if attempt > self.config.max_retries:
            return False

        # Check if exception is retryable
        if not isinstance(exception, self.exceptions):
            return False

        # Check custom giveup condition
        if self.giveup_condition and self.giveup_condition(exception):
            return False

        # Check specific conditions based on exception type
        if isinstance(exception, httpx.HTTPStatusError):
            # Don't retry on client errors (4xx) unless specifically configured
            status_code = exception.response.status_code
            if 400 <= status_code < 500 and status_code not in self.config.retry_on_status_codes:
                return False

        if isinstance(exception, httpx.TimeoutException) and not self.config.retry_on_timeout:
            return False

        if isinstance(exception, httpx.ConnectError) and not self.config.retry_on_connection_error:
            return False

        return True

    def _create_details_dict(
        self,
        attempt: int,
        target: Callable,
        args: tuple,
        kwargs: dict,
        exception: Optional[Exception] = None,
        wait_time: float = 0.0,
        elapsed_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Create details dictionary for event handlers."""
        return {
            "target": target,
            "args": args,
            "kwargs": kwargs,
            "tries": attempt,
            "wait": wait_time,
            "elapsed": elapsed_time,
            "exception": exception,
        }

    async def retry_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Retry an async function with exponential backoff.

        Based on Context7 async retry patterns.
        """
        start_time = time.time()
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):  # +1 for initial attempt
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - record statistics and call success handler
                elapsed_time = time.time() - start_time
                self.backoff.statistics.record_attempt(success=True)

                if self.on_success:
                    details = self._create_details_dict(
                        attempt, func, args, kwargs, elapsed_time=elapsed_time
                    )
                    self.on_success(details)

                logger.debug(f"Operation succeeded on attempt {attempt}")
                return result

            except Exception as exc:
                last_exception = exc
                elapsed_time = time.time() - start_time

                # Convert HTTPX errors to Ollama-specific errors if needed
                if isinstance(exc, httpx.HTTPError) and not isinstance(exc, OllamaError):
                    exc = map_httpx_error_to_ollama_error(exc, operation=func.__name__)

                # Check if we should retry
                if not self.should_retry(exc, attempt):
                    # Record final failure and call giveup handler
                    self.backoff.statistics.record_attempt(success=False, exception=exc)

                    if self.on_giveup:
                        details = self._create_details_dict(
                            attempt, func, args, kwargs, exc, elapsed_time=elapsed_time
                        )
                        self.on_giveup(details)

                    logger.warning(
                        f"Operation failed after {attempt} attempts: {type(exc).__name__}: {exc}"
                    )

                    # If this is the first attempt and exception is not retryable,
                    # raise the original exception instead of wrapping it
                    if attempt == 1 and not isinstance(exc, self.exceptions):
                        raise exc

                    # Otherwise, we exhausted retries for a retryable exception
                    raise OllamaRetryExhaustedError(
                        operation=func.__name__,
                        max_retries=self.config.max_retries,
                        last_error=exc,
                    ) from exc

                # Calculate backoff delay
                wait_time = self.backoff.calculate_delay(attempt)
                self.backoff.statistics.record_attempt(success=False, delay=wait_time, exception=exc)

                # Call backoff handler
                if self.on_backoff:
                    details = self._create_details_dict(
                        attempt, func, args, kwargs, exc, wait_time, elapsed_time
                    )
                    self.on_backoff(details)

                logger.info(
                    f"Attempt {attempt} failed: {type(exc).__name__}: {exc}. "
                    f"Retrying in {wait_time:.2f} seconds..."
                )

                # Wait before retry
                await self.backoff.sleep(wait_time)

        # This should never be reached due to loop logic, but included for safety
        if last_exception:
            raise OllamaRetryExhaustedError(
                operation=func.__name__,
                max_retries=self.config.max_retries,
                last_error=last_exception,
            ) from last_exception
        else:
            raise OllamaRetryExhaustedError(
                operation=func.__name__,
                max_retries=self.config.max_retries,
            )


def create_default_handlers() -> Dict[str, Callable]:
    """Create default event handlers for retry operations."""

    def backoff_handler(details: Dict[str, Any]) -> None:
        """Default backoff handler with structured logging."""
        logger.info(
            "Backing off",
            extra={
                "retry_attempt": details["tries"],
                "wait_time": details["wait"],
                "function": details["target"].__name__,
                "exception": type(details["exception"]).__name__,
                "elapsed_time": details.get("elapsed", 0),
            },
        )

    def giveup_handler(details: Dict[str, Any]) -> None:
        """Default giveup handler with structured logging."""
        logger.error(
            "Giving up after retries exhausted",
            extra={
                "final_attempt": details["tries"],
                "function": details["target"].__name__,
                "exception": type(details["exception"]).__name__,
                "total_elapsed": details.get("elapsed", 0),
            },
        )

    def success_handler(details: Dict[str, Any]) -> None:
        """Default success handler with structured logging."""
        if details["tries"] > 1:
            logger.info(
                "Operation succeeded after retries",
                extra={
                    "attempts": details["tries"],
                    "function": details["target"].__name__,
                    "total_elapsed": details.get("elapsed", 0),
                },
            )

    return {
        "backoff": backoff_handler,
        "giveup": giveup_handler,
        "success": success_handler,
    }


def with_retry(
    config: RetryConfig,
    exceptions: RetryableException = (OllamaError, httpx.HTTPError),
    giveup_condition: Optional[Callable[[Exception], bool]] = None,
    on_backoff: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_giveup: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_success: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable:
    """
    Decorator for adding retry logic to async functions.

    Based on Context7 decorator patterns for async operations.

    Example:
        @with_retry(config, exceptions=(httpx.HTTPError,))
        async def fetch_data():
            # Implementation
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Use default handlers if none provided
            handlers = create_default_handlers()

            retry_handler = RetryHandler(
                config=config,
                exceptions=exceptions,
                giveup_condition=giveup_condition,
                on_backoff=on_backoff or handlers["backoff"],
                on_giveup=on_giveup or handlers["giveup"],
                on_success=on_success or handlers["success"],
            )

            return await retry_handler.retry_async(func, *args, **kwargs)

        return wrapper

    return decorator


def create_ollama_giveup_condition() -> Callable[[Exception], bool]:
    """
    Create a giveup condition specifically for Ollama operations.

    Based on Context7 research of which errors should not be retried.
    """

    def should_giveup(exception: Exception) -> bool:
        # Don't retry on authentication errors
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            if status_code in (401, 403):  # Unauthorized, Forbidden
                return True

        # Don't retry on client errors (4xx) except specific ones
        if isinstance(exception, OllamaServerError):
            if 400 <= exception.context.get("status_code", 0) < 500:
                # Only retry on specific 4xx codes
                retryable_4xx = {408, 429}  # Request Timeout, Too Many Requests
                return exception.context["status_code"] not in retryable_4xx

        # Don't retry on model validation errors
        if isinstance(exception, ValueError):
            return True

        # Don't retry on configuration errors
        if isinstance(exception, (TypeError, AttributeError)):
            return True

        return False

    return should_giveup


async def retry_with_exponential_backoff(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs,
) -> T:
    """
    Utility function for one-off retry operations.

    Convenient alternative to the decorator for cases where you need
    retry logic without decorating the function definition.
    """
    handlers = create_default_handlers()
    retry_handler = RetryHandler(
        config=config,
        giveup_condition=create_ollama_giveup_condition(),
        on_backoff=handlers["backoff"],
        on_giveup=handlers["giveup"],
        on_success=handlers["success"],
    )

    return await retry_handler.retry_async(func, *args, **kwargs)