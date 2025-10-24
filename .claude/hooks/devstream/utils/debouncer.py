#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
DevStream AsyncDebouncer - Prevent Rapid Hook Executions

Provides decorator-based debouncing for async functions to reduce macOS resource
exhaustion from rapid hook executions. Uses monotonic timing for reliability.

Context7 Research Applied:
- Pattern: Python asyncio decorator-based debouncing
- Performance: O(1) lookup, <1ms overhead
- Thread safety: asyncio event loop compatible

Example:
    >>> from debouncer import AsyncDebouncer, debounce
    >>>
    >>> # Method 1: Global instance
    >>> @debounce(delay=0.2)
    >>> async def process_hook():
    ...     # This will skip if called <200ms after last execution
    ...     await expensive_operation()
    >>>
    >>> # Method 2: Custom instance
    >>> custom_debouncer = AsyncDebouncer(delay=0.5)
    >>> @custom_debouncer
    >>> async def critical_hook():
    ...     # This will skip if called <500ms after last execution
    ...     await critical_operation()
"""

import asyncio
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable, TypeVar, ParamSpec
import logging

# Type variables for generic decorator
P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)


class AsyncDebouncer:
    """
    Debouncer for async functions to prevent rapid executions.

    Uses monotonic timing to track last execution time per function and skips
    execution if called within the debounce delay period.

    Attributes:
        delay: Minimum time between executions (seconds)
        _last_execution: Dict mapping function names to last execution timestamps
        _execution_count: Dict tracking execution statistics per function
        _debounced_count: Dict tracking debounced (skipped) calls per function

    Example:
        >>> debouncer = AsyncDebouncer(delay=0.1)
        >>>
        >>> @debouncer
        >>> async def my_hook():
        ...     await process_data()
        >>>
        >>> # First call executes
        >>> await my_hook()  # Executes
        >>>
        >>> # Immediate second call is debounced
        >>> await my_hook()  # Returns None (debounced)
        >>>
        >>> # After delay, executes again
        >>> await asyncio.sleep(0.2)
        >>> await my_hook()  # Executes

    Note:
        - Uses time.monotonic() for monotonic timing (unaffected by system clock changes)
        - Returns None if execution is debounced
        - Thread-safe for asyncio event loops (not multi-threading)
    """

    def __init__(self, delay: float = 0.1):
        """
        Initialize AsyncDebouncer.

        Args:
            delay: Minimum time between executions in seconds (default: 100ms)

        Raises:
            ValueError: If delay <= 0
            TypeError: If delay is not a number
        """
        if not isinstance(delay, (int, float)):
            raise TypeError(
                f"delay must be a number, got {type(delay).__name__}"
            )
        if delay <= 0:
            raise ValueError(
                f"delay must be positive, got {delay}"
            )

        self.delay: float = float(delay)
        self._last_execution: Dict[str, float] = {}
        self._execution_count: Dict[str, int] = {}
        self._debounced_count: Dict[str, int] = {}

        logger.debug(
            f"AsyncDebouncer initialized with delay={self.delay}s"
        )

    def __call__(
        self,
        func: Callable[P, R]
    ) -> Callable[P, Optional[R]]:
        """
        Decorator to debounce an async function.

        Args:
            func: Async function to debounce

        Returns:
            Wrapped async function with debouncing logic

        Raises:
            TypeError: If func is not an async function
        """
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"AsyncDebouncer can only debounce async functions, "
                f"got {type(func).__name__}"
            )

        func_name = func.__name__

        # Initialize tracking for this function
        self._last_execution[func_name] = 0.0
        self._execution_count[func_name] = 0
        self._debounced_count[func_name] = 0

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            """
            Wrapper that implements debouncing logic.

            Returns:
                Function result if executed, None if debounced
            """
            current_time = time.monotonic()
            last_time = self._last_execution[func_name]
            time_since_last = current_time - last_time

            # Check if we should debounce (skip execution)
            if time_since_last < self.delay:
                self._debounced_count[func_name] += 1
                logger.debug(
                    f"Debounced {func_name}: "
                    f"time_since_last={time_since_last:.3f}s < "
                    f"delay={self.delay}s"
                )
                return None

            # Execute the function
            try:
                logger.debug(
                    f"Executing {func_name}: "
                    f"time_since_last={time_since_last:.3f}s >= "
                    f"delay={self.delay}s"
                )
                result = await func(*args, **kwargs)

                # Update tracking
                self._last_execution[func_name] = time.monotonic()
                self._execution_count[func_name] += 1

                return result

            except Exception as e:
                logger.error(
                    f"Error in debounced function {func_name}: {e}",
                    exc_info=True
                )
                raise

        return wrapper

    def reset(self, func: Optional[Callable] = None) -> None:
        """
        Reset debouncer state.

        Args:
            func: Optional specific function to reset. If None, reset all functions.

        Example:
            >>> debouncer = AsyncDebouncer(delay=0.1)
            >>>
            >>> @debouncer
            >>> async def my_hook():
            ...     pass
            >>>
            >>> # Reset specific function
            >>> debouncer.reset(my_hook)
            >>>
            >>> # Reset all functions
            >>> debouncer.reset()
        """
        if func is None:
            # Reset all functions
            count = len(self._last_execution)
            self._last_execution.clear()
            self._execution_count.clear()
            self._debounced_count.clear()
            logger.info(f"Reset all {count} debounced functions")
        else:
            # Reset specific function
            func_name = func.__name__
            if func_name in self._last_execution:
                self._last_execution[func_name] = 0.0
                self._execution_count[func_name] = 0
                self._debounced_count[func_name] = 0
                logger.info(f"Reset debouncer for {func_name}")
            else:
                logger.warning(
                    f"Function {func_name} not found in debouncer"
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get debouncer statistics.

        Returns:
            Dictionary with statistics per function:
            - executions: Number of actual executions
            - debounced: Number of debounced (skipped) calls
            - total_calls: Total calls (executions + debounced)
            - debounce_rate: Percentage of calls that were debounced

        Example:
            >>> debouncer = AsyncDebouncer(delay=0.1)
            >>> stats = debouncer.get_stats()
            >>> print(stats)
            {
                'my_hook': {
                    'executions': 10,
                    'debounced': 45,
                    'total_calls': 55,
                    'debounce_rate': 81.82
                }
            }
        """
        stats = {}

        for func_name in self._last_execution.keys():
            executions = self._execution_count.get(func_name, 0)
            debounced = self._debounced_count.get(func_name, 0)
            total_calls = executions + debounced

            stats[func_name] = {
                'executions': executions,
                'debounced': debounced,
                'total_calls': total_calls,
                'debounce_rate': (
                    (debounced / total_calls * 100) if total_calls > 0 else 0.0
                )
            }

        return stats


# Global shared debouncer instance (100ms delay)
# Use this for consistent debouncing across all DevStream hooks
hook_debouncer = AsyncDebouncer(delay=0.1)


def debounce(delay: float = 0.1) -> Callable:
    """
    Convenience decorator factory for debouncing async functions.

    Creates a new AsyncDebouncer instance with specified delay and returns
    its __call__ method as a decorator.

    Args:
        delay: Minimum time between executions in seconds (default: 100ms)

    Returns:
        Decorator function for debouncing

    Raises:
        ValueError: If delay <= 0
        TypeError: If delay is not a number

    Example:
        >>> @debounce(delay=0.2)
        >>> async def process_hook():
        ...     await expensive_operation()
        >>>
        >>> # First call executes
        >>> await process_hook()  # Executes
        >>>
        >>> # Immediate second call is debounced
        >>> await process_hook()  # Returns None (debounced)
    """
    debouncer = AsyncDebouncer(delay=delay)
    return debouncer


# Performance validation function
def validate_performance() -> bool:
    """
    Validate that debouncer overhead is <1ms.

    Returns:
        True if performance meets requirements

    Raises:
        AssertionError: If overhead exceeds 1ms
    """
    import asyncio

    async def dummy_func():
        """Dummy async function for performance testing."""
        pass

    debouncer = AsyncDebouncer(delay=0.001)
    wrapped = debouncer(dummy_func)

    # Measure overhead
    start = time.monotonic()
    asyncio.run(wrapped())
    overhead = (time.monotonic() - start) * 1000  # Convert to ms

    logger.info(f"Debouncer overhead: {overhead:.3f}ms")

    assert overhead < 1.0, f"Overhead {overhead:.3f}ms exceeds 1ms limit"
    return True


if __name__ == "__main__":
    # Performance validation
    print("AsyncDebouncer Performance Validation")
    print("=" * 50)

    try:
        validate_performance()
        print("✅ Performance validation passed: overhead <1ms")
    except AssertionError as e:
        print(f"❌ Performance validation failed: {e}")

    # Statistics example
    print("\nStatistics Example:")
    print("=" * 50)

    async def demo():
        """Demonstrate debouncer statistics."""
        # Create a debouncer instance for tracking
        demo_debouncer = AsyncDebouncer(delay=0.05)

        @demo_debouncer
        async def test_hook():
            """Test hook for statistics."""
            pass

        # Execute multiple times
        for i in range(10):
            await test_hook()
            if i % 2 == 0:
                await asyncio.sleep(0.06)  # Allow some executions

        # Get stats from the debouncer instance
        stats = demo_debouncer.get_stats()
        print(f"Statistics: {stats}")

    asyncio.run(demo())
    print("\n✅ AsyncDebouncer implementation complete")
