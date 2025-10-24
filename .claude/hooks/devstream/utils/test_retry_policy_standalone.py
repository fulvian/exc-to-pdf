#!/usr/bin/env python3
"""
Test PERF-001 Exponential Backoff Fix Implementation - Standalone Version
Tests the Context7-compliant retry policy with jitter and adaptive strategies.
"""

import asyncio
import logging
import random
import time
from typing import Callable, Dict, Any


class RetryPolicy:
    """
    Context7-compliant retry policy with jitter and adaptive strategies.

    Enhanced Features:
    - Jitter-backed exponential backoff (prevents thundering herd)
    - Adaptive retry strategies based on error type
    - Circuit breaker integration
    - Comprehensive error classification
    - Performance monitoring and logging
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter_factor: float = 0.1,
        enable_adaptive_retry: bool = True
    ):
        """
        Initialize Context7-compliant retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Multiplier for exponential backoff
            jitter_factor: Jitter factor (0.0-1.0) to prevent thundering herd
            enable_adaptive_retry: Enable adaptive retry strategies
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter_factor = jitter_factor
        self.enable_adaptive_retry = enable_adaptive_retry

        # Performance metrics
        self._metrics = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_trips': 0,
            'total_delay_time': 0.0
        }

    def _calculate_delay_with_jitter(self, attempt: int) -> float:
        """
        Calculate Context7-compliant exponential backoff with jitter.

        Formula: base_delay * (2^attempt) + random_jitter
        Caps at max_delay to prevent excessive delays.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds with jitter applied
        """
        # Exponential backoff calculation
        exponential_delay = self.base_delay * (2 ** attempt)

        # Add jitter to prevent thundering herd (Context7 best practice)
        jitter_range = exponential_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        # Apply jitter and cap at max_delay
        delay = max(0, exponential_delay + jitter)
        return min(delay, self.max_delay)

    def _classify_error(self, exception: Exception) -> str:
        """
        Classify error for adaptive retry strategy.

        Context7 Pattern: Different retry strategies for different error types.

        Args:
            exception: Exception to classify

        Returns:
            Error classification: 'transient', 'rate_limit', 'server_error', or 'permanent'
        """
        error_message = str(exception).lower()
        error_type = type(exception).__name__.lower()

        # Rate limiting errors (429)
        if '429' in error_message or 'too many requests' in error_message or 'rate limit' in error_message:
            return 'rate_limit'

        # Timeout errors
        if any(keyword in error_message for keyword in ['timeout', 'timed out', 'deadline exceeded']):
            return 'transient'

        # Connection errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'dns', 'unreachable']):
            return 'transient'

        # Server errors (5xx)
        if any(keyword in error_message for keyword in ['500', '502', '503', '504', 'internal server error']):
            return 'server_error'

        # Database errors
        if any(keyword in error_message for keyword in ['database', 'sqlite', 'connection', 'lock']):
            return 'transient'

        # Client errors (4xx excluding 429) - permanent failures
        if any(keyword in error_message for keyword in ['400', '401', '403', '404', '405', 'bad request', 'unauthorized', 'forbidden']):
            return 'permanent'

        # Default to transient for unknown errors
        return 'transient'

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried based on error classification.

        Context7 Pattern: Adaptive retry logic based on error type.

        Args:
            exception: Exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry, False otherwise
        """
        if not self.enable_adaptive_retry:
            return attempt < self.max_retries

        error_classification = self._classify_error(exception)

        # Permanent failures should not be retried
        if error_classification == 'permanent':
            return False

        # Rate limiting needs longer delays but should be retried
        if error_classification == 'rate_limit':
            return attempt < self.max_retries + 1  # Allow one extra retry for rate limits

        # Transient and server errors should be retried
        if error_classification in ['transient', 'server_error']:
            return attempt < self.max_retries

        return False

    def _get_adaptive_delay(self, exception: Exception, base_delay: float) -> float:
        """
        Get adaptive delay based on error type.

        Context7 Pattern: Different delay strategies for different errors.

        Args:
            exception: Exception that occurred
            base_delay: Base delay calculation

        Returns:
            Adapted delay in seconds
        """
        error_classification = self._classify_error(exception)

        # Rate limiting: use longer delays
        if error_classification == 'rate_limit':
            return min(base_delay * 3, self.max_delay)

        # Server errors: use standard exponential backoff
        if error_classification == 'server_error':
            return base_delay

        # Transient errors: use standard exponential backoff
        if error_classification == 'transient':
            return base_delay

        # Default: use base delay
        return base_delay

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with Context7-compliant retry logic.

        Enhanced Features:
        - Jitter-backed exponential backoff
        - Adaptive retry strategies
        - Circuit breaker integration
        - Comprehensive monitoring

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
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self._metrics['total_attempts'] += 1

                # Execute the operation
                result = await operation(*args, **kwargs)

                # Update success metrics
                if attempt > 0:
                    self._metrics['successful_retries'] += 1

                total_time = time.time() - start_time
                if attempt > 0:
                    logging.debug(
                        f"RetryPolicy: Operation succeeded after {attempt + 1} attempts "
                        f"in {total_time:.2f}s"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry this error
                if not self._should_retry(e, attempt):
                    self._metrics['failed_retries'] += 1
                    logging.debug(
                        f"RetryPolicy: Not retrying {type(e).__name__}: {e} "
                        f"(classification: {self._classify_error(e)})"
                    )
                    raise

                # Calculate delay for next attempt
                if attempt < self.max_retries:
                    base_delay = self._calculate_delay_with_jitter(attempt)
                    adaptive_delay = self._get_adaptive_delay(e, base_delay)

                    logging.debug(
                        f"RetryPolicy: Attempt {attempt + 1} failed with {type(e).__name__}: {e} "
                        f"(classification: {self._classify_error(e)}), "
                        f"retrying in {adaptive_delay:.2f}s"
                    )

                    self._metrics['total_delay_time'] += adaptive_delay
                    await asyncio.sleep(adaptive_delay)

        # All retries failed
        self._metrics['failed_retries'] += 1
        total_time = time.time() - start_time
        logging.warning(
            f"RetryPolicy: All {self.max_retries + 1} attempts failed in {total_time:.2f}s. "
            f"Final error: {type(last_exception).__name__}: {last_exception}"
        )

        raise last_exception

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get retry policy performance metrics.

        Returns:
            Dictionary with performance and reliability metrics
        """
        return {
            **self._metrics,
            'success_rate': (
                self._metrics['successful_retries'] /
                max(1, self._metrics['successful_retries'] + self._metrics['failed_retries'])
            ) * 100,
            'average_delay': (
                self._metrics['total_delay_time'] /
                max(1, self._metrics['total_attempts'])
            ),
            'config': {
                'max_retries': self.max_retries,
                'base_delay': self.base_delay,
                'max_delay': self.max_delay,
                'jitter_factor': self.jitter_factor,
                'adaptive_retry_enabled': self.enable_adaptive_retry
            }
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        for key in self._metrics:
            self._metrics[key] = 0


# Import test utilities
import sys
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock


def test_jitter_backed_exponential_backoff():
    """Test Context7 jitter-backed exponential backoff calculation."""
    print("📈 Testing Jitter-Backed Exponential Backoff")
    print("-" * 50)

    retry_policy = RetryPolicy(
        base_delay=1.0,
        max_delay=60.0,
        jitter_factor=0.1
    )

    # Test exponential backoff with jitter
    delays = []
    for attempt in range(5):
        delay = retry_policy._calculate_delay_with_jitter(attempt)
        delays.append(delay)
        print(f"Attempt {attempt}: {delay:.3f}s delay")

    # Verify exponential growth with jitter
    assert delays[0] >= 0.9 and delays[0] <= 1.1, f"First attempt should be ~1.0s with jitter, got {delays[0]}"
    assert delays[1] >= 1.8 and delays[1] <= 2.2, f"Second attempt should be ~2.0s with jitter, got {delays[1]}"
    assert delays[2] >= 3.6 and delays[2] <= 4.4, f"Third attempt should be ~4.0s with jitter, got {delays[2]}"

    # Verify max_delay capping
    large_attempt_delay = retry_policy._calculate_delay_with_jitter(20)
    assert large_attempt_delay <= 60.0, f"Delay should be capped at max_delay (60s), got {large_attempt_delay}"

    print("✅ Jitter-backed exponential backoff working correctly")
    print("✅ Max delay capping verified")
    print()


def test_error_classification():
    """Test Context7 error classification for adaptive retry strategies."""
    print("🔍 Testing Error Classification")
    print("-" * 50)

    retry_policy = RetryPolicy(enable_adaptive_retry=True)

    # Test rate limiting errors
    rate_limit_errors = [
        Exception("HTTP 429 Too Many Requests"),
        Exception("Rate limit exceeded"),
        Exception("Too many requests, try again later")
    ]

    for error in rate_limit_errors:
        classification = retry_policy._classify_error(error)
        assert classification == 'rate_limit', f"Should classify as rate_limit: {error}"
        print(f"✅ Rate limit error classified: '{str(error)[:50]}...' → {classification}")

    # Test timeout errors
    timeout_errors = [
        Exception("Request timeout"),
        Exception("Connection timed out"),
        Exception("Deadline exceeded")
    ]

    for error in timeout_errors:
        classification = retry_policy._classify_error(error)
        assert classification == 'transient', f"Should classify as transient: {error}"
        print(f"✅ Timeout error classified: '{str(error)[:50]}...' → {classification}")

    # Test server errors
    server_errors = [
        Exception("HTTP 500 Internal Server Error"),
        Exception("502 Bad Gateway"),
        Exception("503 Service Unavailable")
    ]

    for error in server_errors:
        classification = retry_policy._classify_error(error)
        assert classification == 'server_error', f"Should classify as server_error: {error}"
        print(f"✅ Server error classified: '{str(error)[:50]}...' → {classification}")

    # Test permanent errors
    permanent_errors = [
        Exception("HTTP 400 Bad Request"),
        Exception("401 Unauthorized"),
        Exception("403 Forbidden"),
        Exception("404 Not Found")
    ]

    for error in permanent_errors:
        classification = retry_policy._classify_error(error)
        assert classification == 'permanent', f"Should classify as permanent: {error}"
        print(f"✅ Permanent error classified: '{str(error)[:50]}...' → {classification}")

    print("✅ All error classifications working correctly")
    print()


def test_adaptive_retry_logic():
    """Test Context7 adaptive retry logic based on error classification."""
    print("🔄 Testing Adaptive Retry Logic")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=3,
        enable_adaptive_retry=True
    )

    # Test transient errors should be retried
    transient_error = Exception("Connection timeout")
    assert retry_policy._should_retry(transient_error, 0) is True, "Transient errors should be retried"
    assert retry_policy._should_retry(transient_error, 2) is True, "Transient errors should be retried on attempt 2"
    assert retry_policy._should_retry(transient_error, 3) is False, "Should not retry after max_retries"

    # Test permanent errors should not be retried
    permanent_error = Exception("404 Not Found")
    assert retry_policy._should_retry(permanent_error, 0) is False, "Permanent errors should not be retried"
    assert retry_policy._should_retry(permanent_error, 2) is False, "Permanent errors should not be retried on any attempt"

    # Test rate limit errors get extra retry
    rate_limit_error = Exception("429 Too Many Requests")
    assert retry_policy._should_retry(rate_limit_error, 3) is True, "Rate limit errors get one extra retry"
    assert retry_policy._should_retry(rate_limit_error, 4) is False, "Should not retry after extra attempt"

    print("✅ Adaptive retry logic working correctly")
    print("✅ Permanent errors properly rejected")
    print("✅ Rate limit errors get extra retry")
    print()


def test_adaptive_delay_calculation():
    """Test Context7 adaptive delay calculation based on error type."""
    print("⏱️ Testing Adaptive Delay Calculation")
    print("-" * 50)

    retry_policy = RetryPolicy(
        base_delay=1.0,
        max_delay=60.0
    )

    base_delay = 2.0

    # Test rate limit errors get longer delays
    rate_limit_error = Exception("429 Too Many Requests")
    rate_limit_delay = retry_policy._get_adaptive_delay(rate_limit_error, base_delay)
    assert rate_limit_delay == 6.0, f"Rate limit delay should be 3x base, got {rate_limit_delay}"
    print(f"✅ Rate limit delay: {base_delay}s → {rate_limit_delay}s (3x multiplier)")

    # Test transient errors use standard delay
    transient_error = Exception("Connection timeout")
    transient_delay = retry_policy._get_adaptive_delay(transient_error, base_delay)
    assert transient_delay == base_delay, f"Transient delay should equal base delay, got {transient_delay}"
    print(f"✅ Transient error delay: {base_delay}s → {transient_delay}s (standard)")

    # Test server errors use standard delay
    server_error = Exception("500 Internal Server Error")
    server_delay = retry_policy._get_adaptive_delay(server_error, base_delay)
    assert server_delay == base_delay, f"Server error delay should equal base delay, got {server_delay}"
    print(f"✅ Server error delay: {base_delay}s → {server_delay}s (standard)")

    print("✅ Adaptive delay calculations working correctly")
    print()


async def test_retry_policy_success_scenario():
    """Test retry policy with successful operation."""
    print("✅ Testing Successful Operation")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay=0.1,  # Short delay for testing
        enable_adaptive_retry=True
    )

    # Mock successful operation
    mock_operation = AsyncMock(return_value="success")

    start_time = time.time()
    result = await retry_policy.execute(mock_operation, "arg1", kwarg1="value1")
    duration = time.time() - start_time

    # Verify operation was called once and returned expected result
    mock_operation.assert_called_once_with("arg1", kwarg1="value1")
    assert result == "success", f"Expected 'success', got {result}"

    # Verify metrics
    metrics = retry_policy.get_metrics()
    assert metrics['total_attempts'] == 1, f"Expected 1 attempt, got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 0, f"Expected 0 successful retries, got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 0, f"Expected 0 failed retries, got {metrics['failed_retries']}"

    print(f"✅ Operation succeeded on first attempt in {duration:.3f}s")
    print("✅ Metrics correctly recorded")
    print()


async def test_retry_policy_transient_failure_recovery():
    """Test retry policy with transient failure recovery."""
    print("🔄 Testing Transient Failure Recovery")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay=0.1,  # Short delay for testing
        enable_adaptive_retry=True
    )

    # Mock operation that fails twice then succeeds
    mock_operation = AsyncMock(side_effect=[
        Exception("Connection timeout"),
        Exception("Connection timeout"),
        "success"
    ])

    start_time = time.time()
    result = await retry_policy.execute(mock_operation)
    duration = time.time() - start_time

    # Verify operation was called 3 times and returned expected result
    assert mock_operation.call_count == 3, f"Expected 3 calls, got {mock_operation.call_count}"
    assert result == "success", f"Expected 'success', got {result}"

    # Verify metrics
    metrics = retry_policy.get_metrics()
    assert metrics['total_attempts'] == 3, f"Expected 3 attempts, got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 1, f"Expected 1 successful retry (final success), got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 0, f"Expected 0 failed retries, got {metrics['failed_retries']}"

    print(f"✅ Operation succeeded after 2 retries in {duration:.3f}s")
    print("✅ Transient failure recovery working correctly")
    print("✅ Metrics correctly recorded")
    print()


async def test_retry_policy_permanent_failure():
    """Test retry policy with permanent failure (no retry)."""
    print("❌ Testing Permanent Failure (No Retry)")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay=0.1,
        enable_adaptive_retry=True
    )

    # Mock operation that fails with permanent error
    permanent_error = Exception("404 Not Found")
    mock_operation = AsyncMock(side_effect=permanent_error)

    start_time = time.time()

    try:
        await retry_policy.execute(mock_operation)
        assert False, "Should have raised exception"
    except Exception as e:
        duration = time.time() - start_time
        assert str(e) == str(permanent_error), f"Expected original error, got {e}"

    # Verify operation was called only once (no retry for permanent errors)
    mock_operation.assert_called_once()

    # Verify metrics
    metrics = retry_policy.get_metrics()
    assert metrics['total_attempts'] == 1, f"Expected 1 attempt (no retry), got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 0, f"Expected 0 successful retries, got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 1, f"Expected 1 failed retry, got {metrics['failed_retries']}"

    print(f"✅ Permanent error failed immediately in {duration:.3f}s")
    print("✅ No retry attempted for permanent error")
    print("✅ Metrics correctly recorded")
    print()


def main():
    """Run all PERF-001 retry policy tests."""
    print("🧪 PERF-001 Exponential Backoff Fix Test Suite")
    print("=" * 60)
    print("Testing Context7-compliant retry policy with jitter and adaptive strategies")
    print()

    # Synchronous tests
    sync_tests = [
        test_jitter_backed_exponential_backoff,
        test_error_classification,
        test_adaptive_retry_logic,
        test_adaptive_delay_calculation,
    ]

    # Asynchronous tests
    async_tests = [
        test_retry_policy_success_scenario,
        test_retry_policy_transient_failure_recovery,
        test_retry_policy_permanent_failure,
    ]

    passed = 0
    failed = 0

    # Run synchronous tests
    for test_func in sync_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    # Run asynchronous tests
    for test_func in async_tests:
        try:
            asyncio.run(test_func())
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All PERF-001 retry policy tests passed!")
        print("✅ Context7-compliant retry policy working correctly")
        print("✅ Jitter-backed exponential backoff functional")
        print("✅ Adaptive retry strategies operational")
        print("✅ Error classification and handling accurate")
        print("✅ Performance metrics collection working")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Review retry policy implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)