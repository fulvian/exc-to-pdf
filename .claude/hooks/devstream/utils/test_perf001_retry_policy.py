#!/usr/bin/env python3
"""
Test PERF-001 Exponential Backoff Fix Implementation
Tests the Context7-compliant retry policy with jitter and adaptive strategies.
"""

import sys
import os
import asyncio
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RetryPolicy directly from unified_client module
import importlib.util
spec = importlib.util.spec_from_file_location("unified_client", Path(__file__).parent / "unified_client.py")
unified_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified_client)

RetryPolicy = unified_client.RetryPolicy


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
    assert metrics['successful_retries'] == 2, f"Expected 2 successful retries, got {metrics['successful_retries']}"
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


async def test_retry_policy_exhaustion():
    """Test retry policy with retry exhaustion."""
    print("😴 Testing Retry Exhaustion")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=2,
        base_delay=0.1,  # Short delay for testing
        enable_adaptive_retry=True
    )

    # Mock operation that always fails with transient error
    transient_error = Exception("Connection timeout")
    mock_operation = AsyncMock(side_effect=transient_error)

    start_time = time.time()

    try:
        await retry_policy.execute(mock_operation)
        assert False, "Should have raised exception"
    except Exception as e:
        duration = time.time() - start_time
        assert str(e) == str(transient_error), f"Expected original error, got {e}"

    # Verify operation was called max_retries + 1 times
    assert mock_operation.call_count == 3, f"Expected 3 calls (2 retries), got {mock_operation.call_count}"

    # Verify metrics
    metrics = retry_policy.get_metrics()
    assert metrics['total_attempts'] == 3, f"Expected 3 attempts, got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 0, f"Expected 0 successful retries, got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 1, f"Expected 1 failed retry, got {metrics['failed_retries']}"

    print(f"✅ All retries exhausted in {duration:.3f}s")
    print("✅ Operation called 3 times (1 initial + 2 retries)")
    print("✅ Metrics correctly recorded")
    print()


async def test_retry_policy_rate_limiting():
    """Test retry policy with rate limiting (extra retry + longer delay)."""
    print("🚦 Testing Rate Limiting Handling")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=2,
        base_delay=0.1,  # Short delay for testing
        enable_adaptive_retry=True
    )

    # Mock operation that fails with rate limit then succeeds
    mock_operation = AsyncMock(side_effect=[
        Exception("429 Too Many Requests"),
        Exception("429 Too Many Requests"),
        Exception("429 Too Many Requests"),
        "success"
    ])

    start_time = time.time()
    result = await retry_policy.execute(mock_operation)
    duration = time.time() - start_time

    # Verify operation was called 4 times and returned expected result
    assert mock_operation.call_count == 4, f"Expected 4 calls (3 retries), got {mock_operation.call_count}"
    assert result == "success", f"Expected 'success', got {result}"

    # Verify metrics
    metrics = retry_policy.get_metrics()
    assert metrics['total_attempts'] == 4, f"Expected 4 attempts, got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 3, f"Expected 3 successful retries, got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 0, f"Expected 0 failed retries, got {metrics['failed_retries']}"

    print(f"✅ Rate limited operation succeeded after 3 retries in {duration:.3f}s")
    print("✅ Extra retry granted for rate limiting")
    print("✅ Metrics correctly recorded")
    print()


async def test_concurrent_retry_operations():
    """Test concurrent retry operations for thread safety."""
    print("🔀 Testing Concurrent Retry Operations")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=2,
        base_delay=0.05,  # Very short delay for testing
        enable_adaptive_retry=True
    )

    # Mock operations with different failure patterns
    async def operation_a():
        await asyncio.sleep(0.01)
        return "success_a"

    async def operation_b():
        await asyncio.sleep(0.01)
        if not hasattr(operation_b, 'call_count'):
            operation_b.call_count = 0
        operation_b.call_count += 1
        if operation_b.call_count < 2:
            raise Exception("Connection timeout")
        return "success_b"

    async def operation_c():
        await asyncio.sleep(0.01)
        raise Exception("404 Not Found")

    # Execute operations concurrently
    start_time = time.time()

    tasks = [
        retry_policy.execute(operation_a),
        retry_policy.execute(operation_b),
        retry_policy.execute(operation_c)
    ]

    results = []
    exceptions = []

    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            exceptions.append(e)

    duration = time.time() - start_time

    # Verify results
    assert len(results) == 2, f"Expected 2 successful operations, got {len(results)}"
    assert len(exceptions) == 1, f"Expected 1 failed operation, got {len(exceptions)}"
    assert "success_a" in results, "Expected operation_a to succeed"
    assert "success_b" in results, "Expected operation_b to succeed after retry"
    assert "404" in str(exceptions[0]), "Expected operation_c to fail with permanent error"

    print(f"✅ Concurrent operations completed in {duration:.3f}s")
    print("✅ Thread safety verified")
    print("✅ Mixed success/failure scenarios handled correctly")
    print()


async def test_performance_metrics():
    """Test retry policy performance metrics collection."""
    print("📊 Testing Performance Metrics")
    print("-" * 50)

    retry_policy = RetryPolicy(
        max_retries=2,
        base_delay=0.05,
        enable_adaptive_retry=True
    )

    # Reset metrics
    retry_policy.reset_metrics()

    # Mock operation with mixed success/failure
    mock_operation = AsyncMock(side_effect=[
        Exception("Connection timeout"),
        "success"
    ])

    # Execute operation
    await retry_policy.execute(mock_operation)

    # Get metrics
    metrics = retry_policy.get_metrics()

    # Verify metrics structure and values
    required_metrics = [
        'total_attempts', 'successful_retries', 'failed_retries',
        'circuit_breaker_trips', 'total_delay_time', 'success_rate',
        'average_delay', 'config'
    ]

    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    assert metrics['total_attempts'] == 2, f"Expected 2 attempts, got {metrics['total_attempts']}"
    assert metrics['successful_retries'] == 1, f"Expected 1 successful retry, got {metrics['successful_retries']}"
    assert metrics['failed_retries'] == 0, f"Expected 0 failed retries, got {metrics['failed_retries']}"
    assert metrics['success_rate'] == 100.0, f"Expected 100% success rate, got {metrics['success_rate']}"

    # Verify config metrics
    config = metrics['config']
    assert config['max_retries'] == 2, f"Expected max_retries=2, got {config['max_retries']}"
    assert config['jitter_factor'] == 0.1, f"Expected jitter_factor=0.1, got {config['jitter_factor']}"
    assert config['adaptive_retry_enabled'] is True, f"Expected adaptive retry enabled, got {config['adaptive_retry_enabled']}"

    print("✅ All required metrics present")
    print("✅ Metric values correct")
    print("✅ Configuration metrics properly recorded")
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
        test_retry_policy_exhaustion,
        test_retry_policy_rate_limiting,
        test_concurrent_retry_operations,
        test_performance_metrics,
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
        print("✅ Thread safety verified for concurrent operations")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Review retry policy implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)