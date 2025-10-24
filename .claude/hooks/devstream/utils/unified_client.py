#!/usr/bin/env python3
"""
DevStream Unified Client - Context7-inspired Adapter Pattern

Provides unified interface for both MCP server and Direct Database clients.
Implements graceful degradation and fallback mechanisms.
Maintains 100% backward compatibility during migration.

Context7 Patterns:
- Adapter Pattern for backend abstraction
- Strategy Pattern for backend selection
- Circuit Breaker Pattern for fault tolerance
- Graceful degradation with fallbacks
"""

import asyncio
import logging
import random
import time
from typing import Optional, Dict, Any, List, Union, Callable
from contextlib import asynccontextmanager
from enum import Enum
from dataclasses import dataclass
import hashlib

# Context7-compliant imports with graceful fallback
try:
    # Try relative import first (when run as module)
    from .direct_client import DevStreamDirectClient, DatabaseException
    from .mcp_client import DevStreamMCPClient
except ImportError:
    try:
        # Fallback to absolute import (when run as script)
        from direct_client import DevStreamDirectClient, DatabaseException
        from mcp_client import DevStreamMCPClient
    except ImportError as e:
        # Final fallback - define dummy classes for graceful degradation
        import logging

        logging.warning(f"DevStream: direct_client unavailable, using fallback: {e}")

        class DatabaseException(Exception):
            """Fallback DatabaseException when direct_client unavailable."""
            pass

        class DevStreamDirectClient:
            """Fallback direct client when real client unavailable."""

            def __init__(self, db_path=None):
                self.disabled = True
                self.db_path = db_path

            async def store_memory(self, *args, **kwargs):
                return {"success": False, "error": "direct_client unavailable"}

            async def search_memory(self, *args, **kwargs):
                return {"results": [], "success": False, "error": "direct_client unavailable"}

            async def health_check(self):
                return False

            async def trigger_checkpoint(self, *args, **kwargs):
                return {"success": False, "error": "direct_client unavailable"}

# Context7-compliant feature flags import
try:
    # Try relative import first (when run as module)
    from ..config.feature_flags import should_use_direct_client
except ImportError:
    try:
        # Fallback to absolute import (when run as script)
        from config.feature_flags import should_use_direct_client
    except ImportError:
        # Fallback if feature flags not available
        def should_use_direct_client(hook_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
            """Fallback: always use direct client if feature flags not available"""
            return True

# Context7-compliant robustness patterns import
try:
    # Try relative import first (when run as module)
    from .robustness_patterns import (
        CircuitBreaker,
        RetryPolicy,
        CircuitBreakerState as CircuitState,
        RobustnessConfig
    )
except ImportError:
    try:
        # Fallback to absolute import (when run as script)
        from robustness_patterns import (
            CircuitBreaker,
            RetryPolicy,
            CircuitBreakerState as CircuitState,
            RobustnessConfig
        )
    except ImportError:
        # Fallback implementation if robustness_patterns not available
        pass  # Already defined above in the fallback block
except ImportError:
    # Fallback implementation if robustness_patterns not available
    import asyncio
    from enum import Enum
    import time
    import threading

    class CircuitState(Enum):
        """Circuit breaker states - matches robustness_patterns API"""
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    @dataclass
    class RobustnessConfig:
        """Configuration for robustness patterns - simplified fallback."""
        # Circuit breaker configuration
        circuit_breaker_failure_threshold: int = 3
        circuit_breaker_timeout_seconds: int = 30
        circuit_breaker_success_threshold: int = 3

        # Timeout configuration
        default_timeout_seconds: int = 30
        max_retries: int = 3
        retry_backoff_factor: float = 2.0
        retry_base_delay: float = 1.0

    class CircuitBreaker:
        """
        Circuit breaker fallback implementation with dual signature support.

        Context7 Best Practice: Support both config object and individual parameters
        for backward compatibility and graceful degradation.
        CRITICAL FIX: Added atomic operations with threading.Lock for thread safety.
        """

        def __init__(self, *args, **kwargs):
            """
            Initialize circuit breaker with flexible signature.

            Supports both patterns:
            - CircuitBreaker(failure_threshold=3, timeout_seconds=30, expected_exception=Exception)
            - CircuitBreaker(config) where config has required attributes
            """
            # Handle config object pattern (robustness_patterns style)
            if len(args) == 1 and hasattr(args[0], 'circuit_breaker_failure_threshold'):
                config = args[0]
                self.failure_threshold = getattr(config, 'circuit_breaker_failure_threshold', 3)
                self.timeout_seconds = getattr(config, 'circuit_breaker_timeout_seconds', 30)
                self.expected_exception = Exception  # Default for config pattern
            # Handle individual parameters pattern (fallback style)
            else:
                self.failure_threshold = kwargs.get('failure_threshold', 3)
                self.timeout_seconds = kwargs.get('timeout_seconds', 30)
                self.expected_exception = kwargs.get('expected_exception', Exception)

            self.failure_count = 0
            self._state = CircuitState.CLOSED
            self.last_failure_time = None

            # CRITICAL FIX: Add lock for atomic state modifications
            self._lock = threading.Lock()

            # Expose CircuitState as class attribute for API compatibility
            self.CircuitState = CircuitState

        @property
        def state(self):
            """Get current circuit breaker state - thread-safe."""
            with self._lock:
                return self._state

        async def call(self, operation, *args, **kwargs):
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
            # CRITICAL FIX: Check state and timeout atomically
            with self._lock:
                should_proceed = False
                if self._state == CircuitState.OPEN:
                    if (self.last_failure_time and
                        time.time() - self.last_failure_time > self.timeout_seconds):
                        # Transition to HALF_OPEN atomically
                        self._state = CircuitState.HALF_OPEN
                        should_proceed = True
                    else:
                        raise ConnectionError("Circuit breaker is OPEN - service unavailable")
                else:
                    should_proceed = True

            try:
                result = await operation(*args, **kwargs)

                # CRITICAL FIX: Success handling with atomic state updates
                with self._lock:
                    if self._state == CircuitState.HALF_OPEN:
                        self._state = CircuitState.CLOSED
                        self.failure_count = 0
                    elif self._state == CircuitState.CLOSED:
                        # Reset failure count on success in closed state
                        self.failure_count = max(0, self.failure_count - 1)

                return result

            except self.expected_exception as e:
                # CRITICAL FIX: Failure handling with atomic state updates
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    # Check if we should open the circuit
                    if self.failure_count >= self.failure_threshold:
                        self._state = CircuitState.OPEN
                    elif self._state == CircuitState.HALF_OPEN:
                        # Any failure in half-open returns to open
                        self._state = CircuitState.OPEN

                raise

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
            Execute operation with Context7-compliant retry logic and proper exception chaining.

            Enhanced Features:
            - Jitter-backed exponential backoff
            - Adaptive retry strategies
            - Circuit breaker integration
            - Comprehensive monitoring
            - LOG-005: Proper exception chaining with py-buzz patterns

            Args:
                operation: Async callable to execute
                *args: Operation arguments
                **kwargs: Operation keyword arguments

            Returns:
                Result of operation

            Raises:
                Exception: Enhanced exception with full context and retry history
            """
            # LOG-005: Track retry history for proper exception chaining
            retry_history = []
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
                    retry_history.append({
                        'attempt': attempt + 1,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e),
                        'classification': self._classify_error(e),
                        'timestamp': time.time()
                    })

                    # Check if we should retry this error
                    if not self._should_retry(e, attempt):
                        self._metrics['failed_retries'] += 1
                        logging.debug(
                            f"RetryPolicy: Not retrying {type(e).__name__}: {e} "
                            f"(classification: {self._classify_error(e)})"
                        )
                        # LOG-005: Raise with enhanced context
                        raise self._create_enhanced_exception(e, retry_history, start_time)

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

            # All retries failed - LOG-005: Create enhanced exception with full context
            self._metrics['failed_retries'] += 1
            total_time = time.time() - start_time
            logging.warning(
                f"RetryPolicy: All {self.max_retries + 1} attempts failed in {total_time:.2f}s. "
                f"Final error: {type(last_exception).__name__}: {last_exception}"
            )

            raise self._create_enhanced_exception(last_exception, retry_history, start_time)

        def _create_enhanced_exception(self, original_exception: Exception, retry_history: list, start_time: float) -> Exception:
            """
            Create enhanced exception with proper context using Context7 py-buzz patterns.

            Args:
                original_exception: The original exception that caused the failure
                retry_history: List of retry attempts with details
                start_time: When the operation started

            Returns:
                Enhanced exception with full context and retry history
            """
            # LOG-005: Apply Context7 py-buzz pattern for exception chaining
            total_time = time.time() - start_time

            # Create detailed error message with retry context
            error_details = (
                f"Operation failed after {self.max_retries + 1} attempts in {total_time:.2f}s. "
                f"Original error: {type(original_exception).__name__}: {original_exception}. "
                f"Retry history: {self._format_retry_history(retry_history)}. "
                f"Policy: max_retries={self.max_retries}, jitter_factor={self.jitter_factor}, "
                f"adaptive_retry={self.enable_adaptive_retry}"
            )

            # Create enhanced exception that preserves the original
            enhanced_exception = type(original_exception)(
                error_details
            )

            # Set proper exception chaining (Context7 best practice)
            enhanced_exception.__cause__ = original_exception
            enhanced_exception.__context__ = getattr(original_exception, '__context__', None)

            # Add retry history as attribute for debugging
            enhanced_exception.retry_history = retry_history
            enhanced_exception.total_time = total_time
            enhanced_exception.retry_policy_config = {
                'max_retries': self.max_retries,
                'jitter_factor': self.jitter_factor,
                'adaptive_retry': self.enable_adaptive_retry,
                'backoff_factor': self.backoff_factor
            }

            return enhanced_exception

        def _format_retry_history(self, retry_history: list) -> str:
            """
            Format retry history for error messages using Context7 patterns.

            Args:
                retry_history: List of retry attempt details

            Returns:
                Formatted string with retry history
            """
            if not retry_history:
                return "no retries"

            history_parts = []
            for attempt in retry_history[-3:]:  # Show last 3 attempts to avoid message bloat
                history_parts.append(
                    f"attempt_{attempt['attempt']}({attempt['classification']})"
                )

            return " → ".join(history_parts)

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


class BackendType(Enum):
    """Supported backend types - MCP DEPRECATED"""
    DIRECT_DB = "direct_db"  # ONLY Direct DB is used now


class UnifiedClient:
    """
    Unified client providing seamless backend switching.

    Context7 Pattern: Adapter + Strategy + Circuit Breaker
    Provides single interface regardless of underlying backend.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize unified client with both backends available.

        Args:
            db_path: Database path for direct client
        """
        self.logger = logging.getLogger('unified_client')
        self.db_path = db_path

        # Initialize direct client
        self._direct_client: Optional[DevStreamDirectClient] = None
        self._mcp_client: Optional[Any] = None

        # Circuit breakers for each backend
        # Create configuration objects for robustness_patterns CircuitBreaker
        direct_config = RobustnessConfig(
            circuit_breaker_failure_threshold=3,
            circuit_breaker_timeout_seconds=30,
            circuit_breaker_success_threshold=3
        )

        mcp_config = RobustnessConfig(
            circuit_breaker_failure_threshold=3,
            circuit_breaker_timeout_seconds=30,
            circuit_breaker_success_threshold=3
        )

        self._direct_circuit = CircuitBreaker(direct_config)
        self._mcp_circuit = CircuitBreaker(mcp_config)

        # Retry policies with Context7-compliant jitter
        self._retry_policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,  # Increased to Context7 best practice
            backoff_factor=2.0,
            jitter=True  # Enable jitter to prevent thundering herd
        )

        # Performance metrics
        self._metrics = {
            'direct_calls': 0,
            'mcp_calls': 0,
            'direct_failures': 0,
            'mcp_failures': 0,
            'fallback_activations': 0
        }

        self.logger.info("Unified client initialized with adaptive backend selection")

    def _get_client(self, hook_name: str, context: Optional[Dict[str, Any]] = None) -> tuple[BackendType, Any]:
        """
        SIMPLIFIED: Always use Direct DB - MCP DEPRECATED.

        Args:
            hook_name: Name of the hook calling the client
            context: Optional context (ignored - Direct DB only)

        Returns:
            Tuple of (DIRECT_DB, direct_client_instance)
        """
        # MCP IS DEPRECATED - ALWAYS USE DIRECT DB
        return BackendType.DIRECT_DB, self._get_direct_client()

    def _get_direct_client(self) -> DevStreamDirectClient:
        """Get or create direct database client."""
        if self._direct_client is None:
            self._direct_client = DevStreamDirectClient(self.db_path)
            self.logger.info("Direct database client initialized")
        return self._direct_client

    def _get_mcp_client(self) -> Any:
        """Get or create MCP client."""
        if self._mcp_client is None:
            self.logger.info("Initializing DevStream MCP facade (direct client)")
            self._mcp_client = DevStreamMCPClient(self.db_path)

        return self._mcp_client

    async def _execute_with_fallback(
        self,
        operation_name: str,
        hook_name: str,
        direct_operation: callable,
        mcp_operation: callable = None,  # IGNORED - MCP DEPRECATED
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        SIMPLIFIED: Execute Direct DB operation only - MCP DEPRECATED.

        Args:
            operation_name: Name of the operation for logging
            hook_name: Name of the hook calling
            direct_operation: Async function for direct DB operation
            mcp_operation: IGNORED - MCP DEPRECATED
            context: Optional context (ignored)

        Returns:
            Operation result

        Raises:
            Exception: If direct DB operation fails
        """
        start_time = time.time()

        # MCP IS DEPRECATED - USE DIRECT DB ONLY
        try:
            result = await self._retry_policy.execute(
                self._direct_circuit.call,
                direct_operation
            )

            # Update metrics
            self._metrics['direct_calls'] += 1

            duration_ms = (time.time() - start_time) * 1000
            self.logger.debug(
                f"{operation_name} completed via Direct DB in {duration_ms:.1f}ms"
            )

            return result

        except Exception as e:
            self._metrics['direct_failures'] += 1
            self.logger.error(f"{operation_name} failed via Direct DB: {e}")
            raise

    async def store_memory(
        self,
        content: str,
        content_type: str,
        keywords: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        hook_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Store memory via Direct DB with automatic embedding generation.

        Args:
            content: Content to store
            content_type: Type of content (code, documentation, context, output, error, decision, learning)
            keywords: Associated keywords
            session_id: Session identifier
            hook_name: Name of the calling hook

        Returns:
            Dictionary with storage result and embedding metadata
        """
        # Direct DB operation with automatic embedding generation
        async def direct_store():
            client = self._get_direct_client()
            result = await client.store_memory(content, content_type, keywords, session_id)

            # Log embedding generation success
            if result.get("embedding_generated", False):
                self.logger.info(
                    f"✅ Embedding generated via {hook_name}: "
                    f"format={result.get('embedding_format')}, "
                    f"dimensions={result.get('embedding_dimension')}"
                )
            else:
                self.logger.warning(
                    f"⚠️ Embedding generation failed via {hook_name}: graceful degradation"
                )

            return result

        return await self._execute_with_fallback(
            operation_name="store_memory",
            hook_name=hook_name,
            direct_operation=direct_store,
            context={"session_id": session_id}
        )

    async def search_memory(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10,
        hook_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Search memory via Direct DB only - MCP DEPRECATED.

        Args:
            query: Search query
            content_type: Filter by content type
            limit: Maximum results
            hook_name: Name of the calling hook

        Returns:
            Dictionary with search results
        """
        # Direct DB operation only - MCP DEPRECATED
        async def direct_search():
            client = self._get_direct_client()
            return await client.search_memory(query, content_type, limit)

        return await self._execute_with_fallback(
            operation_name="search_memory",
            hook_name=hook_name,
            direct_operation=direct_search,
            context={"query": query, "content_type": content_type}
        )

    async def trigger_checkpoint(
        self,
        reason: str = "tool_trigger",
        hook_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Trigger checkpoint via Direct DB only - MCP DEPRECATED.

        Args:
            reason: Checkpoint reason
            hook_name: Name of the calling hook

        Returns:
            Dictionary with checkpoint result
        """
        # Direct DB operation only - MCP DEPRECATED
        async def direct_checkpoint():
            client = self._get_direct_client()
            return await client.trigger_checkpoint(reason)

        return await self._execute_with_fallback(
            operation_name="trigger_checkpoint",
            hook_name=hook_name,
            direct_operation=direct_checkpoint,
            context={"reason": reason}
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Direct DB only - MCP DEPRECATED.

        Returns:
            Dictionary with health status of Direct DB backend
        """
        health_status = {
            "overall": "healthy",
            "backends": {},
            "metrics": self._metrics.copy()
        }

        # Check direct client only - MCP DEPRECATED
        try:
            if self._direct_client:
                direct_healthy = await self._direct_client.health_check()
                health_status["backends"]["direct_db"] = {
                    "status": "healthy" if direct_healthy else "unhealthy",
                    "circuit_breaker": self._direct_circuit.state.value
                }
            else:
                health_status["backends"]["direct_db"] = {
                    "status": "not_initialized",
                    "circuit_breaker": self._direct_circuit.state.value
                }
        except Exception as e:
            health_status["backends"]["direct_db"] = {
                "status": "error",
                "error": str(e),
                "circuit_breaker": self._direct_circuit.state.value
            }
            health_status["overall"] = "degraded"

        return health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and reliability metrics - Direct DB only."""
        return {
            **self._metrics,
            "direct_circuit_state": self._direct_circuit.state.value,
            "total_calls": self._metrics['direct_calls'],
            "total_failures": self._metrics['direct_failures'],
            "success_rate": (
                (self._metrics['direct_calls'] - self._metrics['direct_failures']) /
                max(1, self._metrics['direct_calls'])
            ) * 100
        }


# Global unified client instance (singleton pattern)
_unified_client: Optional[UnifiedClient] = None


def get_unified_client(db_path: Optional[str] = None) -> UnifiedClient:
    """
    Get singleton unified client instance.

    Args:
        db_path: Database path for direct client

    Returns:
        UnifiedClient instance
    """
    global _unified_client
    if _unified_client is None:
        _unified_client = UnifiedClient(db_path)
    return _unified_client


# Convenience functions for backward compatibility
async def store_memory_unified(
    content: str,
    content_type: str,
    keywords: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    hook_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Unified memory storage with automatic backend selection.

    Context7 Pattern: Facade pattern for simplified access
    """
    client = get_unified_client()
    return await client.store_memory(content, content_type, keywords, session_id, hook_name)


async def search_memory_unified(
    query: str,
    content_type: Optional[str] = None,
    limit: int = 10,
    hook_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Unified memory search with automatic backend selection.
    """
    client = get_unified_client()
    return await client.search_memory(query, content_type, limit, hook_name)


async def trigger_checkpoint_unified(
    reason: str = "tool_trigger",
    hook_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Unified checkpoint trigger with automatic backend selection.
    """
    client = get_unified_client()
    return await client.trigger_checkpoint(reason, hook_name)
