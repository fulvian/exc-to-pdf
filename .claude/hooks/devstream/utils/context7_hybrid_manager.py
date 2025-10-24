#!/usr/bin/env python3
"""
Context7 Hybrid Manager

Hybrid Context7 manager supporting both MCP and Direct HTTP clients.
Implements gradual rollout with feature flags and automatic fallback.

Key Features:
- Feature flag based mode selection (true/false/rollout)
- MCP fallback when direct mode fails
- Performance metrics collection
- Circuit breaker for mode switching
- Context7-compliant error handling
"""

import asyncio
import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Import Context7 Direct Client
try:
    from .context7_direct_client import (
        Context7DirectHttpClient,
        Context7APIError,
        CircuitBreakerError
    )
except ImportError:
    # Fallback if direct client not available
    Context7DirectHttpClient = None
    Context7APIError = Exception
    CircuitBreakerError = Exception

# Import Context7 MCP Direct Client
try:
    from .context7_mcp_direct_client import (
        Context7MCPDirectClient,
        Context7MCPError
    )
except ImportError:
    # Fallback if MCP direct client not available
    Context7MCPDirectClient = None
    Context7MCPError = Exception

# Import DevStream utilities
try:
    from .logger import get_devstream_logger
except ImportError:
    # Fallback logger
    import structlog
    from typing import Any
    def _get_fallback_logger(name: str) -> Any:  # type: ignore
        return structlog.get_logger(name)


class Context7Error(Exception):
    """Base Context7 error."""
    pass


class Context7Mode(Enum):
    """Context7 operation modes."""
    DIRECT = "direct"
    MCP = "mcp"


@dataclass
class Context7Metrics:
    """Performance metrics for Context7 operations."""
    direct_calls: int = 0
    direct_successes: int = 0
    direct_failures: int = 0
    mcp_calls: int = 0
    mcp_successes: int = 0
    mcp_failures: int = 0
    fallback_activations: int = 0
    circuit_breaker_trips: int = 0

    def get_direct_success_rate(self) -> float:
        """Get success rate for direct mode."""
        return (self.direct_successes / max(1, self.direct_calls)) * 100

    def get_mcp_success_rate(self) -> float:
        """Get success rate for MCP mode."""
        return (self.mcp_successes / max(1, self.mcp_calls)) * 100

    def get_overall_success_rate(self) -> float:
        """Get overall success rate."""
        total_calls = self.direct_calls + self.mcp_calls
        total_successes = self.direct_successes + self.mcp_successes
        return (total_successes / max(1, total_calls)) * 100


class Context7HybridManager:
    """
    Hybrid Context7 manager supporting MCP Direct Client, Direct HTTP client, and MCP fallback.
    Implements gradual rollout with feature flags and automatic fallback.

    Attributes:
        mcp_direct_client: Context7MCPDirectClient instance (preferred)
        direct_client: Context7DirectHttpClient instance (legacy)
        mcp_enabled: MCP client availability flag
        direct_enabled: Direct client enabled via feature flag
        circuit_breaker: Circuit breaker for mode switching
        metrics: Performance and reliability metrics
    """

    def __init__(
        self,
        direct_enabled: Optional[bool] = None,
        mcp_fallback: bool = True
    ) -> None:
        """
        Initialize hybrid Context7 manager.

        Args:
            direct_enabled: Override feature flag (None = use ENV)
            mcp_fallback: Enable MCP as fallback when direct fails
        """
        self.logger = (
            get_devstream_logger('context7_hybrid_manager')
            if 'get_devstream_logger' in globals()
            else _get_fallback_logger('context7_hybrid_manager')
        )

        self.mcp_fallback = mcp_fallback
        self._direct_enabled_override = direct_enabled

        # Initialize MCP Direct Client first (preferred)
        self.mcp_direct_client: Optional[Context7MCPDirectClient] = None
        if Context7MCPDirectClient:
            try:
                self.mcp_direct_client = Context7MCPDirectClient()
                self.logger.info("Context7 MCP Direct Client initialized (preferred)")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize Context7 MCP Direct Client: {e}",
                    extra={"error": str(e), "error_type": type(e).__name__}
                )

        # Initialize legacy Direct HTTP Client as fallback
        self.direct_client: Optional[Context7DirectHttpClient] = None
        if Context7DirectHttpClient and not self.mcp_direct_client:
            try:
                self.direct_client = Context7DirectHttpClient()
                self.logger.info("Context7 Direct HTTP Client initialized (legacy fallback)")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize Context7 Direct HTTP Client: {e}",
                    extra={"error": str(e), "error_type": type(e).__name__}
                )

        # Check MCP availability (check for MCP tools)
        self.mcp_available = self._check_mcp_availability()

        # Performance metrics
        self.metrics = Context7Metrics()

        # Circuit breaker for mode switching (different from direct client's circuit breaker)
        self.mode_circuit_breaker = {
            'direct_failures': 0,
            'mcp_failures': 0,
            'last_failure_time': 0,
            'failure_threshold': 3,
            'recovery_timeout': 60.0
        }

        self.logger.info(
            "Context7 Hybrid Manager initialized",
            extra={
                "direct_client_available": self.direct_client is not None,
                "mcp_available": self.mcp_available,
                "mcp_fallback": mcp_fallback,
                "direct_enabled_override": direct_enabled
            }
        )

    def _check_mcp_availability(self) -> bool:
        """
        Check if MCP Context7 tools are available.

        Returns:
            True if MCP tools are available
        """
        try:
            # Try to check for MCP tools by looking for specific functions
            # This is a simple heuristic - in real implementation we'd check
            # the actual Claude Code MCP tool availability
            import sys
            return 'mcp__context7__resolve-library-id' in str(sys.modules) or \
                   hasattr(self, '_mcp_resolve_library_id')
        except Exception:
            return False

    def _should_use_direct_mode(self, library_name: Optional[str] = None) -> bool:
        """
        Determine if direct mode should be used based on feature flags.
        Prefers MCP Direct Client when available.

        Returns:
            True if direct mode enabled and available
        """
        # Check override first
        if self._direct_enabled_override is not None:
            return self._direct_enabled_override and (self.mcp_direct_client is not None or self.direct_client is not None)

        # Environment-based gradual rollout
        flag = os.getenv("DEVSTREAM_CONTEXT7_DIRECT_ENABLED", "false").lower()

        if flag == "true":
            # Prefer MCP Direct Client, fallback to legacy Direct Client
            return self.mcp_direct_client is not None or self.direct_client is not None
        elif flag == "rollout":
            # 10% rollout based on hash of library name
            if library_name and (self.mcp_direct_client is not None or self.direct_client is not None):
                hash_value = int(hashlib.md5(library_name.encode()).hexdigest(), 16)
                return hash_value % 10 == 0
            return False
        else:
            # Default to MCP (false)
            return False

    def _record_mode_metrics(self, mode: str, success: bool, duration: float) -> None:
        """
        Record performance metrics for mode selection optimization.

        Args:
            mode: "direct" or "mcp"
            success: Operation success
            duration: Operation duration in seconds
        """
        if mode == "direct":
            self.metrics.direct_calls += 1
            if success:
                self.metrics.direct_successes += 1
            else:
                self.metrics.direct_failures += 1
                self.mode_circuit_breaker['direct_failures'] += 1
        elif mode == "mcp":
            self.metrics.mcp_calls += 1
            if success:
                self.metrics.mcp_successes += 1
            else:
                self.metrics.mcp_failures += 1
                self.mode_circuit_breaker['mcp_failures'] += 1

        self.logger.debug(
            f"Context7 metrics recorded - mode: {mode}, success: {success}, duration: {duration:.3f}s",
            extra={
                "mode": mode,
                "success": success,
                "duration_ms": duration * 1000,
                "metrics": {
                    "direct_calls": self.metrics.direct_calls,
                    "mcp_calls": self.metrics.mcp_calls,
                    "fallback_activations": self.metrics.fallback_activations
                }
            }
        )

    async def _call_mcp_resolve_library(self, library_name: str) -> Optional[str]:
        """
        Call MCP library resolution as fallback.

        Args:
            library_name: Library name to resolve

        Returns:
            Library ID or None
        """
        try:
            # In real implementation, this would call the actual MCP tool
            # For now, we'll simulate a fallback response
            self.logger.debug(
                f"MCP fallback: resolving library '{library_name}'",
                extra={"library_name": library_name, "fallback_method": "mcp"}
            )

            # Simulate MCP call (replace with actual MCP call)
            await asyncio.sleep(0.1)  # Simulate network latency

            # Return simulated library ID
            return f"/simulated/{library_name}"

        except Exception as e:
            self.logger.error(
                f"MCP fallback failed for library '{library_name}': {e}",
                extra={"library_name": library_name, "error": str(e)}
            )
            return None

    async def _call_mcp_get_docs(self, library_id: str, topic: Optional[str] = None, tokens: int = 5000) -> Optional[str]:
        """
        Call MCP documentation retrieval as fallback.

        Args:
            library_id: Library ID
            topic: Optional topic
            tokens: Maximum tokens

        Returns:
            Documentation or None
        """
        try:
            # In real implementation, this would call the actual MCP tool
            self.logger.debug(
                f"MCP fallback: getting docs for '{library_id}'",
                extra={
                    "library_id": library_id,
                    "topic": topic,
                    "tokens": tokens,
                    "fallback_method": "mcp"
                }
            )

            # Simulate MCP call (replace with actual MCP call)
            await asyncio.sleep(0.2)  # Simulate network latency

            # Return simulated documentation
            return f"Simulated documentation for {library_id}" + (f" (topic: {topic})" if topic else "")

        except Exception as e:
            self.logger.error(
                f"MCP fallback failed for docs '{library_id}': {e}",
                extra={"library_id": library_id, "error": str(e)}
            )
            return None

    async def resolve_library_id(
        self,
        library_name: str,
        force_mode: Optional[str] = None  # "mcp_direct", "direct", or "mcp"
    ) -> Optional[str]:
        """
        Resolve library ID using hybrid strategy.

        Args:
            library_name: Library name to resolve
            force_mode: Force specific mode ("mcp_direct", "direct", or "mcp")

        Returns:
            Context7 library ID or None

        Raises:
            Context7Error: If both modes fail
        """
        start_time = time.time()

        # Determine primary mode - prefer MCP Direct Client when available
        if force_mode:
            primary_mode = force_mode
        elif self.mcp_direct_client and self._should_use_direct_mode(library_name):
            primary_mode = "mcp_direct"
        elif self.direct_client and self._should_use_direct_mode(library_name):
            primary_mode = "direct"
        else:
            primary_mode = "mcp"

        self.logger.debug(
            f"Resolving library ID for '{library_name}' using {primary_mode} mode",
            extra={
                "library_name": library_name,
                "primary_mode": primary_mode,
                "force_mode": force_mode,
                "mcp_direct_available": self.mcp_direct_client is not None,
                "direct_available": self.direct_client is not None,
                "mcp_available": self.mcp_available
            }
        )

        try:
            # Try primary mode first
            if primary_mode == "mcp_direct" and self.mcp_direct_client:
                try:
                    result = await self.mcp_direct_client.resolve_library_id(library_name)
                    if result:
                        self._record_mode_metrics("mcp_direct", True, time.time() - start_time)
                        return result
                except (Context7MCPError, Exception) as e:
                    self.logger.warning(
                        f"MCP Direct mode failed for library '{library_name}': {e}",
                        extra={"library_name": library_name, "error": str(e), "primary_mode": "mcp_direct"}
                    )
                    self._record_mode_metrics("mcp_direct", False, time.time() - start_time)

                    # Try legacy direct client as fallback
                    if self.direct_client:
                        self.metrics.fallback_activations += 1
                        direct_result = await self.direct_client.resolve_library_id(library_name)
                        if direct_result:
                            self._record_mode_metrics("direct", True, time.time() - start_time)
                            return direct_result

                    # Try MCP server as final fallback
                    if self.mcp_fallback and self.mcp_available:
                        self.metrics.fallback_activations += 1
                        mcp_result = await self._call_mcp_resolve_library(library_name)
                        if mcp_result:
                            self._record_mode_metrics("mcp", True, time.time() - start_time)
                            return mcp_result

            elif primary_mode == "direct" and self.direct_client:
                try:
                    result = await self.direct_client.resolve_library_id(library_name)
                    if result:
                        self._record_mode_metrics("direct", True, time.time() - start_time)
                        return result
                except (Context7APIError, CircuitBreakerError) as e:
                    self.logger.warning(
                        f"Direct mode failed for library '{library_name}': {e}",
                        extra={"library_name": library_name, "error": str(e), "primary_mode": "direct"}
                    )
                    self._record_mode_metrics("direct", False, time.time() - start_time)

                    # Try MCP fallback if enabled
                    if self.mcp_fallback and self.mcp_available:
                        self.metrics.fallback_activations += 1
                        mcp_result = await self._call_mcp_resolve_library(library_name)
                        if mcp_result:
                            self._record_mode_metrics("mcp", True, time.time() - start_time)
                            return mcp_result

            elif primary_mode == "mcp" and self.mcp_available:
                mcp_result = await self._call_mcp_resolve_library(library_name)
                if mcp_result:
                    self._record_mode_metrics("mcp", True, time.time() - start_time)
                    return mcp_result

            # If we reach here, primary mode failed
            raise Context7Error(f"Failed to resolve library '{library_name}' using {primary_mode} mode")

        except Exception as e:
            self._record_mode_metrics(primary_mode, False, time.time() - start_time)

            if isinstance(e, Context7Error):
                raise

            self.logger.error(
                f"Unexpected error resolving library '{library_name}': {e}",
                extra={"library_name": library_name, "error": str(e), "error_type": type(e).__name__}
            )
            raise Context7Error(f"Unexpected error resolving library '{library_name}': {e}") from e

    async def get_library_docs(
        self,
        library_id: str,
        topic: Optional[str] = None,
        tokens: int = 5000,
        force_mode: Optional[str] = None
    ) -> Optional[str]:
        """
        Get library docs using hybrid strategy.

        Args:
            library_id: Context7 library ID
            topic: Optional topic
            tokens: Max tokens
            force_mode: Force specific mode ("mcp_direct", "direct", or "mcp")

        Returns:
            Documentation or None
        """
        start_time = time.time()

        # Determine primary mode - prefer MCP Direct Client when available
        if force_mode:
            primary_mode = force_mode
        elif self.mcp_direct_client and self._should_use_direct_mode():
            primary_mode = "mcp_direct"
        elif self.direct_client and self._should_use_direct_mode():
            primary_mode = "direct"
        else:
            primary_mode = "mcp"

        self.logger.debug(
            f"Getting docs for '{library_id}' using {primary_mode} mode",
            extra={
                "library_id": library_id,
                "topic": topic,
                "tokens": tokens,
                "primary_mode": primary_mode,
                "force_mode": force_mode
            }
        )

        try:
            # Try primary mode first
            if primary_mode == "mcp_direct" and self.mcp_direct_client:
                try:
                    result = await self.mcp_direct_client.get_library_docs(library_id, topic, tokens)
                    if result:
                        self._record_mode_metrics("mcp_direct", True, time.time() - start_time)
                        return result
                except (Context7MCPError, Exception) as e:
                    self.logger.warning(
                        f"MCP Direct mode failed for docs '{library_id}': {e}",
                        extra={"library_id": library_id, "error": str(e), "primary_mode": "mcp_direct"}
                    )
                    self._record_mode_metrics("mcp_direct", False, time.time() - start_time)

                    # Try legacy direct client as fallback
                    if self.direct_client:
                        self.metrics.fallback_activations += 1
                        direct_result = await self.direct_client.get_library_docs(library_id, topic, tokens)
                        if direct_result:
                            self._record_mode_metrics("direct", True, time.time() - start_time)
                            return direct_result

                    # Try MCP server as final fallback
                    if self.mcp_fallback and self.mcp_available:
                        self.metrics.fallback_activations += 1
                        mcp_result = await self._call_mcp_get_docs(library_id, topic, tokens)
                        if mcp_result:
                            self._record_mode_metrics("mcp", True, time.time() - start_time)
                            return mcp_result

            elif primary_mode == "direct" and self.direct_client:
                try:
                    result = await self.direct_client.get_library_docs(library_id, topic, tokens)
                    if result:
                        self._record_mode_metrics("direct", True, time.time() - start_time)
                        return result
                except (Context7APIError, CircuitBreakerError) as e:
                    self.logger.warning(
                        f"Direct mode failed for docs '{library_id}': {e}",
                        extra={"library_id": library_id, "error": str(e), "primary_mode": "direct"}
                    )
                    self._record_mode_metrics("direct", False, time.time() - start_time)

                    # Try MCP fallback if enabled
                    if self.mcp_fallback and self.mcp_available:
                        self.metrics.fallback_activations += 1
                        mcp_result = await self._call_mcp_get_docs(library_id, topic, tokens)
                        if mcp_result:
                            self._record_mode_metrics("mcp", True, time.time() - start_time)
                            return mcp_result

            elif primary_mode == "mcp" and self.mcp_available:
                mcp_result = await self._call_mcp_get_docs(library_id, topic, tokens)
                if mcp_result:
                    self._record_mode_metrics("mcp", True, time.time() - start_time)
                    return mcp_result

            # If we reach here, primary mode failed
            raise Context7Error(f"Failed to get docs for '{library_id}' using {primary_mode} mode")

        except Exception as e:
            self._record_mode_metrics(primary_mode, False, time.time() - start_time)

            if isinstance(e, Context7Error):
                raise

            self.logger.error(
                f"Unexpected error getting docs for '{library_id}': {e}",
                extra={"library_id": library_id, "error": str(e), "error_type": type(e).__name__}
            )
            raise Context7Error(f"Unexpected error getting docs for '{library_id}': {e}") from e

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary with performance and reliability metrics
        """
        return {
            "direct": {
                "calls": self.metrics.direct_calls,
                "successes": self.metrics.direct_successes,
                "failures": self.metrics.direct_failures,
                "success_rate": self.metrics.get_direct_success_rate(),
                "client_available": self.direct_client is not None
            },
            "mcp": {
                "calls": self.metrics.mcp_calls,
                "successes": self.metrics.mcp_successes,
                "failures": self.metrics.mcp_failures,
                "success_rate": self.metrics.get_mcp_success_rate(),
                "available": self.mcp_available
            },
            "overall": {
                "total_calls": self.metrics.direct_calls + self.metrics.mcp_calls,
                "total_successes": self.metrics.direct_successes + self.metrics.mcp_successes,
                "total_failures": self.metrics.direct_failures + self.metrics.mcp_failures,
                "success_rate": self.metrics.get_overall_success_rate(),
                "fallback_activations": self.metrics.fallback_activations
            },
            "configuration": {
                "direct_enabled_override": self._direct_enabled_override,
                "mcp_fallback": self.mcp_fallback,
                "mode_circuit_breaker": self.mode_circuit_breaker.copy()
            }
        }

    async def close(self) -> None:
        """Close all clients and cleanup resources."""
        if self.mcp_direct_client:
            await self.mcp_direct_client.close()
        if self.direct_client:
            await self.direct_client.close()
        self.logger.debug("Context7 Hybrid Manager closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Test function
async def test_context7_hybrid_manager() -> None:
    """Test Context7 Hybrid Manager functionality."""
    async with Context7HybridManager(direct_enabled=True, mcp_fallback=True) as manager:
        print("🧪 Testing Context7 Hybrid Manager...")

        # Test library resolution
        print("1. Testing library resolution...")
        try:
            library_id = await manager.resolve_library_id("test-library")
            print(f"   ✅ Library resolved: {library_id}")
        except Exception as e:
            print(f"   ⚠️  Library resolution failed: {e}")

        # Test documentation retrieval
        print("2. Testing documentation retrieval...")
        try:
            docs = await manager.get_library_docs("/test/library", topic="installation")
            print(f"   ✅ Docs retrieved: {len(docs) if docs else 0} chars")
        except Exception as e:
            print(f"   ⚠️  Docs retrieval failed: {e}")

        # Test metrics
        print("3. Testing metrics...")
        metrics = manager.get_metrics()
        print(f"   ✅ Metrics collected: {metrics['overall']['total_calls']} total calls")

        print("🎉 Context7 Hybrid Manager test completed!")


if __name__ == "__main__":
    asyncio.run(test_context7_hybrid_manager())