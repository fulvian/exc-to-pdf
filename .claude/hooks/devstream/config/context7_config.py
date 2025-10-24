#!/usr/bin/env python3
"""
Context7 Configuration Management

Handles configuration for Context7 Direct Client integration
with feature flags for gradual rollout.
"""

import os
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class Context7Config:
    """Configuration for Context7 Direct Client integration."""
    direct_enabled: Union[bool, str] = False
    cache_size: int = 100
    timeout: int = 30
    circuit_breaker_threshold: int = 3
    metrics_enabled: bool = True
    mcp_fallback: bool = True

    @classmethod
    def from_env(cls) -> 'Context7Config':
        """Load configuration from environment variables."""
        direct_enabled_str = os.getenv("DEVSTREAM_CONTEXT7_DIRECT_ENABLED", "false").lower()

        # Convert string to boolean or keep rollout string
        if direct_enabled_str == "true":
            direct_enabled = True
        elif direct_enabled_str == "false":
            direct_enabled = False
        else:
            direct_enabled = direct_enabled_str  # Keep "rollout" or other values

        return cls(
            direct_enabled=direct_enabled,
            cache_size=int(os.getenv("DEVSTREAM_CONTEXT7_DIRECT_CACHE_SIZE", "100")),
            timeout=int(os.getenv("DEVSTREAM_CONTEXT7_DIRECT_TIMEOUT", "30")),
            circuit_breaker_threshold=int(os.getenv("DEVSTREAM_CONTEXT7_DIRECT_CB_THRESHOLD", "3")),
            metrics_enabled=os.getenv("DEVSTREAM_CONTEXT7_METRICS_ENABLED", "true").lower() == "true",
            mcp_fallback=os.getenv("DEVSTREAM_CONTEXT7_MCP_FALLBACK", "true").lower() == "true"
        )

    def should_use_direct_mode(self, library_name: Optional[str] = None) -> bool:
        """
        Determine if direct mode should be used based on configuration.

        Args:
            library_name: Optional library name for rollout distribution

        Returns:
            True if direct mode should be used
        """
        if isinstance(self.direct_enabled, bool):
            return self.direct_enabled
        elif self.direct_enabled == "rollout":
            # 10% rollout based on hash if library name provided
            if library_name:
                import hashlib
                hash_value = int(hashlib.md5(library_name.encode()).hexdigest(), 16)
                return hash_value % 10 == 0
            return False
        return False

    def get_mode_description(self) -> str:
        """Get human-readable description of current mode."""
        if isinstance(self.direct_enabled, bool):
            return "Direct mode" if self.direct_enabled else "MCP mode"
        elif self.direct_enabled == "rollout":
            return f"Gradual rollout (10% based on library hash)"
        return "Unknown mode"

    def validate(self) -> list[str]:
        """
        Validate configuration values.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.cache_size < 1 or self.cache_size > 10000:
            errors.append(f"Invalid cache_size: {self.cache_size}. Must be between 1 and 10000")

        if self.timeout < 1 or self.timeout > 300:
            errors.append(f"Invalid timeout: {self.timeout}. Must be between 1 and 300 seconds")

        if self.circuit_breaker_threshold < 1 or self.circuit_breaker_threshold > 10:
            errors.append(f"Invalid circuit_breaker_threshold: {self.circuit_breaker_threshold}. Must be between 1 and 10")

        if self.direct_enabled not in [True, False, "rollout", "false", "true"]:
            errors.append(f"Invalid direct_enabled: {self.direct_enabled}. Must be true, false, or rollout")

        return errors

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "direct_enabled": self.direct_enabled,
            "cache_size": self.cache_size,
            "timeout": self.timeout,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "metrics_enabled": self.metrics_enabled,
            "mcp_fallback": self.mcp_fallback,
            "mode_description": self.get_mode_description()
        }


# Global configuration instance
_config: Optional[Context7Config] = None


def get_context7_config() -> Context7Config:
    """
    Get singleton Context7 configuration instance.

    Returns:
        Context7Config instance loaded from environment
    """
    global _config
    if _config is None:
        _config = Context7Config.from_env()

        # Validate configuration
        errors = _config.validate()
        if errors:
            import sys
            print("⚠️  Context7 configuration errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)

    return _config


def reload_context7_config() -> Context7Config:
    """
    Reload Context7 configuration from environment.

    Returns:
        New Context7Config instance
    """
    global _config
    _config = None
    return get_context7_config()


# Test function
def test_context7_config() -> None:
    """Test Context7 configuration functionality."""
    import json

    config = get_context7_config()

    print("🧪 Testing Context7 Configuration...")
    print(f"1. Mode: {config.get_mode_description()}")
    print(f"2. Cache size: {config.cache_size}")
    print(f"3. Timeout: {config.timeout}s")
    print(f"4. Circuit breaker threshold: {config.circuit_breaker_threshold}")
    print(f"5. Metrics enabled: {config.metrics_enabled}")
    print(f"6. MCP fallback: {config.mcp_fallback}")

    # Test mode selection
    test_libraries = ["fastapi", "django", "react", "pytest", "aiohttp"]
    direct_count = sum(1 for lib in test_libraries if config.should_use_direct_mode(lib))
    print(f"7. Direct mode for test libraries: {direct_count}/{len(test_libraries)}")

    # Validation
    errors = config.validate()
    if errors:
        print(f"8. Validation errors: {len(errors)}")
        for error in errors:
            print(f"   - {error}")
    else:
        print("8. Validation: ✅ Passed")

    # Dict export
    config_dict = config.to_dict()
    print(f"9. Configuration dict: {json.dumps(config_dict, indent=2)}")

    print("🎉 Context7 Configuration test completed!")


if __name__ == "__main__":
    test_context7_config()