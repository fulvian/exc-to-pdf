#!/usr/bin/env python3
"""
DevStream Feature Flag System

Provides feature flag management for gradual migration from MCP server
to direct database connections. Supports environment-based configuration
and runtime flag evaluation.

Key Features:
- Environment-based flag configuration
- Runtime flag evaluation with caching
- Gradual migration support
- Performance monitoring
- Rollback capabilities
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import threading
import hashlib


class FeatureFlagType(Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    ALLOW_LIST = "allow_list"
    BLOCK_LIST = "block_list"


class FeatureFlag:
    """
    Individual feature flag with evaluation logic.
    """

    def __init__(
        self,
        key: str,
        flag_type: FeatureFlagType,
        default_value: Any,
        description: str = ""
    ):
        """
        Initialize feature flag.

        Args:
            key: Flag key (snake_case)
            flag_type: Type of flag
            default_value: Default value when not configured
            description: Human-readable description
        """
        self.key = key
        self.flag_type = flag_type
        self.default_value = default_value
        self.description = description
        self._cached_value: Optional[Any] = None
        self._cache_expiry: Optional[datetime] = None
        self._lock = threading.Lock()

    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate flag with given context.

        Args:
            context: Optional evaluation context (session_id, user_id, etc.)

        Returns:
            Flag value based on configuration
        """
        # Check cache first
        with self._lock:
            if (self._cached_value is not None and
                self._cache_expiry and
                datetime.now() < self._cache_expiry):
                return self._cached_value

        # Get value from environment or use default
        raw_value = os.getenv(f"DEVSTREAM_FEATURE_{self.key.upper()}", "")

        if not raw_value:
            result = self.default_value
        else:
            # Parse based on flag type
            result = self._parse_value(raw_value, context or {})

        # Cache result for 5 minutes
        with self._lock:
            self._cached_value = result
            self._cache_expiry = datetime.now() + timedelta(minutes=5)

        return result

    def _parse_value(self, raw_value: str, context: Dict[str, Any]) -> Any:
        """
        Parse raw value based on flag type.

        Args:
            raw_value: Raw value from environment
            context: Evaluation context

        Returns:
            Parsed value
        """
        if self.flag_type == FeatureFlagType.BOOLEAN:
            return raw_value.lower() in ('true', '1', 'yes', 'on', 'enabled')

        elif self.flag_type == FeatureFlagType.PERCENTAGE:
            try:
                percentage = float(raw_value)
                if percentage < 0 or percentage > 100:
                    return False

                # Use session_id or user_id for consistent hashing
                identifier = context.get('session_id') or context.get('user_id') or 'default'
                hash_value = int(hashlib.sha256(identifier.encode()).hexdigest(), 16)
                return (hash_value % 100) < percentage

            except ValueError:
                return False

        elif self.flag_type == FeatureFlagType.ALLOW_LIST:
            try:
                allowed_items = set(json.loads(raw_value))
                identifier = context.get('session_id') or context.get('user_id')
                return identifier in allowed_items
            except (json.JSONDecodeError, TypeError):
                return False

        elif self.flag_type == FeatureFlagType.BLOCK_LIST:
            try:
                blocked_items = set(json.loads(raw_value))
                identifier = context.get('session_id') or context.get('user_id')
                return identifier not in blocked_items
            except (json.JSONDecodeError, TypeError):
                return True

        return self.default_value

    def clear_cache(self) -> None:
        """Clear cached value."""
        with self._lock:
            self._cached_value = None
            self._cache_expiry = None


class FeatureFlagManager:
    """
    Central feature flag manager with caching and monitoring.

    Provides a unified interface for evaluating all feature flags
    with performance monitoring and debugging capabilities.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize feature flag manager.

        Args:
            config_file: Optional path to JSON configuration file
        """
        self.logger = logging.getLogger(f"feature_flag_manager_{id(self)}")
        self.config_file = config_file
        self._flags: Dict[str, FeatureFlag] = {}
        self._evaluation_stats = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'flag_evaluations': {},
            'last_reload': None
        }
        self._lock = threading.Lock()

        # Initialize default flags
        self._initialize_default_flags()

        # Load additional configuration if provided
        if config_file and Path(config_file).exists():
            self.load_config(config_file)

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags for MCP to direct DB migration."""
        # Main migration flag
        self.register_flag(
            FeatureFlag(
                key="direct_db_enabled",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Enable direct database connections instead of MCP server"
            )
        )

        # Gradual rollout flags
        self.register_flag(
            FeatureFlag(
                key="direct_db_post_tool_use",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Enable direct DB for PostToolUse hook"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="direct_db_pre_tool_use",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Enable direct DB for PreToolUse hook"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="direct_db_context_enhancer",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Enable direct DB for User Query Context Enhancer"
            )
        )

        # Performance and reliability flags
        self.register_flag(
            FeatureFlag(
                key="enable_robustness_patterns",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable Context7-inspired robustness patterns"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="enable_connection_pooling",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable connection pooling for direct DB"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="enable_circuit_breaker",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable circuit breaker pattern"
            )
        )

        # Monitoring and debugging flags
        self.register_flag(
            FeatureFlag(
                key="enable_performance_monitoring",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Enable detailed performance monitoring"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="enable_migration_logging",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable detailed migration logging"
            )
        )

        # Testing flags
        self.register_flag(
            FeatureFlag(
                key="force_mcp_fallback",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Force fallback to MCP server (testing only)"
            )
        )

        self.register_flag(
            FeatureFlag(
                key="simulate_mcp_failures",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=False,
                description="Simulate MCP failures for testing resilience"
            )
        )

    def register_flag(self, flag: FeatureFlag) -> None:
        """
        Register a new feature flag.

        Args:
            flag: Feature flag instance
        """
        with self._lock:
            self._flags[flag.key] = flag
            self._evaluation_stats['flag_evaluations'][flag.key] = {
                'evaluations': 0,
                'last_evaluation': None
            }

    def evaluate_flag(self, key: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate a specific feature flag.

        Args:
            key: Flag key
            context: Optional evaluation context

        Returns:
            Flag value or default if flag not found
        """
        with self._lock:
            self._evaluation_stats['total_evaluations'] += 1

        if key not in self._flags:
            self.logger.warning(f"Unknown feature flag: {key}")
            return None

        flag = self._flags[key]

        # Check if value is cached
        is_cached = (flag._cached_value is not None and
                    flag._cache_expiry and
                    datetime.now() < flag._cache_expiry)

        if is_cached:
            with self._lock:
                self._evaluation_stats['cache_hits'] += 1

        # Evaluate flag
        start_time = datetime.now()
        value = flag.evaluate(context)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update statistics
        with self._lock:
            self._evaluation_stats['flag_evaluations'][key]['evaluations'] += 1
            self._evaluation_stats['flag_evaluations'][key]['last_evaluation'] = {
                'value': value,
                'duration_ms': duration_ms,
                'timestamp': datetime.now().isoformat(),
                'context_keys': list(context.keys()) if context else []
            }

        # Log evaluation if monitoring enabled (avoid recursion)
        if key != "enable_migration_logging" and self.is_enabled("enable_migration_logging", context):
            self.logger.debug(
                f"Feature flag evaluated: {key} = {value}",
                extra={
                    'flag_key': key,
                    'value': value,
                    'duration_ms': duration_ms,
                    'context': context
                }
            )

        return value

    def is_enabled(self, key: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a boolean feature flag is enabled.

        Args:
            key: Flag key
            context: Optional evaluation context

        Returns:
            True if flag is enabled, False otherwise
        """
        value = self.evaluate_flag(key, context)
        return bool(value)

    def load_config(self, config_file: str) -> None:
        """
        Load feature flag configuration from JSON file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            for flag_data in config.get('flags', []):
                flag = FeatureFlag(
                    key=flag_data['key'],
                    flag_type=FeatureFlagType(flag_data['type']),
                    default_value=flag_data['default_value'],
                    description=flag_data.get('description', '')
                )
                self.register_flag(flag)

            with self._lock:
                self._evaluation_stats['last_reload'] = datetime.now().isoformat()

            self.logger.info(f"Loaded {len(config.get('flags', []))} feature flags from {config_file}")

        except Exception as e:
            self.logger.error(f"Failed to load feature flag config: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get feature flag evaluation statistics.

        Returns:
            Dictionary with evaluation statistics
        """
        with self._lock:
            return {
                **self._evaluation_stats,
                'cache_hit_rate': (
                    self._evaluation_stats['cache_hits'] /
                    max(1, self._evaluation_stats['total_evaluations'])
                ),
                'registered_flags': list(self._flags.keys()),
                'timestamp': datetime.now().isoformat()
            }

    def clear_all_caches(self) -> None:
        """Clear all cached flag values."""
        with self._lock:
            for flag in self._flags.values():
                flag.clear_cache()
            self.logger.info("All feature flag caches cleared")

    def get_flag_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific feature flag.

        Args:
            key: Flag key

        Returns:
            Dictionary with flag information or None if not found
        """
        if key not in self._flags:
            return None

        flag = self._flags[key]
        stats = self._evaluation_stats['flag_evaluations'].get(key, {})

        return {
            'key': flag.key,
            'type': flag.flag_type.value,
            'default_value': flag.default_value,
            'description': flag.description,
            'evaluations': stats.get('evaluations', 0),
            'last_evaluation': stats.get('last_evaluation'),
            'is_cached': (
                flag._cached_value is not None and
                flag._cache_expiry and
                datetime.now() < flag._cache_expiry
            )
        }

    def list_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered flags.

        Returns:
            Dictionary mapping flag keys to flag information
        """
        return {key: self.get_flag_info(key) for key in self._flags.keys()}


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager(config_file: Optional[str] = None) -> FeatureFlagManager:
    """
    Get global feature flag manager instance.

    Args:
        config_file: Optional configuration file path

    Returns:
        FeatureFlagManager instance
    """
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager(config_file)
    return _feature_flag_manager


# Convenience functions for common flag checks
def is_direct_db_enabled(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if direct database mode is enabled.

    Args:
        context: Optional evaluation context

    Returns:
        True if direct DB is enabled, False otherwise
    """
    manager = get_feature_flag_manager()
    return manager.is_enabled("direct_db_enabled", context)


def should_use_direct_client(
    hook_name: str,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Check if a specific hook should use direct database client.

    Args:
        hook_name: Name of the hook (post_tool_use, pre_tool_use, etc.)
        context: Optional evaluation context

    Returns:
        True if hook should use direct client, False otherwise
    """
    manager = get_feature_flag_manager()

    # Check if direct DB is globally enabled
    if not manager.is_enabled("direct_db_enabled", context):
        return False

    # Check if force MCP fallback is enabled
    if manager.is_enabled("force_mcp_fallback", context):
        return False

    # Check hook-specific flag
    hook_flag_map = {
        "post_tool_use": "direct_db_post_tool_use",
        "pre_tool_use": "direct_db_pre_tool_use",
        "user_query_context_enhancer": "direct_db_context_enhancer"
    }

    flag_key = hook_flag_map.get(hook_name)
    if flag_key:
        return manager.is_enabled(flag_key, context)

    return True


# Convenience function for gradual migration
def get_migration_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get current migration status for all components.

    Args:
        context: Optional evaluation context

    Returns:
        Dictionary with migration status for all components
    """
    manager = get_feature_flag_manager()

    return {
        "direct_db_enabled": manager.is_enabled("direct_db_enabled", context),
        "post_tool_use": manager.is_enabled("direct_db_post_tool_use", context),
        "pre_tool_use": manager.is_enabled("direct_db_pre_tool_use", context),
        "context_enhancer": manager.is_enabled("direct_db_context_enhancer", context),
        "robustness_patterns": manager.is_enabled("enable_robustness_patterns", context),
        "connection_pooling": manager.is_enabled("enable_connection_pooling", context),
        "circuit_breaker": manager.is_enabled("enable_circuit_breaker", context),
        "performance_monitoring": manager.is_enabled("enable_performance_monitoring", context),
        "migration_logging": manager.is_enabled("enable_migration_logging", context)
    }


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing feature flag system...")

    # Test basic flag evaluation
    manager = get_feature_flag_manager()

    # Test boolean flag
    result = manager.evaluate_flag("direct_db_enabled")
    print(f"direct_db_enabled: {result}")

    # Test flag info
    info = manager.get_flag_info("direct_db_enabled")
    print(f"Flag info: {info}")

    # Test convenience functions
    context = {"session_id": "test-session-123"}
    enabled = is_direct_db_enabled(context)
    print(f"Direct DB enabled for session: {enabled}")

    # Test migration status
    status = get_migration_status(context)
    print(f"Migration status: {status}")

    # Test statistics
    stats = manager.get_stats()
    print(f"Evaluation stats: {stats}")

    print("✅ Feature flag system test completed!")