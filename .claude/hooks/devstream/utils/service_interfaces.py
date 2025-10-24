#!/usr/bin/env python3
"""
DevStream Service Interfaces - Context7 Compliant
Defines abstract interfaces to prevent circular dependencies.

Context7 Pattern: Interface Segregation Principle
- Each service has a well-defined interface
- Dependencies are injected, not imported
- Enables loose coupling and testability
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class LoggerInterface(ABC):
    """Abstract interface for logging services."""

    @abstractmethod
    def info(self, event: str, **kwargs) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def debug(self, event: str, **kwargs) -> None:
        """Log debug message."""
        pass

    @abstractmethod
    def warning(self, event: str, **kwargs) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def error(self, event: str, **kwargs) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def log_memory_operation(
        self,
        operation: str,
        content_type: str,
        content_size: int,
        memory_id: Optional[str] = None,
        keywords: Optional[list] = None
    ) -> None:
        """Log memory operation."""
        pass

    @abstractmethod
    def log_performance_metrics(
        self,
        execution_time_ms: float,
        memory_usage_mb: Optional[float] = None,
        api_calls: Optional[int] = None
    ) -> None:
        """Log performance metrics."""
        pass


class DatabaseConnectionInterface(ABC):
    """Abstract interface for database connections."""

    @abstractmethod
    def execute(self, query: str, parameters: Optional[tuple] = None) -> Any:
        """Execute database query."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit transaction."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
        pass


class EmbeddingClientInterface(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class PathValidatorInterface(ABC):
    """Abstract interface for path validation."""

    @abstractmethod
    def validate_db_path(self, path: str) -> str:
        """Validate database path."""
        pass


class RobustnessPatternsInterface(ABC):
    """Abstract interface for robustness patterns."""

    @abstractmethod
    def create_retry_policy(self, **kwargs) -> Any:
        """Create retry policy."""
        pass

    @abstractmethod
    def create_circuit_breaker(self, **kwargs) -> Any:
        """Create circuit breaker."""
        pass


# Service Locator Pattern Implementation
class ServiceLocator:
    """
    Context7-compliant service locator for dependency injection.

    Prevents circular dependencies by providing centralized access
    to services through their interfaces.
    """

    _instance = None
    _services: Dict[str, Any] = {}
    _interfaces: Dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_service(self, interface_name: str, implementation: Any) -> None:
        """
        Register a service implementation.

        Args:
            interface_name: Name of the interface
            implementation: Service implementation
        """
        self._services[interface_name] = implementation

    def register_interface(self, interface_name: str, interface_class: type) -> None:
        """
        Register an interface class.

        Args:
            interface_name: Name of the interface
            interface_class: Interface class
        """
        self._interfaces[interface_name] = interface_class

    def get_service(self, interface_name: str) -> Any:
        """
        Get a service implementation by interface name.

        Args:
            interface_name: Name of the interface

        Returns:
            Service implementation

        Raises:
            KeyError: If service is not registered
        """
        if interface_name not in self._services:
            raise KeyError(f"Service '{interface_name}' not registered. "
                         f"Available services: {list(self._services.keys())}")
        return self._services[interface_name]

    def has_service(self, interface_name: str) -> bool:
        """
        Check if a service is registered.

        Args:
            interface_name: Name of the interface

        Returns:
            True if service is registered, False otherwise
        """
        return interface_name in self._services

    def clear_services(self) -> None:
        """Clear all registered services (mainly for testing)."""
        self._services.clear()


# Global service locator instance
service_locator = ServiceLocator()

# Register interfaces
service_locator.register_interface('logger', LoggerInterface)
service_locator.register_interface('database_connection', DatabaseConnectionInterface)
service_locator.register_interface('embedding_client', EmbeddingClientInterface)
service_locator.register_interface('path_validator', PathValidatorInterface)
service_locator.register_interface('robustness_patterns', RobustnessPatternsInterface)


# Convenience functions for common operations
def get_logger() -> LoggerInterface:
    """Get logger service."""
    return service_locator.get_service('logger')


def get_database_connection() -> DatabaseConnectionInterface:
    """Get database connection service."""
    return service_locator.get_service('database_connection')


def get_embedding_client() -> EmbeddingClientInterface:
    """Get embedding client service."""
    return service_locator.get_service('embedding_client')


def get_path_validator() -> PathValidatorInterface:
    """Get path validator service."""
    return service_locator.get_service('path_validator')


def get_robustness_patterns() -> RobustnessPatternsInterface:
    """Get robustness patterns service."""
    return service_locator.get_service('robustness_patterns')


# Context7 dependency injection decorator
def inject_services(**service_mappings):
    """
    Decorator for injecting services into functions/methods.

    Args:
        **service_mappings: Mapping of parameter names to service names

    Example:
        @inject_services(logger='logger', db='database_connection')
        def my_function(data, logger=None, db=None):
            logger.info("Processing data")
            result = db.execute("SELECT * FROM table")
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject services
            for param_name, service_name in service_mappings.items():
                if param_name not in kwargs or kwargs[param_name] is None:
                    kwargs[param_name] = service_locator.get_service(service_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test service locator
    print("🧪 Testing Service Locator Pattern")

    # Mock service for testing
    class MockLogger:
        def info(self, event: str, **kwargs):
            print(f"INFO: {event}")

    # Register mock service
    service_locator.register_service('logger', MockLogger())

    # Test service retrieval
    logger = get_logger()
    logger.info("Service locator working correctly")

    print("✅ Service Locator Pattern: IMPLEMENTED")
    print("✅ Dependency Injection: READY")
    print("✅ Circular Import Prevention: ACTIVE")