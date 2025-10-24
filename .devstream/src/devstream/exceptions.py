"""
DevStream Exception Hierarchy

Defines all custom exceptions used throughout the DevStream project.
Provides structured error handling with context and categorization.
"""


class DevStreamError(Exception):
    """Base exception for all DevStream errors"""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self):
        return self.message


class DatabaseError(DevStreamError):
    """Database-related errors"""
    pass


class ConfigurationError(DevStreamError):
    """Configuration-related errors"""
    pass


class ValidationError(DevStreamError):
    """Data validation errors"""
    pass


class MemoryError(DevStreamError):
    """Memory system errors"""
    pass


class OllamaError(DevStreamError):
    """Ollama integration errors"""
    pass


class TaskError(DevStreamError):
    """Task management errors"""
    pass