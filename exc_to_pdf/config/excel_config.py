"""
Configuration management for Excel processing operations.

This module provides configuration classes for controlling Excel file
processing behavior, validation settings, and performance optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExcelConfig:
    """
    Configuration object for Excel file processing settings.

    Controls various aspects of Excel file processing including validation
    limits, performance settings, and error handling behavior.
    """

    # File validation settings
    max_file_size_mb: int = 500  # Increased for large file support
    allowed_extensions: List[str] = field(default_factory=lambda: [".xlsx", ".xls"])
    strict_validation: bool = True

    # Processing settings
    read_only_mode: bool = True
    enable_data_cleaning: bool = True
    max_row_count: int = 5_000_000  # Increased for large files
    max_col_count: int = 1_000

    # Performance settings
    streaming_enabled: bool = True
    chunk_size: int = 10_000
    memory_limit_mb: int = 2048  # 2GB memory limit for large files
    parallel_processing: bool = True
    max_workers: int = 4
    enable_progress: bool = True
    progress_interval: float = 1.0  # seconds

    # Caching settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    cache_to_disk: bool = True

    # Table detection settings
    enable_table_detection: bool = True
    min_table_rows: int = 2
    min_table_cols: int = 2

    # Error handling settings
    raise_on_validation_error: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if self.max_row_count <= 0:
            raise ValueError("max_row_count must be positive")
        if self.max_col_count <= 0:
            raise ValueError("max_col_count must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        if self.cache_size_mb <= 0:
            raise ValueError("cache_size_mb must be positive")
        if self.min_table_rows <= 0:
            raise ValueError("min_table_rows must be positive")
        if self.min_table_cols <= 0:
            raise ValueError("min_table_cols must be positive")
        if not self.allowed_extensions:
            raise ValueError("allowed_extensions cannot be empty")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


# Default configuration instance
DEFAULT_CONFIG = ExcelConfig()
