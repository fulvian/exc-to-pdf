"""
Configuration management for PDF generation operations.

This module provides configuration classes for controlling PDF file
generation, styling, and AI optimization settings.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from ..exceptions import ConfigurationException

logger = structlog.get_logger()


@dataclass
class PDFConfig:
    """Configuration for PDF generation settings."""

    # Page settings
    page_size: str = "A4"
    orientation: str = "portrait"
    margin_top: float = 72  # points
    margin_bottom: float = 72
    margin_left: float = 72
    margin_right: float = 72

    # Table styling
    table_style: str = "modern"
    header_background: str = "#2E86AB"
    header_text_color: str = "#FFFFFF"
    alternate_rows: bool = True
    alternate_row_color: str = "#F8F8F8"

    # AI optimization
    include_metadata: bool = True
    optimize_for_notebooklm: bool = True
    include_bookmarks: bool = True

    # Performance
    max_table_rows_per_page: int = 50
    enable_table_splitting: bool = True
    font_size: int = 10
    header_font_size: int = 12

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        try:
            self._validate_config()
        except ValueError as e:
            logger.error(
                "PDF configuration validation failed",
                extra={"config_param": "unknown", "error": str(e)}
            )
            raise ConfigurationException(f"Invalid PDF configuration: {e}") from e

    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        # Validate page size
        valid_page_sizes = ["A4", "A3", "A5", "Letter", "Legal"]
        if self.page_size not in valid_page_sizes:
            raise ValueError(f"Invalid page_size: {self.page_size}. Must be one of {valid_page_sizes}")

        # Validate orientation
        valid_orientations = ["portrait", "landscape"]
        if self.orientation not in valid_orientations:
            raise ValueError(f"Invalid orientation: {self.orientation}. Must be one of {valid_orientations}")

        # Validate margins
        for margin_name, margin_value in [
            ("margin_top", self.margin_top),
            ("margin_bottom", self.margin_bottom),
            ("margin_left", self.margin_left),
            ("margin_right", self.margin_right),
        ]:
            if margin_value <= 0:
                raise ValueError(f"{margin_name} must be positive, got {margin_value}")
            if margin_value > 200:  # Reasonable upper limit in points
                raise ValueError(f"{margin_name} too large: {margin_value}. Maximum is 200 points")

        # Validate table style
        valid_table_styles = ["modern", "classic", "minimal", "corporate"]
        if self.table_style not in valid_table_styles:
            raise ValueError(f"Invalid table_style: {self.table_style}. Must be one of {valid_table_styles}")

        # Validate colors (basic hex color validation)
        for color_name, color_value in [
            ("header_background", self.header_background),
            ("header_text_color", self.header_text_color),
            ("alternate_row_color", self.alternate_row_color),
        ]:
            if not isinstance(color_value, str) or not color_value.startswith("#"):
                raise ValueError(f"{color_name} must be a hex color string (e.g., '#FF0000'), got {color_value}")
            if len(color_value) != 7:
                raise ValueError(f"{color_name} must be 7 characters hex color (e.g., '#FF0000'), got {color_value}")

        # Validate font sizes
        if self.font_size <= 0 or self.font_size > 72:
            raise ValueError(f"font_size must be between 1 and 72, got {self.font_size}")
        if self.header_font_size <= 0 or self.header_font_size > 72:
            raise ValueError(f"header_font_size must be between 1 and 72, got {self.header_font_size}")

        # Validate performance settings
        if self.max_table_rows_per_page <= 0 or self.max_table_rows_per_page > 1000:
            raise ValueError(f"max_table_rows_per_page must be between 1 and 1000, got {self.max_table_rows_per_page}")

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate page size
        valid_page_sizes = ["A4", "A3", "A5", "Letter", "Legal"]
        if self.page_size not in valid_page_sizes:
            errors.append(f"Invalid page_size: {self.page_size}. Must be one of {valid_page_sizes}")

        # Validate orientation
        valid_orientations = ["portrait", "landscape"]
        if self.orientation not in valid_orientations:
            errors.append(f"Invalid orientation: {self.orientation}. Must be one of {valid_orientations}")

        # Validate margins
        for margin_name, margin_value in [
            ("margin_top", self.margin_top),
            ("margin_bottom", self.margin_bottom),
            ("margin_left", self.margin_left),
            ("margin_right", self.margin_right),
        ]:
            if margin_value <= 0:
                errors.append(f"{margin_name} must be positive, got {margin_value}")
            if margin_value > 200:  # Reasonable upper limit in points
                errors.append(f"{margin_name} too large: {margin_value}. Maximum is 200 points")

        # Validate table style
        valid_table_styles = ["modern", "classic", "minimal", "corporate"]
        if self.table_style not in valid_table_styles:
            errors.append(f"Invalid table_style: {self.table_style}. Must be one of {valid_table_styles}")

        # Validate colors (basic hex color validation)
        for color_name, color_value in [
            ("header_background", self.header_background),
            ("header_text_color", self.header_text_color),
            ("alternate_row_color", self.alternate_row_color),
        ]:
            if not isinstance(color_value, str) or not color_value.startswith("#"):
                errors.append(f"{color_name} must be a hex color string (e.g., '#FF0000'), got {color_value}")
            elif len(color_value) != 7:
                errors.append(f"{color_name} must be 7 characters hex color (e.g., '#FF0000'), got {color_value}")

        # Validate font sizes
        if self.font_size <= 0 or self.font_size > 72:
            errors.append(f"font_size must be between 1 and 72, got {self.font_size}")
        if self.header_font_size <= 0 or self.header_font_size > 72:
            errors.append(f"header_font_size must be between 1 and 72, got {self.header_font_size}")

        # Validate performance settings
        if self.max_table_rows_per_page <= 0 or self.max_table_rows_per_page > 1000:
            errors.append(f"max_table_rows_per_page must be between 1 and 1000, got {self.max_table_rows_per_page}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "page_size": self.page_size,
            "orientation": self.orientation,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "margin_left": self.margin_left,
            "margin_right": self.margin_right,
            "table_style": self.table_style,
            "header_background": self.header_background,
            "header_text_color": self.header_text_color,
            "alternate_rows": self.alternate_rows,
            "alternate_row_color": self.alternate_row_color,
            "include_metadata": self.include_metadata,
            "optimize_for_notebooklm": self.optimize_for_notebooklm,
            "include_bookmarks": self.include_bookmarks,
            "max_table_rows_per_page": self.max_table_rows_per_page,
            "enable_table_splitting": self.enable_table_splitting,
            "font_size": self.font_size,
            "header_font_size": self.header_font_size,
        }


# Default configuration instance
DEFAULT_PDF_CONFIG = PDFConfig()