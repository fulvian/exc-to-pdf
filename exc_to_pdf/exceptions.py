"""
Custom exceptions for Excel processing operations.

This module provides a hierarchy of custom exceptions for handling various
error conditions that can occur during Excel file processing.
"""

from typing import Dict, Optional, Any


class ExcelReaderError(Exception):
    """
    Base exception for all Excel reading operations.

    Provides context for Excel file processing errors including file path
    and additional context information for debugging.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ExcelReader error.

        Args:
            message: Human-readable error message
            file_path: Path to the Excel file being processed
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation with file path context."""
        if self.file_path:
            return f"{self.message} (file: {self.file_path})"
        return self.message


class InvalidFileException(ExcelReaderError):
    """
    Exception raised when the file is not a valid Excel file or doesn't exist.
    """

    pass


class WorkbookException(ExcelReaderError):
    """
    Exception raised when workbook operations fail.
    """

    pass


class WorksheetNotFoundException(ExcelReaderError):
    """
    Exception raised when a requested worksheet doesn't exist.
    """

    pass


class DataExtractionException(ExcelReaderError):
    """
    Exception raised when data extraction from worksheet fails.
    """

    pass


class WorkbookInitializationException(ExcelReaderError):
    """
    Exception raised when workbook initialization fails.
    """

    pass


class ConfigurationException(ExcelReaderError):
    """
    Exception raised when configuration is invalid.
    """

    pass


class PDFGenerationException(ExcelReaderError):
    """
    Exception raised when PDF generation operations fail.
    """

    pass


class TableRenderingException(ExcelReaderError):
    """
    Exception raised when table rendering for PDF fails.
    """

    pass
