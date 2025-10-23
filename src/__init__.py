"""
exc-to-pdf - Excel to PDF converter for Google NotebookLM

A Python tool that converts Excel files into PDF documents optimized for AI analysis.
"""

__version__ = "0.1.0"
__author__ = "exc-to-pdf team"
__description__ = "Excel to PDF converter optimized for Google NotebookLM"

# Import main classes for convenient access
from .pdf_generator import PDFGenerator
from .config.pdf_config import PDFConfig
from .exceptions import (
    ExcelReaderError,
    InvalidFileException,
    WorkbookException,
    WorksheetNotFoundException,
    DataExtractionException,
    WorkbookInitializationException,
    ConfigurationException,
    PDFGenerationException,
    TableRenderingException,
)

# Define what gets imported with "from exc_to_pdf import *"
__all__ = [
    "PDFGenerator",
    "PDFConfig",
    "ExcelReaderError",
    "InvalidFileException",
    "WorkbookException",
    "WorksheetNotFoundException",
    "DataExtractionException",
    "WorkbookInitializationException",
    "ConfigurationException",
    "PDFGenerationException",
    "TableRenderingException",
]