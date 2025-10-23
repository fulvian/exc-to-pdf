"""
Unit tests for ExcelReader class.

This module contains comprehensive tests for the Excel processing functionality
including initialization, sheet discovery, and data extraction.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from exc_to_pdf.excel_processor import ExcelReader, SheetData
from exc_to_pdf.exceptions import (
    InvalidFileException,
    WorkbookException,
    WorksheetNotFoundException,
    DataExtractionException,
    WorkbookInitializationException,
)
from exc_to_pdf.config.excel_config import ExcelConfig


class TestExcelReaderInit:
    """Test ExcelReader initialization."""

    def test_excel_reader_init_with_valid_file(self):
        """Test initialization with a valid Excel file path."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)  # Write some dummy data
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                assert reader.file_path == Path(tmp_file.name)
                assert reader.config is not None
                assert reader.workbook is None
                assert reader._is_read_only is False
            finally:
                os.unlink(tmp_file.name)

    def test_excel_reader_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ExcelConfig(max_file_size_mb=50, strict_validation=False)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name, config=config)
                assert reader.config.max_file_size_mb == 50
                assert reader.config.strict_validation is False
            finally:
                os.unlink(tmp_file.name)

    def test_excel_reader_init_file_not_found(self):
        """Test initialization with non-existent file raises FileNotFoundError."""
        with pytest.raises(InvalidFileException, match="File not found"):
            ExcelReader("nonexistent_file.xlsx")

    def test_excel_reader_init_invalid_extension(self):
        """Test initialization with invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"dummy content")
            tmp_file.flush()

            try:
                with pytest.raises(
                    InvalidFileException, match="Invalid file extension"
                ):
                    ExcelReader(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

    def test_excel_reader_init_empty_file(self):
        """Test initialization with empty file raises InvalidFileException."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            # Don't write anything, keep it empty
            tmp_file.flush()

            try:
                with pytest.raises(InvalidFileException, match="File is empty"):
                    ExcelReader(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

    def test_excel_reader_init_file_too_large(self):
        """Test initialization with file exceeding size limit."""
        config = ExcelConfig(max_file_size_mb=1)  # 1MB limit

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            # Write 2MB of data
            tmp_file.write(b"\x00" * (2 * 1024 * 1024))
            tmp_file.flush()

            try:
                with pytest.raises(InvalidFileException, match="File too large"):
                    ExcelReader(tmp_file.name, config=config)
            finally:
                os.unlink(tmp_file.name)


class TestDiscoverSheets:
    """Test sheet discovery functionality."""

    @patch("src.excel_processor.load_workbook")
    def test_discover_sheets_success(self, mock_load_workbook):
        """Test successful sheet discovery."""
        # Mock workbook with multiple sheets
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1", "Sheet2", "Data"]
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheets = reader.discover_sheets()

                assert sheets == ["Sheet1", "Sheet2", "Data"]
                assert reader.workbook == mock_workbook
                assert reader._is_read_only is True
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_discover_sheets_single_sheet(self, mock_load_workbook):
        """Test sheet discovery with single sheet."""
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheets = reader.discover_sheets()

                assert sheets == ["Sheet1"]
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_discover_sheets_workbook_error(self, mock_load_workbook):
        """Test sheet discovery when workbook loading fails."""
        mock_load_workbook.side_effect = Exception("Workbook error")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                with pytest.raises(
                    WorkbookInitializationException,
                    match="Failed to initialize workbook",
                ):
                    reader.discover_sheets()
            finally:
                os.unlink(tmp_file.name)


class TestExtractSheetData:
    """Test sheet data extraction functionality."""

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_success(self, mock_load_workbook):
        """Test successful sheet data extraction."""
        # Mock worksheet
        mock_worksheet = Mock()
        mock_worksheet.max_row = 3
        mock_worksheet.max_column = 2
        mock_worksheet.sheet_state = "visible"
        mock_worksheet.title = "Sheet1"
        mock_worksheet.page_setup.orientation = "portrait"
        mock_worksheet.page_setup.paperSize = 9
        mock_worksheet.tables = {}

        # Mock row iteration
        mock_worksheet.iter_rows.return_value = [
            (1, 2),  # Header row
            (3, 4),  # Data row 1
            (5, 6),  # Data row 2
        ]

        # Mock workbook
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__ = Mock(return_value=mock_worksheet)
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheet_data = reader.extract_sheet_data("Sheet1")

                assert sheet_data.sheet_name == "Sheet1"
                assert sheet_data.row_count == 3
                assert sheet_data.col_count == 2
                assert sheet_data.has_data is True
                assert sheet_data.tables == []
                assert len(sheet_data.raw_data) == 3
                assert sheet_data.raw_data[0] == [1, 2]
                assert sheet_data.metadata["title"] == "Sheet1"
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_sheet_not_found(self, mock_load_workbook):
        """Test extraction when sheet doesn't exist."""
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1", "Sheet2"]
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                with pytest.raises(
                    WorksheetNotFoundException,
                    match="Worksheet 'NonExistent' not found",
                ):
                    reader.extract_sheet_data("NonExistent")
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_worksheet_error(self, mock_load_workbook):
        """Test extraction when worksheet access fails."""
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__ = Mock(side_effect=Exception("Worksheet error"))
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                with pytest.raises(
                    DataExtractionException,
                    match="Failed to extract data from sheet 'Sheet1'",
                ):
                    reader.extract_sheet_data("Sheet1")
            finally:
                os.unlink(tmp_file.name)


class TestMemoryManagement:
    """Test memory management functionality."""

    @patch("src.excel_processor.load_workbook")
    def test_close_workbook(self, mock_load_workbook):
        """Test closing workbook releases resources."""
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                reader.discover_sheets()  # Initialize workbook

                assert reader.workbook == mock_workbook

                reader.close()

                mock_workbook.close.assert_called_once()
                assert reader.workbook is None
                assert reader._is_read_only is False
            finally:
                os.unlink(tmp_file.name)

    def test_close_no_workbook(self):
        """Test closing when no workbook is initialized."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                # Should not raise an exception
                reader.close()
                assert reader.workbook is None
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_context_manager(self, mock_load_workbook):
        """Test using ExcelReader as context manager."""
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                with ExcelReader(tmp_file.name) as reader:
                    assert reader.workbook == mock_workbook
                    assert reader._is_read_only is True

                # Workbook should be closed after context
                mock_workbook.close.assert_called_once()
            finally:
                os.unlink(tmp_file.name)

    def test_close_workbook_exception(self):
        """Test handling exceptions when closing workbook."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)

                # Mock workbook that raises exception on close
                mock_workbook = Mock()
                mock_workbook.close.side_effect = Exception("Close failed")
                reader.workbook = mock_workbook

                # Should handle exception gracefully
                reader.close()

                # Workbook should still be None despite exception
                assert reader.workbook is None
                assert reader._is_read_only is False
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_discover_sheets_with_exceptions(self, mock_load_workbook):
        """Test sheet discovery error handling."""
        # Test exception during sheet discovery
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)

                # Mock workbook that raises exception when accessing sheetnames
                mock_workbook = Mock()
                mock_workbook.sheetnames = Mock(
                    side_effect=Exception("Sheet access failed")
                )
                mock_load_workbook.return_value = mock_workbook

                with pytest.raises(WorkbookException) as exc_info:
                    reader.discover_sheets()

                assert "Failed to discover sheets" in str(exc_info.value)
                assert "Sheet access failed" in str(exc_info.value.__cause__)
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_row_limit(self, mock_load_workbook):
        """Test data extraction with row limit."""
        mock_worksheet = Mock()

        # Create mock rows that exceed the limit
        mock_rows = []
        for i in range(105):  # More than default max_row_count (100)
            mock_row = Mock()
            mock_row.__iter__ = Mock(return_value=iter([f"Cell{i}1", f"Cell{i}2"]))
            mock_rows.append(mock_row)

        mock_worksheet.iter_rows.return_value = mock_rows
        mock_worksheet.tables = {}

        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__.return_value = mock_worksheet
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheet_data = reader.extract_sheet_data("Sheet1")

                # Should have limited rows due to max_row_count
                assert len(sheet_data.raw_data) <= 100
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_empty_row_handling(self, mock_load_workbook):
        """Test handling of rows with empty trailing cells."""
        mock_worksheet = Mock()

        # Create mock rows with trailing None values
        mock_rows = [
            Mock(),  # Header row
            Mock(),  # Data row with trailing None
        ]

        # Mock iteration to return rows with trailing None values
        mock_rows[0].__iter__ = Mock(
            return_value=iter(["Header1", "Header2", None, None])
        )
        mock_rows[1].__iter__ = Mock(return_value=iter(["Data1", "Data2", None, None]))

        mock_worksheet.iter_rows.return_value = mock_rows
        mock_worksheet.tables = {}

        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__.return_value = mock_worksheet
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheet_data = reader.extract_sheet_data("Sheet1")

                # Should have cleaned trailing None values
                assert len(sheet_data.raw_data[0]) == 2  # Header row cleaned
                assert len(sheet_data.raw_data[1]) == 2  # Data row cleaned
                assert sheet_data.raw_data[0] == ["Header1", "Header2"]
                assert sheet_data.raw_data[1] == ["Data1", "Data2"]
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_extract_sheet_data_with_excel_tables(self, mock_load_workbook):
        """Test data extraction with Excel table metadata."""
        mock_worksheet = Mock()
        mock_worksheet.iter_rows.return_value = []

        # Mock Excel table
        mock_table = Mock()
        mock_table.name = "Table1"
        mock_table.ref = "A1:C10"
        mock_table.headerRowCount = 1
        mock_table.insertRow = False
        mock_table.totalsRowCount = 0

        mock_worksheet.tables = {"Table1": mock_table}

        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__.return_value = mock_worksheet
        mock_load_workbook.return_value = mock_workbook

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)
                sheet_data = reader.extract_sheet_data("Sheet1")

                # Should have extracted table metadata
                assert len(sheet_data.tables) == 1
                assert sheet_data.tables[0]["name"] == "Table1"
                assert sheet_data.tables[0]["ref"] == "A1:C10"
                assert sheet_data.tables[0]["headerRowCount"] == 1
                assert sheet_data.tables[0]["insertRow"] is False
                assert sheet_data.tables[0]["totalsRowCount"] == 0
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.load_workbook")
    def test_detect_tables_error_handling(self, mock_load_workbook):
        """Test error handling in detect_tables method."""
        # Test with non-existent sheet
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)

                mock_workbook = Mock()
                mock_workbook.sheetnames = ["Sheet1"]  # Sheet2 doesn't exist
                mock_load_workbook.return_value = mock_workbook

                with pytest.raises(WorksheetNotFoundException) as exc_info:
                    reader.detect_tables("Sheet2")

                assert "Worksheet 'Sheet2' not found" in str(exc_info.value)
            finally:
                os.unlink(tmp_file.name)

        # Test with load_workbook failure
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)

                # Mock load_workbook to raise exception
                mock_load_workbook.side_effect = Exception("Workbook load failed")

                with pytest.raises(WorkbookException) as exc_info:
                    reader.detect_tables("Sheet1")

                assert "Workbook not initialized" in str(exc_info.value)
            finally:
                os.unlink(tmp_file.name)

    @patch("src.excel_processor.DataValidator")
    @patch("src.excel_processor.load_workbook")
    def test_validate_sheet_data_integration(
        self, mock_load_workbook, mock_data_validator_class
    ):
        """Test data validation integration."""
        # Setup mocks
        mock_worksheet = Mock()
        mock_worksheet.iter_rows.return_value = []
        mock_worksheet.tables = {}

        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__.return_value = mock_worksheet
        mock_load_workbook.return_value = mock_workbook

        mock_data_validator = Mock()
        mock_validation_result = Mock()
        mock_data_validator.validate_table_data.return_value = mock_validation_result
        mock_data_validator_class.return_value = mock_data_validator

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_file.write(b"\x00" * 100)
            tmp_file.flush()

            try:
                reader = ExcelReader(tmp_file.name)

                # Mock extract_sheet_data to return test data
                mock_sheet_data = Mock()
                mock_sheet_data.raw_data = [["Header1", "Header2"], ["Data1", "Data2"]]
                with patch.object(
                    reader, "extract_sheet_data", return_value=mock_sheet_data
                ):
                    result = reader.validate_sheet_data("Sheet1")

                # Should have called data validation
                mock_data_validator.validate_table_data.assert_called_once()
                assert result == mock_validation_result
            finally:
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
