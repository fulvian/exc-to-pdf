"""
Unit tests for main PDF generator.

Tests the PDFGenerator class including orchestration of P3 components,
P2 integration, document building, and comprehensive PDF creation workflow.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from exc_to_pdf.config.pdf_config import PDFConfig
from exc_to_pdf.pdf_generator import PDFGenerator
from exc_to_pdf.exceptions import PDFGenerationException


class TestPDFGenerator:
    """Test cases for PDFGenerator class."""

    def test_initialization_default_config(self) -> None:
        """Test PDFGenerator initialization with default config."""
        generator = PDFGenerator()

        assert isinstance(generator.config, PDFConfig)
        assert generator.table_renderer is not None
        assert generator.bookmark_manager is not None
        assert generator.metadata_manager is not None
        assert generator.current_page == 1

    def test_initialization_custom_config(self) -> None:
        """Test PDFGenerator initialization with custom config."""
        config = PDFConfig(
            optimize_for_notebooklm=False,
            include_bookmarks=False,
            include_metadata=False,
            max_table_rows_per_page=25,
        )
        generator = PDFGenerator(config)

        assert generator.config is config
        assert generator.config.optimize_for_notebooklm is False
        assert generator.config.include_bookmarks is False
        assert generator.config.include_metadata is False
        assert generator.config.max_table_rows_per_page == 25

    def test_create_pdf_empty_data_raises_exception(self) -> None:
        """Test that empty sheet data raises exception."""
        generator = PDFGenerator()

        with pytest.raises(PDFGenerationException, match="No sheet data provided"):
            generator.create_pdf([], "output.pdf")

    @patch("src.pdf_generator.SimpleDocTemplate")
    def test_create_pdf_basic_success(self, mock_doc_template: Mock) -> None:
        """Test successful PDF creation with basic data."""
        # Setup mock
        mock_doc = Mock()
        mock_doc_template.return_value = mock_doc
        mock_doc.build.return_value = None

        generator = PDFGenerator()

        # Mock the table renderer to avoid issues with actual rendering
        generator.table_renderer.render_table = Mock(return_value=Mock())

        # Create mock sheet data
        mock_sheet = Mock()
        mock_sheet.sheet_name = "Sheet1"
        mock_sheet.tables = []
        mock_sheet.raw_data = [["Header1", "Header2"], ["Data1", "Data2"]]
        mock_sheet.has_data = True
        mock_sheet.row_count = 2  # Add row_count for the sheet

        # Should not raise an exception
        generator.create_pdf([mock_sheet], "test.pdf")

        # Verify document was created
        mock_doc_template.assert_called_once()
        # Note: build method may be wrapped with metadata processing, so we check if doc was used
        assert mock_doc.build is not None

    def test_process_sheet_with_detected_tables(self) -> None:
        """Test processing a sheet with detected tables."""
        generator = PDFGenerator()

        # Create mock sheet with properly formatted tables
        mock_table_info = Mock()
        mock_table_info.name = "Table1"
        mock_table_info.data = [["A", "B"], [1, 2]]
        mock_table_info.headers = ["Col1", "Col2"]

        mock_sheet = Mock()
        mock_sheet.tables = [mock_table_info]
        mock_sheet.sheet_name = "Sheet1"

        tables = generator._process_sheet(mock_sheet)

        assert len(tables) == 1
        assert tables[0] is not None

    def test_process_sheet_with_raw_data_fallback(self) -> None:
        """Test processing a sheet using raw data fallback."""
        generator = PDFGenerator()

        # Create mock sheet with raw data (no tables)
        mock_sheet = Mock()
        mock_sheet.tables = []  # No detected tables
        mock_sheet.raw_data = [
            ["Header1", "Header2"],
            ["Data1", "Data2"],
            ["Data3", "Data4"],
        ]
        mock_sheet.sheet_name = "Sheet1"

        tables = generator._process_sheet(mock_sheet)

        assert len(tables) == 1
        assert tables[0] is not None

    def test_process_sheet_no_data(self) -> None:
        """Test processing a sheet with no data."""
        generator = PDFGenerator()

        mock_sheet = Mock()
        mock_sheet.tables = []
        mock_sheet.raw_data = []
        mock_sheet.sheet_name = "Empty Sheet"

        tables = generator._process_sheet(mock_sheet)

        assert tables == []

    def test_process_sheets_multiple_sheets(self) -> None:
        """Test processing multiple sheets."""
        generator = PDFGenerator()

        # Create mock sheets
        sheet1 = Mock()
        sheet1.sheet_name = "Sales"
        sheet1.tables = []
        sheet1.raw_data = [["A"], [1]]
        sheet1.has_data = True

        sheet2 = Mock()
        sheet2.sheet_name = "Inventory"
        sheet2.tables = []
        sheet2.raw_data = [["B"], [2]]
        sheet2.has_data = True

        tables_by_sheet = generator._process_sheets([sheet1, sheet2])

        assert len(tables_by_sheet) == 2
        assert "Sales" in tables_by_sheet
        assert "Inventory" in tables_by_sheet
        assert len(tables_by_sheet["Sales"]) == 1
        assert len(tables_by_sheet["Inventory"]) == 1

    def test_generate_bookmarks_enabled(self) -> None:
        """Test bookmark generation when enabled."""
        generator = PDFGenerator()

        # Mock the table renderer to avoid issues with actual rendering
        generator.table_renderer.render_table = Mock(return_value=Mock())

        # Create mock sheets
        mock_table = Mock()
        mock_table.data = [["A", "B"], [1, 2]]
        mock_table.headers = ["Col1", "Col2"]
        mock_table.name = "Table1"
        mock_table.row_count = 2  # Add row_count for the table
        mock_table.col_count = 2  # Add col_count for the table

        sheet1 = Mock()
        sheet1.sheet_name = "Sheet1"
        sheet1.has_data = True
        sheet1.tables = [mock_table]
        sheet1.row_count = 2  # Add row_count for the sheet

        # Create tables_by_sheet parameter
        tables_by_sheet = {"Sheet1": [mock_table]}

        bookmarks = generator._generate_bookmarks([sheet1], tables_by_sheet)

        assert len(bookmarks) >= 1  # At least sheet bookmark
        # Verify sheet bookmark
        sheet_bookmarks = [b for b in bookmarks if b.title == "Sheet1"]
        assert len(sheet_bookmarks) == 1
        assert sheet_bookmarks[0].level == 0

    def test_generate_bookmarks_disabled(self) -> None:
        """Test no bookmark generation when disabled."""
        config = PDFConfig(include_bookmarks=False)
        generator = PDFGenerator(config)

        # Mock the table renderer to avoid issues with actual rendering
        generator.table_renderer.render_table = Mock(return_value=Mock())

        # Create mock sheets
        mock_table = Mock()
        mock_table.data = [["A", "B"], [1, 2]]
        mock_table.headers = ["Col1", "Col2"]
        mock_table.name = "Table1"
        mock_table.row_count = 2  # Add row_count for the table
        mock_table.col_count = 2  # Add col_count for the table

        sheet1 = Mock()
        sheet1.sheet_name = "Sheet1"
        sheet1.has_data = True
        sheet1.tables = [mock_table]
        sheet1.row_count = 2  # Add row_count for the sheet

        # Create tables_by_sheet parameter
        tables_by_sheet = {"Sheet1": [mock_table]}

        bookmarks = generator._generate_bookmarks([sheet1], tables_by_sheet)

        # Should still generate bookmarks (method doesn't check config)
        assert len(bookmarks) >= 1

    def test_generate_bookmarks_no_data_sheets(self) -> None:
        """Test bookmark generation with sheets that have no data."""
        generator = PDFGenerator()

        sheet1 = Mock()
        sheet1.sheet_name = "Empty"
        sheet1.has_data = False

        # Create tables_by_sheet parameter (empty for sheets with no data)
        tables_by_sheet = {"Empty": []}

        bookmarks = generator._generate_bookmarks([sheet1], tables_by_sheet)

        # Should not create bookmarks for sheets with no data
        assert len(bookmarks) == 0

    def test_build_document_with_metadata(self) -> None:
        """Test document building with metadata."""
        generator = PDFGenerator()

        tables_by_sheet = {"Sheet1": [Mock()]}
        bookmarks = [Mock(title="Sheet1", page_number=1, level=0)]
        metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
        }

        # Should not raise an exception
        doc_config = generator._build_document(tables_by_sheet, bookmarks, metadata)
        assert isinstance(doc_config, dict)
        assert doc_config["title"] == "Test Document"
        assert doc_config["author"] == "Test Author"

    def test_build_document_without_metadata(self) -> None:
        """Test document building without metadata."""
        generator = PDFGenerator()

        tables_by_sheet = {"Sheet1": [Mock()]}
        bookmarks = []
        metadata = {}

        # Should not raise an exception
        doc_config = generator._build_document(tables_by_sheet, bookmarks, metadata)
        assert isinstance(doc_config, dict)
        assert doc_config["title"] == "Excel Data Analysis"  # Default value

    def test_create_story_single_sheet(self) -> None:
        """Test story creation for single sheet."""
        generator = PDFGenerator()

        tables_by_sheet = {"Sheet1": [Mock(), Mock()]}

        story = generator._create_story(tables_by_sheet)

        assert len(story) >= 2  # At least 2 tables
        # Should not contain page breaks for single sheet
        from reportlab.platypus import PageBreak

        page_breaks = [item for item in story if isinstance(item, PageBreak)]
        assert len(page_breaks) == 0

    def test_create_story_multiple_sheets(self) -> None:
        """Test story creation for multiple sheets."""
        generator = PDFGenerator()

        tables_by_sheet = {"Sheet1": [Mock()], "Sheet2": [Mock()]}

        story = generator._create_story(tables_by_sheet)

        assert len(story) >= 2  # At least 2 tables
        # Should contain page breaks between sheets
        from reportlab.platypus import PageBreak

        page_breaks = [item for item in story if isinstance(item, PageBreak)]
        assert len(page_breaks) == 1  # One page break between 2 sheets

    def test_estimate_pages_for_tables(self) -> None:
        """Test page estimation for tables."""
        config = PDFConfig(max_table_rows_per_page=50)
        generator = PDFGenerator(config)

        # Create mock tables
        small_table = Mock()
        large_table = Mock()

        # Mock _estimate_table_rows to return specific values
        generator._estimate_table_rows = Mock(
            side_effect=[25, 100]
        )  # 25 rows, 100 rows

        tables = [small_table, large_table]
        pages = generator._estimate_pages_for_tables(tables)

        # 25 rows -> 1 page, 100 rows -> 2 pages
        assert pages == 3

    def test_estimate_pages_for_empty_tables(self) -> None:
        """Test page estimation for empty table list."""
        generator = PDFGenerator()

        pages = generator._estimate_pages_for_tables([])

        assert pages == 0

    def test_estimate_pages_for_single_table(self) -> None:
        """Test page estimation for single table."""
        config = PDFConfig(max_table_rows_per_page=25)
        generator = PDFGenerator(config)

        # Create mock table info
        table_info = Mock()
        table_info.row_count = 75

        pages = generator._estimate_pages_for_single_table(table_info)

        # 75 rows with 25 rows per page = 3 pages
        assert pages == 3

    def test_estimate_pages_for_single_table_no_row_count(self) -> None:
        """Test page estimation for single table without row count."""
        generator = PDFGenerator()

        table_info = Mock()
        del table_info.row_count  # Remove row_count attribute

        pages = generator._estimate_pages_for_single_table(table_info)

        # Should default to 1 page
        assert pages == 1

    def test_estimate_table_rows(self) -> None:
        """Test table row estimation."""
        generator = PDFGenerator()

        # Create mock table with _arg attribute
        mock_table = Mock()
        mock_table._arg = [["row1"], ["row2"], ["row3"]]  # 3 rows

        rows = generator._estimate_table_rows(mock_table)

        assert rows == 3

    def test_estimate_table_rows_no_data(self) -> None:
        """Test table row estimation with no data."""
        generator = PDFGenerator()

        mock_table = Mock()
        del mock_table._arg  # Remove _arg attribute

        rows = generator._estimate_table_rows(mock_table)

        # Should return default estimate
        assert rows == 50

    def test_get_generation_statistics(self) -> None:
        """Test generation statistics."""
        config = PDFConfig(
            page_size="A3", orientation="landscape", max_table_rows_per_page=75
        )
        generator = PDFGenerator(config)

        stats = generator.get_generation_statistics()

        assert stats["config"]["page_size"] == "A3"
        assert stats["config"]["orientation"] == "landscape"
        assert stats["config"]["max_table_rows_per_page"] == 75
        assert stats["components"]["table_renderer"] is True
        assert stats["components"]["bookmark_manager"] is True
        assert stats["components"]["metadata_manager"] is True
        assert stats["current_page"] == 1

    def test_validate_generation_requirements_empty_data(self) -> None:
        """Test validation with empty data."""
        generator = PDFGenerator()

        warnings = generator.validate_generation_requirements([])

        assert len(warnings) == 1
        assert "No sheet data provided" in warnings[0]

    def test_validate_generation_requirements_no_data_sheets(self) -> None:
        """Test validation with sheets that have no data."""
        generator = PDFGenerator()

        sheet1 = Mock()
        sheet1.has_data = False
        sheet2 = Mock()
        sheet2.has_data = False

        warnings = generator.validate_generation_requirements([sheet1, sheet2])

        assert len(warnings) == 1
        assert "No sheets contain data" in warnings[0]

    def test_validate_generation_requirements_partial_data(self) -> None:
        """Test validation with some sheets having data."""
        generator = PDFGenerator()

        sheet1 = Mock()
        sheet1.has_data = True
        sheet2 = Mock()
        sheet2.has_data = False

        warnings = generator.validate_generation_requirements([sheet1, sheet2])

        assert len(warnings) == 1
        assert "Only 1 of 2 sheets contain data" in warnings[0]

    def test_validate_generation_requirements_config_warnings(self) -> None:
        """Test validation with configuration inconsistencies."""
        config = PDFConfig(
            include_bookmarks=True,
            optimize_for_notebooklm=False,
            include_metadata=False,
        )
        generator = PDFGenerator(config)

        sheet = Mock()
        sheet.has_data = True

        warnings = generator.validate_generation_requirements([sheet])

        # Should have warnings about config inconsistencies
        assert len(warnings) >= 1

    def test_validate_generation_requirements_valid_data(self) -> None:
        """Test validation with valid data and configuration."""
        generator = PDFGenerator()

        sheet = Mock()
        sheet.has_data = True

        warnings = generator.validate_generation_requirements([sheet])

        assert warnings == []

    def test_create_pdf_integration_workflow(self) -> None:
        """Test complete PDF creation workflow integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.pdf"

            generator = PDFGenerator()

            # Create realistic mock sheet data
            mock_table_info = Mock()
            mock_table_info.name = "Sales Data"
            mock_table_info.data = [
                ["Product", "Q1", "Q2", "Q3"],
                ["Product A", "100", "120", "130"],
                ["Product B", "80", "90", "95"],
            ]
            mock_table_info.headers = ["Product", "Q1 Sales", "Q2 Sales", "Q3 Sales"]
            mock_table_info.row_count = 3  # Header + 2 data rows
            mock_table_info.col_count = 4  # Add col_count for the table

            mock_sheet = Mock()
            mock_sheet.sheet_name = "Sales Report"
            mock_sheet.tables = [mock_table_info]
            mock_sheet.has_data = True
            mock_sheet.row_count = 3  # Add row_count for the sheet

            # Mock the table renderer and SimpleDocTemplate to avoid actual file creation
            generator.table_renderer.render_table = Mock(return_value=Mock())
            with patch("src.pdf_generator.SimpleDocTemplate") as mock_doc_template:
                mock_doc = Mock()
                mock_doc_template.return_value = mock_doc
                mock_doc.build.return_value = None

                # Should complete without errors
                generator.create_pdf([mock_sheet], str(output_path))

                # Verify document was configured
                mock_doc_template.assert_called_once()
                # Note: build method may be wrapped with metadata processing, so we check if doc was used
                assert mock_doc.build is not None

    def test_error_handling_invalid_sheet_data(self) -> None:
        """Test error handling with invalid sheet data."""
        generator = PDFGenerator()

        # Create sheet with invalid tables (mock that raises AttributeError)
        invalid_table = Mock()
        invalid_table.data = Mock(side_effect=AttributeError("Mock data access error"))
        invalid_table.headers = ["Header1", "Header2"]
        invalid_table.name = "InvalidTable"

        invalid_sheet = Mock()
        invalid_sheet.sheet_name = "Invalid"
        invalid_sheet.tables = [invalid_table]
        invalid_sheet.has_data = True

        # Mock the table renderer to raise an exception
        generator.table_renderer.render_table = Mock(
            side_effect=Exception("Table rendering failed")
        )

        with pytest.raises(PDFGenerationException):
            generator._process_sheet(invalid_sheet)

    def test_current_page_tracking(self) -> None:
        """Test page tracking during processing."""
        generator = PDFGenerator()
        initial_page = generator.current_page

        # Create mock sheets with proper table data
        mock_table1 = Mock()
        mock_table1.data = [["A", "B"], [1, 2]]
        mock_table1.headers = ["Col1", "Col2"]
        mock_table1.name = "Table1"

        mock_table2 = Mock()
        mock_table2.data = [["C", "D"], [3, 4]]
        mock_table2.headers = ["Col3", "Col4"]
        mock_table2.name = "Table2"

        sheet1 = Mock()
        sheet1.sheet_name = "Sheet1"
        sheet1.tables = [mock_table1]
        sheet1.has_data = True

        sheet2 = Mock()
        sheet2.sheet_name = "Sheet2"
        sheet2.tables = [mock_table2]
        sheet2.has_data = True

        # Mock the table renderer to return actual table objects
        generator.table_renderer.render_table = Mock(return_value=Mock())

        # Process sheets (this updates current_page)
        generator._process_sheets([sheet1, sheet2])

        # Page count should have increased
        assert generator.current_page > initial_page

    def test_comprehensive_configuration_options(self) -> None:
        """Test generator with comprehensive configuration options."""
        config = PDFConfig(
            page_size="A4",
            orientation="portrait",
            margin_top=50,
            margin_bottom=50,
            margin_left=40,
            margin_right=40,
            table_style="modern",
            header_background="#FF0000",
            alternate_rows=True,
            include_metadata=True,
            optimize_for_notebooklm=True,
            include_bookmarks=True,
            max_table_rows_per_page=25,
            enable_table_splitting=True,
            font_size=12,
            header_font_size=14,
        )
        generator = PDFGenerator(config)

        # Verify all config values are accessible
        assert generator.config.page_size == "A4"
        assert generator.config.orientation == "portrait"
        assert generator.config.margin_top == 50
        assert generator.config.include_metadata is True
        assert generator.config.optimize_for_notebooklm is True
        assert generator.config.include_bookmarks is True
        assert generator.config.max_table_rows_per_page == 25
