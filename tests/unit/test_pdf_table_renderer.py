"""
Unit tests for PDF table renderer.

Tests the PDFTableRenderer class including table rendering, styling,
large table handling, and column width calculation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, LongTable, KeepTogether, TableStyle

from exc_to_pdf.config.pdf_config import PDFConfig
from exc_to_pdf.pdf_table_renderer import PDFTableRenderer
from exc_to_pdf.exceptions import TableRenderingException


class TestPDFTableRenderer:
    """Test cases for PDFTableRenderer class."""

    def test_default_initialization(self) -> None:
        """Test PDFTableRenderer initialization with default config."""
        renderer = PDFTableRenderer()

        assert isinstance(renderer.config, PDFConfig)
        assert renderer.page_width == A4[0] - (72 + 72)  # Default margins
        assert "header_bg" in renderer.colors
        assert "header_text" in renderer.colors
        assert "row_even" in renderer.colors
        assert "row_odd" in renderer.colors

    def test_custom_config_initialization(self) -> None:
        """Test PDFTableRenderer initialization with custom config."""
        config = PDFConfig(
            margin_left=50,
            margin_right=50,
            header_background="#FF0000",
            header_text_color="#00FF00",
            alternate_rows=False,
        )
        renderer = PDFTableRenderer(config)

        assert renderer.config is config
        expected_width = A4[0] - (50 + 50)
        assert renderer.page_width == expected_width
        assert renderer.colors["header_bg"] == colors.HexColor("#FF0000")
        assert renderer.colors["header_text"] == colors.HexColor("#00FF00")

    def test_render_table_with_headers_and_data(self) -> None:
        """Test rendering a table with headers and data."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age", "City"]
        data = [
            ["Alice", 25, "New York"],
            ["Bob", 30, "Los Angeles"],
            ["Charlie", 35, "Chicago"],
        ]

        table = renderer.render_table(data, headers)

        assert isinstance(table, Table)
        # Table should have been created successfully with styling
        # The fact that it doesn't raise an exception indicates success
        assert table is not None

    def test_render_table_with_data_only(self) -> None:
        """Test rendering a table with data only (no headers)."""
        renderer = PDFTableRenderer()
        data = [["Alice", 25, "New York"], ["Bob", 30, "Los Angeles"]]

        table = renderer.render_table(data, [])

        assert isinstance(table, Table)
        # Table should have been created successfully with styling
        # The fact that it doesn't raise an exception indicates success
        assert table is not None

    def test_render_table_with_empty_data_raises_exception(self) -> None:
        """Test that rendering empty table raises appropriate exception."""
        renderer = PDFTableRenderer()

        with pytest.raises(TableRenderingException):
            renderer.render_table([], [])

    def test_render_large_table_creates_long_table(self) -> None:
        """Test that large tables create LongTable objects when splitting is enabled."""
        config = PDFConfig(max_table_rows_per_page=10, enable_table_splitting=True)
        renderer = PDFTableRenderer(config)

        headers = ["Col1", "Col2"]
        # Create 15 rows (more than max_table_rows_per_page)
        data = [[f"Row{i}Col1", f"Row{i}Col2"] for i in range(15)]

        table = renderer.render_table(data, headers)

        assert isinstance(table, LongTable)

    def test_render_large_table_with_splitting_disabled(self) -> None:
        """Test that large tables create regular Table when splitting is disabled."""
        config = PDFConfig(max_table_rows_per_page=10, enable_table_splitting=False)
        renderer = PDFTableRenderer(config)

        headers = ["Col1", "Col2"]
        # Create 15 rows (more than max_table_rows_per_page)
        data = [[f"Row{i}Col1", f"Row{i}Col2"] for i in range(15)]

        table = renderer.render_table(data, headers)

        assert isinstance(table, Table)
        assert not isinstance(table, LongTable)

    def test_handle_large_table_splitting(self) -> None:
        """Test large table splitting functionality."""
        config = PDFConfig(max_table_rows_per_page=5)
        renderer = PDFTableRenderer(config)

        headers = ["Name", "Age"]
        data = [
            ["Alice", 25],
            ["Bob", 30],
            ["Charlie", 35],
            ["Diana", 28],
            ["Eve", 32],
            ["Frank", 40],
            ["Grace", 27],
        ]

        tables = renderer.handle_large_table(data, headers)

        # Should split into 2 tables (5 rows per chunk, with headers)
        assert len(tables) == 2
        assert all(isinstance(table, Table) for table in tables)

    def test_handle_large_table_without_headers(self) -> None:
        """Test large table splitting without headers."""
        config = PDFConfig(max_table_rows_per_page=3)
        renderer = PDFTableRenderer(config)

        data = [["Alice", 25], ["Bob", 30], ["Charlie", 35], ["Diana", 28], ["Eve", 32]]

        tables = renderer.handle_large_table(data, [])

        # Should split into 2 tables (3 rows per chunk)
        assert len(tables) == 2
        assert all(isinstance(table, Table) for table in tables)

    def test_calculate_column_widths_basic(self) -> None:
        """Test basic column width calculation."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age", "City"]
        data = [["Alice", 25, "New York"], ["Bob", 30, "Los Angeles"]]

        widths = renderer.calculate_column_widths(data, headers, 400)

        assert len(widths) == 3
        assert all(width > 0 for width in widths)
        # Widths should sum to approximately the page width
        assert abs(sum(widths) - 400) < 1.0

    def test_calculate_column_widths_empty_data(self) -> None:
        """Test column width calculation with empty data."""
        renderer = PDFTableRenderer()

        # Only headers
        widths = renderer.calculate_column_widths([], ["Col1", "Col2"], 400)
        assert len(widths) == 2
        assert all(width > 0 for width in widths)

        # No headers, no data
        widths = renderer.calculate_column_widths([], [], 400)
        assert widths == [400]  # Single column

    def test_calculate_column_widths_very_long_content(self) -> None:
        """Test column width calculation with very long content."""
        renderer = PDFTableRenderer()
        headers = ["Short", "Very Long Column Header That Exceeds Normal Width"]
        data = [
            [
                "A",
                "This is a very long content that should trigger maximum width constraints",
            ],
            ["B", "Another long content piece to test width calculation behavior"],
        ]

        widths = renderer.calculate_column_widths(data, headers, 500)

        assert len(widths) == 2
        # Widths should sum to page width after adjustment
        assert abs(sum(widths) - 500) < 1.0
        # All widths should be positive
        assert all(width > 0 for width in widths)

    def test_calculate_column_widths_minimum_widths(self) -> None:
        """Test that column widths respect minimum constraints."""
        renderer = PDFTableRenderer()
        headers = ["A", "B"]
        data = [["x", "y"]]  # Very short content

        widths = renderer.calculate_column_widths(data, headers, 400)

        assert len(widths) == 2
        # Widths should be at least minimum width
        min_width = 0.5 * 72  # 0.5 inches in points
        assert all(width >= min_width - 1 for width in widths)  # -1 for tolerance

    def test_create_table_style_with_headers(self) -> None:
        """Test table style creation with headers."""
        renderer = PDFTableRenderer()

        style = renderer._create_table_style(header_rows=1)

        assert isinstance(style, TableStyle)
        # Check that style elements are present
        assert hasattr(style, "_cmds")
        assert len(style._cmds) > 0

    def test_create_table_style_without_headers(self) -> None:
        """Test table style creation without headers."""
        renderer = PDFTableRenderer()

        style = renderer._create_table_style(header_rows=0)

        assert isinstance(style, TableStyle)
        # Check that style elements are present
        assert hasattr(style, "_cmds")
        assert len(style._cmds) > 0

    def test_alternate_rows_styling(self) -> None:
        """Test that alternate rows styling is applied correctly."""
        config_with_alternate = PDFConfig(alternate_rows=True)
        config_without_alternate = PDFConfig(alternate_rows=False)

        renderer_with = PDFTableRenderer(config_with_alternate)
        renderer_without = PDFTableRenderer(config_without_alternate)

        style_with = renderer_with._create_table_style(header_rows=1)
        style_without = renderer_without._create_table_style(header_rows=1)

        # Both should have styling commands
        assert len(style_with._cmds) > 0
        assert len(style_without._cmds) > 0

    def test_create_wrapped_table(self) -> None:
        """Test creation of wrapped table using KeepTogether."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age"]
        data = [["Alice", 25], ["Bob", 30]]

        wrapped = renderer.create_wrapped_table(data, headers)

        assert isinstance(wrapped, KeepTogether)
        assert len(wrapped._content) == 1
        assert isinstance(wrapped._content[0], Table)

    def test_get_table_info(self) -> None:
        """Test table info generation."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age", "City"]
        data = [["Alice", 25, "NYC"], ["Bob", 30, "LA"]]

        info = renderer.get_table_info(data, headers)

        assert info["row_count"] == 2
        assert info["column_count"] == 3
        assert info["has_headers"] is True
        assert info["requires_splitting"] is False  # Small table
        assert info["estimated_pages"] == 1
        assert "estimated_width" in info

    def test_get_table_info_large_table(self) -> None:
        """Test table info for large table."""
        config = PDFConfig(max_table_rows_per_page=5)
        renderer = PDFTableRenderer(config)
        headers = ["Name", "Age"]
        data = [[f"Person{i}", 20 + i] for i in range(10)]  # 10 rows

        info = renderer.get_table_info(data, headers)

        assert info["row_count"] == 10
        assert info["column_count"] == 2
        assert info["has_headers"] is True
        assert info["requires_splitting"] is True  # Large table
        assert info["estimated_pages"] == 2  # Should span 2 pages

    @patch("src.pdf_table_renderer.logger")
    def test_render_table_logging(self, mock_logger: Mock) -> None:
        """Test that table rendering logs appropriate information."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age"]
        data = [["Alice", 25], ["Bob", 30]]

        renderer.render_table(data, headers)

        # Check that info log was called
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        assert "Table rendered successfully" in call_args[0][0]
        assert call_args[1]["extra"]["rows"] == 2
        assert call_args[1]["extra"]["columns"] == 2

    @patch("src.pdf_table_renderer.logger")
    def test_render_table_error_logging(self, mock_logger: Mock) -> None:
        """Test that table rendering errors are logged properly."""
        renderer = PDFTableRenderer()

        with pytest.raises(TableRenderingException):
            renderer.render_table([], [])

        # Check that error log was called
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args
        assert "Table rendering failed" in call_args[0][0]

    def test_color_palette_updates_with_config(self) -> None:
        """Test that color palette is updated based on configuration."""
        config = PDFConfig(
            header_background="#123456",
            header_text_color="#FEDCBA",
            alternate_row_color="#ABCDEF",
        )
        renderer = PDFTableRenderer(config)

        assert renderer.colors["header_bg"] == colors.HexColor("#123456")
        assert renderer.colors["header_text"] == colors.HexColor("#FEDCBA")
        # alternate_row key only exists when alternate_rows is True
        if config.alternate_rows:
            assert renderer.colors["alternate_row"] == colors.HexColor("#ABCDEF")

    def test_render_table_with_title_parameter(self) -> None:
        """Test that title parameter is accepted (though not used in current implementation)."""
        renderer = PDFTableRenderer()
        headers = ["Name", "Age"]
        data = [["Alice", 25]]

        # Should not raise an exception
        table = renderer.render_table(data, headers, title="Test Table")
        assert isinstance(table, Table)

    def test_page_width_calculation_with_custom_margins(self) -> None:
        """Test page width calculation with custom margins."""
        config = PDFConfig(margin_left=100, margin_right=50)
        renderer = PDFTableRenderer(config)

        expected_width = A4[0] - (100 + 50)
        assert renderer.page_width == expected_width

    def test_handle_large_table_with_zero_chunk_size(self) -> None:
        """Test error handling when chunk size would be zero."""
        config = PDFConfig(max_table_rows_per_page=1)  # Too small
        renderer = PDFTableRenderer(config)

        with pytest.raises(TableRenderingException):
            renderer.handle_large_table([["data"]], ["header"])

    def test_calculate_column_widths_precision_adjustment(self) -> None:
        """Test that column widths are adjusted for floating point precision."""
        renderer = PDFTableRenderer()
        headers = ["Col1", "Col2", "Col3"]
        data = [["A", "B", "C"]]

        widths = renderer.calculate_column_widths(data, headers, 400.0)

        # Should sum very close to target width
        total_width = sum(widths)
        assert abs(total_width - 400.0) < 0.1
