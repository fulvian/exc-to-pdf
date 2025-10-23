"""
Unit tests for PDF configuration system.

Tests the PDFConfig dataclass validation, error handling, and configuration
management functionality.
"""

import pytest

from src.config.pdf_config import PDFConfig, DEFAULT_PDF_CONFIG
from src.exceptions import ConfigurationException


class TestPDFConfig:
    """Test cases for PDFConfig class."""

    def test_default_config_creation(self) -> None:
        """Test that default configuration creates successfully."""
        config = PDFConfig()

        assert config.page_size == "A4"
        assert config.orientation == "portrait"
        assert config.margin_top == 72
        assert config.margin_bottom == 72
        assert config.margin_left == 72
        assert config.margin_right == 72
        assert config.table_style == "modern"
        assert config.header_background == "#2E86AB"
        assert config.header_text_color == "#FFFFFF"
        assert config.alternate_rows is True
        assert config.alternate_row_color == "#F8F8F8"
        assert config.include_metadata is True
        assert config.optimize_for_notebooklm is True
        assert config.include_bookmarks is True
        assert config.max_table_rows_per_page == 50
        assert config.enable_table_splitting is True
        assert config.font_size == 10
        assert config.header_font_size == 12

    def test_pdf_config_validation(self) -> None:
        """Test PDF configuration validation with valid values."""
        config = PDFConfig(
            page_size="A3",
            orientation="landscape",
            margin_top=50,
            margin_bottom=50,
            margin_left=40,
            margin_right=40,
            table_style="classic",
            header_background="#FF0000",
            header_text_color="#000000",
            alternate_rows=False,
            alternate_row_color="#EEEEEE",
            include_metadata=False,
            optimize_for_notebooklm=False,
            include_bookmarks=False,
            max_table_rows_per_page=100,
            enable_table_splitting=False,
            font_size=12,
            header_font_size=14
        )

        # Should not raise any exceptions
        config._validate_config()

        # Check values are set correctly
        assert config.page_size == "A3"
        assert config.orientation == "landscape"
        assert config.margin_top == 50
        assert config.font_size == 12
        assert config.header_font_size == 14

    def test_invalid_page_size_raises_exception(self) -> None:
        """Test that invalid page size raises ConfigurationException."""
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(page_size="INVALID_SIZE")

    def test_invalid_orientation_raises_exception(self) -> None:
        """Test that invalid orientation raises ConfigurationException."""
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(orientation="invalid_orientation")

    def test_invalid_margins_raise_exception(self) -> None:
        """Test that invalid margins raise ConfigurationException."""
        # Negative margin
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(margin_top=-10)

        # Zero margin
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(margin_bottom=0)

        # Too large margin
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(margin_left=300)

    def test_invalid_table_style_raises_exception(self) -> None:
        """Test that invalid table style raises ConfigurationException."""
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(table_style="invalid_style")

    def test_invalid_colors_raise_exception(self) -> None:
        """Test that invalid color values raise ConfigurationException."""
        # Missing # prefix
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(header_background="FF0000")

        # Wrong length
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(header_text_color="#FF0")

        # Non-string
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(alternate_row_color=123456)  # type: ignore

    def test_invalid_font_sizes_raise_exception(self) -> None:
        """Test that invalid font sizes raise ConfigurationException."""
        # Negative font size
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(font_size=-5)

        # Too large font size
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(header_font_size=100)

        # Zero font size
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(font_size=0)

    def test_invalid_max_table_rows_raises_exception(self) -> None:
        """Test that invalid max_table_rows_per_page raises ConfigurationException."""
        # Negative value
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(max_table_rows_per_page=-10)

        # Zero value
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(max_table_rows_per_page=0)

        # Too large value
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(max_table_rows_per_page=2000)

    def test_validate_method_returns_errors_list(self) -> None:
        """Test that validate method returns list of errors."""
        # Create a valid config first
        config = PDFConfig()
        # Then manually set invalid values to test validate method
        config.page_size = "INVALID_SIZE"
        config.margin_top = -10

        errors = config.validate()

        assert len(errors) == 2
        assert any("Invalid page_size" in error for error in errors)
        assert any("margin_top must be positive" in error for error in errors)

    def test_validate_method_returns_empty_list_for_valid_config(self) -> None:
        """Test that validate method returns empty list for valid configuration."""
        config = PDFConfig()
        errors = config.validate()

        assert errors == []

    def test_to_dict_method(self) -> None:
        """Test that to_dict method returns correct dictionary representation."""
        config = PDFConfig(
            page_size="A3",
            font_size=12,
            header_background="#FF0000"
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["page_size"] == "A3"
        assert config_dict["font_size"] == 12
        assert config_dict["header_background"] == "#FF0000"
        assert config_dict["orientation"] == "portrait"  # Default value
        assert len(config_dict) == 18  # All configuration fields

    def test_default_pdf_config_instance(self) -> None:
        """Test that DEFAULT_PDF_CONFIG is properly initialized."""
        assert isinstance(DEFAULT_PDF_CONFIG, PDFConfig)
        assert DEFAULT_PDF_CONFIG.page_size == "A4"
        assert DEFAULT_PDF_CONFIG.table_style == "modern"

    def test_config_with_all_valid_page_sizes(self) -> None:
        """Test configuration with all valid page sizes."""
        valid_sizes = ["A4", "A3", "A5", "Letter", "Legal"]

        for page_size in valid_sizes:
            config = PDFConfig(page_size=page_size)
            assert config.page_size == page_size
            # Should not raise any exceptions
            config._validate_config()

    def test_config_with_both_orientations(self) -> None:
        """Test configuration with both valid orientations."""
        valid_orientations = ["portrait", "landscape"]

        for orientation in valid_orientations:
            config = PDFConfig(orientation=orientation)
            assert config.orientation == orientation
            # Should not raise any exceptions
            config._validate_config()

    def test_config_with_all_table_styles(self) -> None:
        """Test configuration with all valid table styles."""
        valid_styles = ["modern", "classic", "minimal", "corporate"]

        for style in valid_styles:
            config = PDFConfig(table_style=style)
            assert config.table_style == style
            # Should not raise any exceptions
            config._validate_config()

    def test_edge_case_margins(self) -> None:
        """Test edge case margin values."""
        # Minimum valid margin
        config = PDFConfig(margin_top=0.1)
        assert config.margin_top == 0.1

        # Maximum valid margin
        config = PDFConfig(margin_bottom=200)
        assert config.margin_bottom == 200

        # Just over maximum should fail
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(margin_left=200.1)

    def test_edge_case_font_sizes(self) -> None:
        """Test edge case font size values."""
        # Minimum valid font size
        config = PDFConfig(font_size=1)
        assert config.font_size == 1

        # Maximum valid font size
        config = PDFConfig(header_font_size=72)
        assert config.header_font_size == 72

        # Just over maximum should fail
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(font_size=73)

    def test_edge_case_max_table_rows(self) -> None:
        """Test edge case max_table_rows_per_page values."""
        # Minimum valid value
        config = PDFConfig(max_table_rows_per_page=1)
        assert config.max_table_rows_per_page == 1

        # Maximum valid value
        config = PDFConfig(max_table_rows_per_page=1000)
        assert config.max_table_rows_per_page == 1000

        # Just over maximum should fail
        with pytest.raises(ConfigurationException, match="Invalid PDF configuration"):
            PDFConfig(max_table_rows_per_page=1001)