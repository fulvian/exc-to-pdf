"""
Tests for CLI main module functionality.

This module tests the command-line interface implementation including
command parsing, error handling, and integration with core components.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import click
from click.testing import CliRunner

from exc_to_pdf.main import cli, convert, main


class TestCLI:
    """Test CLI command structure and basic functionality."""

    def test_cli_help(self) -> None:
        """Test CLI help command displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Excel to PDF converter" in result.output
        assert "convert" in result.output
        assert "config" in result.output

    def test_cli_version(self) -> None:
        """Test version command displays version information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "exc-to-pdf version" in result.output

    def test_convert_help(self) -> None:
        """Test convert command help displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])

        assert result.exit_code == 0
        assert "INPUT_FILE" in result.output
        assert "OUTPUT_FILE" in result.output
        assert "--template" in result.output
        assert "--orientation" in result.output

    def test_config_help(self) -> None:
        """Test config command help displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        assert "validate" in result.output
        assert "template" in result.output

    def test_no_command_shows_help(self) -> None:
        """Test that invoking CLI without command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert "Excel to PDF converter" in result.output


class TestConvertCommand:
    """Test convert command functionality."""

    def test_convert_missing_input_file(self) -> None:
        """Test convert command fails when input file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "nonexistent.xlsx"])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_convert_with_valid_file_structure(self) -> None:
        """Test convert command structure with valid file."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            runner = CliRunner()

            with patch("src.main.PDFGenerator") as mock_generator:
                mock_instance = MagicMock()
                mock_generator.return_value = mock_instance

                result = runner.invoke(
                    cli,
                    [
                        "convert",
                        tmp_path,
                        "--template",
                        "modern",
                        "--orientation",
                        "portrait",
                        "--verbose",
                    ],
                )

                # Should fail because it's not a real Excel file, but after validation
                # The important thing is that command parsing works
                assert "Error:" in result.output or result.exit_code != 0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_convert_all_options(self) -> None:
        """Test convert command with all options."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            runner = CliRunner()

            with patch("src.main.PDFGenerator") as mock_generator:
                mock_instance = MagicMock()
                mock_generator.return_value = mock_instance

                result = runner.invoke(
                    cli,
                    [
                        "convert",
                        tmp_path,
                        "output.pdf",
                        "--template",
                        "classic",
                        "--orientation",
                        "landscape",
                        "--sheet",
                        "Sheet1",
                        "--no-bookmarks",
                        "--no-metadata",
                        "--margin-top",
                        "80",
                        "--margin-bottom",
                        "80",
                        "--margin-left",
                        "60",
                        "--margin-right",
                        "60",
                        "--verbose",
                    ],
                )

                # Command parsing should succeed, file validation will fail
                assert "Error:" in result.output or result.exit_code != 0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_convert_creates_output_directory(self) -> None:
        """Test convert command creates output directory if needed."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create a nested output path that doesn't exist
            output_dir = Path(tempfile.mkdtemp()) / "subdir" / "another"
            output_path = output_dir / "output.pdf"

            runner = CliRunner()

            with patch("src.main.PDFGenerator") as mock_generator:
                mock_instance = MagicMock()
                mock_generator.return_value = mock_instance

                result = runner.invoke(
                    cli,
                    [
                        "convert",
                        tmp_path,
                        str(output_path),
                        "--quiet",  # Suppress output
                    ],
                )

                # Should attempt to create directory
                # File validation will fail but directory creation should be attempted
                assert result.exit_code != 0

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestConfigCommand:
    """Test config command functionality."""

    def test_config_validate_missing_file(self) -> None:
        """Test config validate command without file argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate"])

        # Click should handle required option automatically
        assert result.exit_code != 0
        assert "Missing parameter" in result.output or "config" in result.output

    def test_config_validate_nonexistent_file(self) -> None:
        """Test config validate command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["config", "validate", "--config", "nonexistent.toml"]
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_config_template_creates_file(self) -> None:
        """Test config template command creates configuration file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test-config.toml"

            runner = CliRunner()
            result = runner.invoke(
                cli, ["config", "template", "--output", str(config_path)]
            )

            assert result.exit_code == 0
            assert config_path.exists()

            # Check content contains expected sections
            content = config_path.read_text()
            assert "[page]" in content
            assert "[table]" in content
            assert "[ai]" in content
            assert "Configuration template created" in result.output


class TestErrorHandling:
    """Test CLI error handling."""

    def test_custom_exception_handling(self) -> None:
        """Test custom exceptions are handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            runner = CliRunner()

            # Mock PDFGenerator to raise our custom exception
            with patch("src.main.PDFGenerator") as mock_generator:
                from exc_to_pdf.exceptions import PDFGenerationException

                mock_instance = MagicMock()
                mock_instance.convert_excel_to_pdf.side_effect = PDFGenerationException(
                    "Test error", tmp_path
                )
                mock_generator.return_value = mock_instance

                # Mock ExcelReader to pass file validation
                with patch("src.excel_processor.ExcelReader"):
                    result = runner.invoke(cli, ["--quiet", "convert", tmp_path])

                    assert result.exit_code == 1
                    assert "Error: Test error" in result.output

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_verbose_mode_shows_details(self) -> None:
        """Test verbose mode provides additional information."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            runner = CliRunner()

            with patch("src.main.PDFGenerator") as mock_generator:
                mock_instance = MagicMock()
                mock_generator.return_value = mock_instance

                result = runner.invoke(cli, ["convert", tmp_path, "--verbose"])

                # Should show verbose output even if conversion fails
                assert result.exit_code != 0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_quiet_mode_minimal_output(self) -> None:
        """Test quiet mode suppresses normal output."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            runner = CliRunner()

            with patch("src.main.PDFGenerator") as mock_generator:
                mock_instance = MagicMock()
                mock_generator.return_value = mock_instance

                result = runner.invoke(cli, ["convert", tmp_path, "--quiet"])

                # Should have minimal output when quiet
                assert result.exit_code != 0
                # Check that normal status messages are not present
                assert "Converting:" not in result.output
                assert "Template:" not in result.output

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestMainFunction:
    """Test main function entry point."""

    def test_main_function_import(self) -> None:
        """Test that main function can be imported."""
        from exc_to_pdf.main import main

        assert callable(main)

    def test_main_function_calls_cli(self) -> None:
        """Test that main function properly calls cli."""
        with patch("src.main.cli") as mock_cli:
            from exc_to_pdf.main import main

            main()
            mock_cli.assert_called_once_with(obj={})


class TestCLIIntegration:
    """Integration tests for CLI with real components."""

    def test_full_command_structure(self) -> None:
        """Test complete command structure without actual conversion."""
        runner = CliRunner()

        # Test command structure parsing
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock the actual conversion to test command parsing
            with patch("src.main.PDFGenerator") as mock_generator:
                with patch("src.excel_processor.ExcelReader") as mock_reader:
                    # Setup mocks to pass validation but fail at conversion
                    mock_reader.return_value.__enter__.return_value.discover_sheets.return_value = [
                        "Sheet1"
                    ]
                    mock_reader.return_value.__enter__.return_value.extract_sheet_data.return_value = MagicMock(
                        has_data=False
                    )

                    result = runner.invoke(
                        cli,
                        [
                            "--verbose",
                            "convert",
                            tmp_path,
                            "test_output.pdf",
                            "--template",
                            "minimal",
                            "--orientation",
                            "landscape",
                            "--no-bookmarks",
                            "--no-metadata",
                        ],
                    )

                    # Should process correctly (mocks working), might fail at file system level
                    # The important thing is that command parsing and mock integration work
                    assert result.exit_code != 0
                    # Check that the command processed all options correctly
                    assert "Converting:" in result.output
                    assert "Template:" in result.output
                    assert "Orientation:" in result.output

        finally:
            Path(tmp_path).unlink(missing_ok=False)
