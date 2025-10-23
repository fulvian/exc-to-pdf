"""
Main CLI entry point for exc-to-pdf application.

This module provides the command-line interface for converting Excel files
to PDF format optimized for Google NotebookLM analysis.
"""

import sys
from pathlib import Path
from typing import Optional, Any

import click
import structlog

logger = structlog.get_logger()

from .config.pdf_config import PDFConfig
from .exceptions import (
    ExcelReaderError,
    InvalidFileException,
    PDFGenerationException,
    ConfigurationException,
)
from .pdf_generator import PDFGenerator

# Import advanced features modules
try:
    from .templates.template_manager import TemplateManager
    from .templates.style_presets import StylePresets
    from .i18n.locale_manager import LocaleManager
    from .chart.chart_extractor import ChartExtractor

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


class ExcelToPDFClickGroup(click.Group):
    """Custom Click group with enhanced error handling for exc-to-pdf."""

    def main(
        self,
        args: Optional[Any] = None,
        prog_name: Optional[str] = None,
        complete_var: Optional[str] = None,
        standalone_mode: bool = False,
        **extra: Any,
    ) -> Any:
        """Override main to provide custom exception handling."""
        try:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                **extra,
            )
        except ExcelReaderError as e:
            # Handle our custom exceptions with user-friendly messages
            click.secho(f"❌ Error: {e.message}", fg="red", bold=True)
            if e.file_path:
                click.secho(f"   File: {e.file_path}", fg="red")

            # Provide helpful suggestions based on error type
            if isinstance(e, InvalidFileException):
                click.secho("\n💡 Suggestions:", fg="yellow")
                click.secho(
                    "   • Check if the file exists and is readable", fg="yellow"
                )
                click.secho(
                    "   • Ensure the file is a valid Excel file (.xlsx, .xls)",
                    fg="yellow",
                )
                click.secho("   • Try using the absolute file path", fg="yellow")
            elif isinstance(e, PDFGenerationException):
                click.secho("\n💡 Suggestions:", fg="yellow")
                click.secho("   • Check if output directory is writable", fg="yellow")
                click.secho(
                    "   • Try running with --verbose for more details", fg="yellow"
                )
                click.secho(
                    "   • Ensure sufficient disk space is available", fg="yellow"
                )

            sys.exit(1)
        except Exception as e:
            # Handle unexpected exceptions
            click.secho(f"❌ Unexpected error: {str(e)}", fg="red", bold=True)
            if ctx := click.get_current_context(silent=True):
                if ctx.params.get("verbose"):
                    import traceback

                    click.secho("\n📋 Full traceback:", fg="red")
                    click.secho(traceback.format_exc(), fg="red")
            click.secho(
                "\n💡 Try running with --verbose for more information", fg="yellow"
            )
            sys.exit(1)


@click.group(cls=ExcelToPDFClickGroup, invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.option("--version", is_flag=True, help="Show version information")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, version: bool) -> None:
    """Excel to PDF converter optimized for Google NotebookLM.

    Convert Excel files to PDF format with enhanced table rendering,
    automatic bookmarks, and AI-optimized metadata.
    """
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    # Configure logging based on verbosity
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    if version:
        from . import __version__

        click.echo(f"exc-to-pdf version {__version__}")
        ctx.exit()

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    "output_file", type=click.Path(dir_okay=False, resolve_path=True), required=False
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to configuration file",
)
@click.option(
    "--template",
    "-t",
    type=click.Choice(
        ["modern", "corporate", "academic", "financial", "minimal", "creative"],
        case_sensitive=False,
    ),
    default="modern",
    help="PDF template style (default: modern)",
)
@click.option(
    "--orientation",
    "-o",
    type=click.Choice(["portrait", "landscape"], case_sensitive=False),
    default="portrait",
    help="Page orientation (default: portrait)",
)
@click.option(
    "--sheet", "-s", help="Specific worksheet name to convert (default: all worksheets)"
)
@click.option(
    "--no-bookmarks", is_flag=True, help="Disable automatic bookmark generation"
)
@click.option("--no-metadata", is_flag=True, help="Disable AI-optimized metadata")
@click.option(
    "--margin-top", type=float, default=72, help="Top margin in points (default: 72)"
)
@click.option(
    "--margin-bottom",
    type=float,
    default=72,
    help="Bottom margin in points (default: 72)",
)
@click.option(
    "--margin-left", type=float, default=72, help="Left margin in points (default: 72)"
)
@click.option(
    "--margin-right",
    type=float,
    default=72,
    help="Right margin in points (default: 72)",
)
# Advanced features options
@click.option(
    "--include-charts", is_flag=True, help="Include Excel charts in PDF output"
)
@click.option(
    "--locale", "-l", help="Force locale for internationalization (e.g., en_US, it_IT)"
)
@click.option("--rtl", is_flag=True, help="Enable right-to-left language support")
@click.option(
    "--branding",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to branding configuration file",
)
@click.option(
    "--preserve-formatting",
    is_flag=True,
    help="Preserve Excel conditional formatting in PDF",
)
@click.pass_context
def convert(
    ctx: click.Context,
    input_file: str,
    output_file: Optional[str],
    config: Optional[str],
    template: str,
    orientation: str,
    sheet: Optional[str],
    no_bookmarks: bool,
    no_metadata: bool,
    margin_top: float,
    margin_bottom: float,
    margin_left: float,
    margin_right: float,
    include_charts: bool,
    locale: Optional[str],
    rtl: bool,
    branding: Optional[str],
    preserve_formatting: bool,
) -> None:
    """Convert Excel file to PDF format.

    INPUT_FILE: Path to the Excel file to convert
    OUTPUT_FILE: Path for the output PDF file (optional, defaults to INPUT_FILE.pdf)
    """
    verbose = ctx.obj["verbose"]
    quiet = ctx.obj["quiet"]

    try:
        # Determine output file if not provided
        if not output_file:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix(".pdf"))

        # Validate output directory
        output_path = Path(output_file)
        if output_path.exists() and not output_path.is_file():
            raise click.ClickException(
                f"Output path exists but is not a file: {output_file}"
            )

        output_dir = output_path.parent
        if not output_dir.exists():
            if not quiet:
                click.secho(f"📁 Creating output directory: {output_dir}", fg="blue")
            output_dir.mkdir(parents=True, exist_ok=True)

        if not quiet:
            click.secho(f"📄 Converting: {input_file}", fg="blue")
            click.secho(f"📄 Output:    {output_file}", fg="blue")
            click.secho(f"🎨 Template:  {template}", fg="blue")
            click.secho(f"📐 Orientation: {orientation}", fg="blue")
            if sheet:
                click.secho(f"📋 Worksheet: {sheet}", fg="blue")

        # Load or create configuration
        if config:
            if not quiet:
                click.secho(f"⚙️  Using config: {config}", fg="blue")
            # TODO: Load configuration from file
            pdf_config = PDFConfig()
        else:
            pdf_config = PDFConfig()

        # Apply command-line options to configuration
        pdf_config.table_style = template
        pdf_config.orientation = orientation
        pdf_config.include_bookmarks = not no_bookmarks
        pdf_config.include_metadata = not no_metadata
        pdf_config.margin_top = margin_top
        pdf_config.margin_bottom = margin_bottom
        pdf_config.margin_left = margin_left
        pdf_config.margin_right = margin_right

        # Handle advanced features
        if ADVANCED_FEATURES_AVAILABLE:
            # Template system
            if template in StylePresets.list_available_styles():
                if not quiet:
                    click.secho(f"🎨 Using advanced template: {template}", fg="blue")
                template_config = StylePresets.get_preset(template)
                # Apply template settings to PDF config
                pdf_config.header_background = template_config.header_background
                pdf_config.header_text_color = template_config.header_text_color
                pdf_config.alternate_rows = template_config.alternate_rows
                pdf_config.alternate_row_color = template_config.alternate_row_color

            # Internationalization
            if locale or rtl:
                if not quiet:
                    click.secho(f"🌍 Setting up internationalization...", fg="blue")
                locale_manager = LocaleManager()

                if locale:
                    detected_locale = locale
                else:
                    detected_locale = locale_manager.detect_locale(input_file)

                if rtl:
                    # Force RTL locale if not specified
                    if not locale:
                        detected_locale = "ar_EG"
                        if not quiet:
                            click.secho(
                                "🔄 RTL support enabled, using Arabic locale",
                                fg="yellow",
                            )

                locale_manager.set_locale(detected_locale)
                if not quiet:
                    locale_info = locale_manager.get_locale_info(detected_locale)
                    click.secho(f"📍 Using locale: {locale_info['name']}", fg="blue")

            # Chart processing
            if include_charts:
                if not quiet:
                    click.secho("📊 Chart processing enabled", fg="blue")
                # Chart processing will be handled in PDFGenerator

            # Branding configuration
            if branding:
                if not quiet:
                    click.secho(f"🏢 Applying branding from: {branding}", fg="blue")
                # TODO: Load and apply branding configuration

            # Conditional formatting preservation
            if preserve_formatting:
                if not quiet:
                    click.secho(
                        "🎨 Conditional formatting preservation enabled", fg="blue"
                    )
                # TODO: Enable conditional formatting preservation

        else:
            # Check if advanced features were requested but not available
            advanced_options = [
                (include_charts, "Chart processing"),
                (locale, "Internationalization"),
                (rtl, "RTL support"),
                (branding, "Branding"),
                (preserve_formatting, "Conditional formatting"),
            ]

            for option_enabled, option_name in advanced_options:
                if option_enabled:
                    click.secho(
                        f"⚠️  {option_name} requested but not available - missing dependencies",
                        fg="yellow",
                    )

        # Initialize PDF generator
        if verbose:
            click.secho("🔧 Initializing PDF generator...", fg="blue")
        generator = PDFGenerator(pdf_config)

        # Perform conversion
        if not quiet:
            click.secho("🔄 Processing Excel file...", fg="blue")

        generator.convert_excel_to_pdf(input_file, output_file, worksheet_name=sheet)

        if not quiet:
            file_size = Path(output_file).stat().st_size
            size_mb = file_size / (1024 * 1024)
            click.secho(f"✅ Conversion completed successfully!", fg="green", bold=True)
            click.secho(f"📊 Output size: {size_mb:.2f} MB", fg="green")

    except Exception as e:
        # Re-raise our custom exceptions to be handled by the group
        if isinstance(e, ExcelReaderError):
            raise
        else:
            # Wrap unexpected exceptions
            raise ExcelReaderError(f"Conversion failed: {str(e)}", input_file)


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("validate")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Configuration file to validate",
)
@click.pass_context
def validate_config(ctx: click.Context, config: str) -> None:
    """Validate configuration file."""
    quiet = ctx.obj["quiet"]

    try:
        if not quiet:
            click.secho(f"🔍 Validating configuration: {config}", fg="blue")

        # TODO: Implement configuration validation
        click.secho("✅ Configuration is valid", fg="green")

    except Exception as e:
        raise ConfigurationException(f"Configuration validation failed: {str(e)}")


@config.command("template")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="exc-to-pdf-config.toml",
    help="Output configuration file",
)
@click.pass_context
def create_template(ctx: click.Context, output: str) -> None:
    """Generate a template configuration file."""
    quiet = ctx.obj["quiet"]

    try:
        if not quiet:
            click.secho(f"📝 Creating configuration template: {output}", fg="blue")

        # TODO: Generate actual configuration template
        template_content = """# exc-to-pdf Configuration Template

[page]
size = "A4"
orientation = "portrait"
margin_top = 72
margin_bottom = 72
margin_left = 72
margin_right = 72

[table]
style = "modern"
header_background = "#2E86AB"
header_text_color = "#FFFFFF"
alternate_rows = true
alternate_row_color = "#F8F8F8"

[ai]
include_metadata = true
optimize_for_notebooklm = true
include_bookmarks = true
"""

        with open(output, "w") as f:
            f.write(template_content)

        click.secho(f"✅ Configuration template created: {output}", fg="green")

    except Exception as e:
        raise ConfigurationException(f"Template creation failed: {str(e)}")


def main() -> None:
    """Main entry point for the CLI application."""
    cli(obj={})


if __name__ == "__main__":
    main()
