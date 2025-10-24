"""
Main PDF generation orchestrator with P2 integration and modern styling.

This module provides the main PDF generation engine that orchestrates all
P3 components (table renderer, bookmark manager, metadata manager) and
integrates with P2 Excel processing to create complete PDF documents.
"""

from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Spacer, PageBreak

import structlog

from .bookmark_manager import BookmarkManager, BookmarkInfo
from .config.pdf_config import PDFConfig
from .exceptions import PDFGenerationException
from .metadata_manager import MetadataManager
from .pdf_table_renderer import PDFTableRenderer
from .excel_processor import ExcelReader
from .excel_fallback_reader import ExcelFallbackReader

logger = structlog.get_logger()


class PDFGenerator:
    """Main PDF generation engine with P2 integration and modern styling."""

    def __init__(self, config: Optional[PDFConfig] = None) -> None:
        """Initialize PDF generator with components.

        Args:
            config: Optional PDF configuration. Uses default if None.

        Raises:
            ConfigurationException: If configuration validation fails
        """
        self.config = config or PDFConfig()

        # Initialize components
        self.table_renderer = PDFTableRenderer(self.config)
        self.bookmark_manager = BookmarkManager()
        self.metadata_manager = MetadataManager(self.config)

        # Page tracking
        self.current_page = 1

        logger.info(
            "PDF generator initialized",
            extra={
                "optimize_for_notebooklm": self.config.optimize_for_notebooklm,
                "include_bookmarks": self.config.include_bookmarks,
                "include_metadata": self.config.include_metadata,
            },
        )

    def create_pdf(
        self,
        sheet_data_list: List[Any],
        output_path: str,
        source_file: Optional[str] = None,
    ) -> None:
        """Generate PDF from Excel sheet data.

        Args:
            sheet_data_list: List of SheetData objects from P2
            output_path: Output PDF file path
            source_file: Original Excel file path for metadata

        Raises:
            PDFGenerationException: If PDF generation fails
        """
        try:
            if not sheet_data_list:
                raise ValueError("No sheet data provided for PDF generation")

            logger.info(
                "Starting PDF generation",
                extra={
                    "output_path": output_path,
                    "sheet_count": len(sheet_data_list),
                    "source_file": source_file,
                },
            )

            # Process sheets and organize tables
            tables_by_sheet = self._process_sheets(sheet_data_list)

            # Generate bookmarks if enabled (pass already processed tables)
            bookmarks = []
            if self.config.include_bookmarks:
                bookmarks = self._generate_bookmarks(sheet_data_list, tables_by_sheet)

            # Generate metadata if enabled
            metadata = {}
            if self.config.include_metadata:
                metadata = self.metadata_manager.create_pdf_metadata(
                    sheet_data_list, source_file or "unknown.xlsx"
                )

            # Build document configuration
            doc_config = self._build_document(tables_by_sheet, bookmarks, metadata)

            # Create and build PDF document
            doc = SimpleDocTemplate(output_path, **doc_config)

            # Create story with metadata if needed
            story = self._create_story(tables_by_sheet)

            # Add metadata processing to the story if metadata is available
            if metadata:
                # Override the build method to add metadata
                original_build = doc.build

                def build_with_metadata(story_elements: List[Any]) -> None:
                    def on_first_page(canvas: Any, doc_obj: Any) -> None:
                        # Add custom metadata to PDF
                        canvas.setCreator(
                            doc_config.get("creator", "exc-to-pdf v2.2.0")
                        )
                        canvas.setProducer(
                            doc_config.get("producer", "ReportLab PDF Library")
                        )

                        # Add AI optimization metadata
                        if doc_config.get("ai_optimized"):
                            canvas.setSubject(
                                f"{doc_config.get('subject', '')} (AI Optimized for NotebookLM)"
                            )

                    doc.onFirstPage = on_first_page  # type: ignore[attr-defined]
                    return original_build(story_elements)

                doc.build = build_with_metadata  # type: ignore[assignment]
                doc.build(story)
            else:
                doc.build(story)

            logger.info(
                "PDF generated successfully",
                extra={
                    "output_path": output_path,
                    "total_pages": self.current_page - 1,
                    "total_sheets": len(sheet_data_list),
                    "total_tables": sum(
                        len(tables) for tables in tables_by_sheet.values()
                    ),
                },
            )

        except Exception as e:
            logger.error(
                "PDF generation failed",
                extra={
                    "output_path": output_path,
                    "sheet_count": len(sheet_data_list) if sheet_data_list else 0,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to generate PDF: {e}") from e

    def convert_excel_to_pdf(
        self, excel_file: str, pdf_file: str, worksheet_name: Optional[str] = None
    ) -> None:
        """Convert Excel file directly to PDF format.

        This is a convenience method that combines Excel reading and PDF generation
        for CLI usage. It processes the Excel file and generates a PDF with the
        current configuration settings.

        Args:
            excel_file: Path to the input Excel file
            pdf_file: Path to the output PDF file
            worksheet_name: Optional specific worksheet name to process

        Raises:
            PDFGenerationException: If conversion fails
            InvalidFileException: If Excel file is invalid
            WorksheetNotFoundException: If specified worksheet doesn't exist
        """
        try:
            logger.info(
                "Starting Excel to PDF conversion",
                extra={
                    "excel_file": excel_file,
                    "pdf_file": pdf_file,
                    "worksheet_name": worksheet_name,
                },
            )

            # Read Excel file with fallback
            sheet_data_list = []

            try:
                # Try standard ExcelReader first
                with ExcelReader(excel_file) as reader:
                    if worksheet_name:
                        # Process specific worksheet
                        sheet_data = reader.extract_sheet_data(worksheet_name)
                        if sheet_data.has_data:
                            sheet_data_list.append(sheet_data)
                        else:
                            logger.warning(
                                "Worksheet has no data",
                                extra={"worksheet_name": worksheet_name},
                            )
                    else:
                        # Process all worksheets with data
                        sheet_names = reader.discover_sheets()
                        for sheet_name in sheet_names:
                            try:
                                sheet_data = reader.extract_sheet_data(sheet_name)
                                if sheet_data.has_data:
                                    sheet_data_list.append(sheet_data)
                            except Exception as e:
                                logger.warning(
                                    "Failed to process worksheet",
                                    extra={"worksheet_name": sheet_name, "error": str(e)},
                                )
                                continue

                if not sheet_data_list:
                    logger.warning("Standard ExcelReader found no data, trying fallback...")
                    raise PDFGenerationException("No data found in Excel file")

            except Exception as primary_error:
                logger.warning(f"Standard ExcelReader failed: {primary_error}")
                logger.info("Attempting recovery with fallback reader...")

                # Fallback to enhanced reader with recovery
                fallback_reader = ExcelFallbackReader()
                recovery_result = fallback_reader.read_excel_with_fallback(excel_file, max_attempts=3)

                if not recovery_result.success:
                    logger.error(f"All recovery methods failed: {recovery_result.error_message}")
                    raise PDFGenerationException(
                        f"Unable to read Excel file: {recovery_result.error_message}"
                    )

                logger.info(f"Recovery successful using: {recovery_result.fallback_used}")

                # Convert recovery data to SheetData format
                if recovery_result.data:
                    if isinstance(recovery_result.data, dict):
                        # Data is in format {sheet_name: [rows]}
                        for sheet_name, rows in recovery_result.data.items():
                            if rows and isinstance(rows, list):
                                # Create a mock SheetData object
                                mock_sheet_data = type('MockSheetData', (), {
                                    'sheet_name': sheet_name,
                                    'tables': [],
                                    'metadata': {'title': sheet_name},
                                    'raw_data': rows,
                                    'row_count': len(rows),
                                    'col_count': len(rows[0]) if rows and rows[0] else 0,
                                    'has_data': bool(rows)
                                })()
                                sheet_data_list.append(mock_sheet_data)
                    else:
                        # Fallback data is not a dict, cannot process
                        logger.warning(f"Unexpected data format in recovery: {type(recovery_result.data)}")
                        # Try to use empty data to prevent complete failure
                        mock_sheet_data = type('MockSheetData', (), {
                            'sheet_name': 'Recovered_Data',
                            'tables': [],
                            'metadata': {'title': 'Recovered Data'},
                            'raw_data': [['No usable data recovered']],
                            'row_count': 1,
                            'col_count': 1,
                            'has_data': False
                        })()
                        sheet_data_list.append(mock_sheet_data)

                if not sheet_data_list:
                    raise PDFGenerationException("No data found even after recovery attempts")

                # Generate PDF
                self.create_pdf(sheet_data_list, pdf_file, excel_file)

            logger.info(
                "Excel to PDF conversion completed",
                extra={
                    "excel_file": excel_file,
                    "pdf_file": pdf_file,
                    "sheets_processed": len(sheet_data_list),
                },
            )

        except Exception as e:
            if isinstance(e, PDFGenerationException):
                raise
            else:
                raise PDFGenerationException(
                    f"Excel to PDF conversion failed: {e}"
                ) from e

    def _process_sheet(self, sheet_data: Any) -> List[Any]:
        """Process a single sheet into PDF tables.

        Args:
            sheet_data: Sheet data from P2 processing

        Returns:
            List of formatted tables for PDF

        Raises:
            PDFGenerationException: If sheet processing fails
        """
        try:
            tables = []

            # Check if sheet has tables detected
            if hasattr(sheet_data, "tables") and sheet_data.tables:
                for table_info in sheet_data.tables:
                    # Extract table data
                    if hasattr(table_info, "data") and table_info.data:
                        table_data = table_info.data
                        headers = (
                            table_info.headers if hasattr(table_info, "headers") else []
                        )

                        # Create formatted table
                        table = self.table_renderer.render_table(
                            table_data, headers, table_info.name
                        )
                        tables.append(table)
                    else:
                        logger.warning(
                            "Table info missing data, skipping",
                            extra={
                                "table_name": getattr(table_info, "name", "unknown")
                            },
                        )

            # Fallback to raw data if no tables detected
            elif hasattr(sheet_data, "raw_data") and sheet_data.raw_data:
                if sheet_data.raw_data:
                    # Use first row as headers if available
                    try:
                        headers = (
                            sheet_data.raw_data[0]
                            if len(sheet_data.raw_data[0]) > 0
                            else []
                        )
                        data = (
                            sheet_data.raw_data[1:]
                            if len(sheet_data.raw_data) > 1
                            else []
                        )
                    except (TypeError, AttributeError):
                        # Handle cases where raw_data[0] doesn't support len()
                        headers = (
                            sheet_data.raw_data[0] if sheet_data.raw_data[0] else []
                        )
                        data = (
                            sheet_data.raw_data[1:]
                            if len(sheet_data.raw_data) > 1
                            else []
                        )

                    if data:
                        table = self.table_renderer.render_table(
                            data, headers, sheet_data.sheet_name
                        )
                        tables.append(table)

            if not tables:
                logger.warning(
                    "No tables created for sheet",
                    extra={"sheet_name": getattr(sheet_data, "sheet_name", "unknown")},
                )

            return tables

        except Exception as e:
            logger.error(
                "Failed to process sheet",
                extra={
                    "sheet_name": getattr(sheet_data, "sheet_name", "unknown"),
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to process sheet: {e}") from e

    def _process_sheets(self, sheet_data_list: List[Any]) -> Dict[str, List[Any]]:
        """Process all sheets and organize tables by sheet name.

        Args:
            sheet_data_list: List of sheet data objects from P2

        Returns:
            Dictionary mapping sheet names to lists of tables

        Raises:
            PDFGenerationException: If sheet processing fails
        """
        tables_by_sheet: Dict[str, List[Any]] = {}

        try:
            for sheet_data in sheet_data_list:
                sheet_name = getattr(
                    sheet_data, "sheet_name", f"Sheet_{len(tables_by_sheet) + 1}"
                )

                logger.debug(
                    "Processing sheet",
                    extra={
                        "sheet_name": sheet_name,
                        "has_data": getattr(sheet_data, "has_data", False),
                    },
                )

                # Process sheet into tables
                tables = self._process_sheet(sheet_data)
                tables_by_sheet[sheet_name] = tables

                # Add page break between sheets (except last one)
                if tables and sheet_data != sheet_data_list[-1]:
                    self.current_page += self._estimate_pages_for_tables(tables)

            return tables_by_sheet

        except Exception as e:
            logger.error(
                "Failed to process sheets",
                extra={"sheet_count": len(sheet_data_list), "error": str(e)},
            )
            raise PDFGenerationException(f"Failed to process sheets: {e}") from e

    def _generate_bookmarks(
        self, sheet_data_list: List[Any], tables_by_sheet: Dict[str, List[Any]]
    ) -> List[BookmarkInfo]:
        """Generate bookmarks for all sheets and tables.

        Args:
            sheet_data_list: List of sheet data objects
            tables_by_sheet: Pre-processed tables organized by sheet name

        Returns:
            List of bookmark information objects

        Raises:
            PDFGenerationException: If bookmark generation fails
        """
        try:
            bookmarks: List[BookmarkInfo] = []
            current_page = 1

            for sheet_data in sheet_data_list:
                if not getattr(sheet_data, "has_data", False):
                    continue

                sheet_name = getattr(
                    sheet_data, "sheet_name", f"Sheet_{len(bookmarks) + 1}"
                )

                # Add sheet bookmark
                sheet_bookmark = self.bookmark_manager.add_sheet_bookmark(
                    sheet_name, current_page
                )
                bookmarks.append(sheet_bookmark)

                # Use pre-processed tables to estimate pages
                sheet_tables = tables_by_sheet.get(sheet_name, [])
                pages_for_sheet = self._estimate_pages_for_tables(sheet_tables)

                # Add table bookmarks if tables exist
                if hasattr(sheet_data, "tables") and sheet_data.tables:
                    table_page = current_page + 1  # Tables start on next page
                    for table_info in sheet_data.tables:
                        table_name = getattr(
                            table_info, "name", f"Table_{len(bookmarks)}"
                        )
                        table_bookmark = self.bookmark_manager.add_table_bookmark(
                            table_name, table_page, sheet_name, 1
                        )
                        bookmarks.append(table_bookmark)
                        table_page += self._estimate_pages_for_single_table(table_info)

                current_page += pages_for_sheet

            return bookmarks

        except Exception as e:
            logger.error(
                "Failed to generate bookmarks",
                extra={"sheet_count": len(sheet_data_list), "error": str(e)},
            )
            raise PDFGenerationException(f"Failed to generate bookmarks: {e}") from e

    def _build_document(
        self,
        tables_by_sheet: Dict[str, List[Any]],
        bookmarks: List[BookmarkInfo],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build complete PDF document configuration.

        Args:
            tables_by_sheet: Tables organized by sheet name
            bookmarks: Bookmark structure for navigation
            metadata: PDF metadata

        Returns:
            Dictionary containing document configuration

        Raises:
            PDFGenerationException: If document building fails
        """
        try:
            # Create document configuration
            doc_config = {
                "pagesize": (
                    A4 if self.config.page_size == "A4" else self.config.page_size
                ),
                "leftMargin": self.config.margin_left,
                "rightMargin": self.config.margin_right,
                "topMargin": self.config.margin_top,
                "bottomMargin": self.config.margin_bottom,
            }

            # Add metadata to configuration
            if metadata is not None:
                doc_config.update(
                    {
                        "title": metadata.get("title", "Excel Data Analysis"),
                        "author": metadata.get("author", "exc-to-pdf converter"),
                        "subject": metadata.get(
                            "subject", "Structured data for AI analysis"
                        ),
                        "creator": metadata.get("creator", "exc-to-pdf v2.2.0"),
                        "producer": metadata.get("producer", "ReportLab PDF Library"),
                        "ai_optimized": metadata.get("ai_optimized", False),
                    }
                )

            logger.debug(
                "PDF document configuration built",
                extra={
                    "sheet_count": len(tables_by_sheet),
                    "bookmark_count": len(bookmarks),
                    "has_metadata": bool(metadata),
                },
            )

            return doc_config

        except Exception as e:
            logger.error(
                "Failed to build document configuration",
                extra={
                    "tables_count": len(tables_by_sheet) if tables_by_sheet else 0,
                    "bookmarks_count": len(bookmarks) if bookmarks else 0,
                    "metadata_fields": len(metadata) if metadata else 0,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to build document: {e}") from e

    def _create_story(self, tables_by_sheet: Dict[str, List[Any]]) -> List[Any]:
        """Create the story flowable for the PDF document.

        Args:
            tables_by_sheet: Tables organized by sheet name

        Returns:
            List of flowable elements for the document

        Raises:
            PDFGenerationException: If story creation fails
        """
        try:
            story = []
            sheet_names = list(tables_by_sheet.keys())

            for i, (sheet_name, tables) in enumerate(tables_by_sheet.items()):
                if not tables:
                    continue

                # Add sheet title if multiple sheets
                if len(sheet_names) > 1:
                    # Add title for sheet
                    story.append(Spacer(1, 20))  # Space before title

                # Add tables for this sheet
                for j, table in enumerate(tables):
                    story.append(table)

                    # Add space between tables in same sheet
                    if j < len(tables) - 1:
                        story.append(Spacer(1, 15))

                # Add page break between sheets (except last one)
                if i < len(sheet_names) - 1:
                    story.append(PageBreak())  # type: ignore[arg-type]

            return story

        except Exception as e:
            logger.error(
                "Failed to create story",
                extra={
                    "sheet_count": len(tables_by_sheet) if tables_by_sheet else 0,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to create story: {e}") from e

    def _estimate_pages_for_tables(self, tables: List[Any]) -> int:
        """Estimate number of pages required for tables.

        Args:
            tables: List of table objects

        Returns:
            Estimated number of pages
        """
        if not tables:
            return 0

        # Rough estimation: assume each table takes 1-2 pages depending on size
        pages = 0
        for table in tables:
            # Estimate based on configuration
            max_rows = self.config.max_table_rows_per_page
            estimated_rows = self._estimate_table_rows(table)
            table_pages = max(1, (estimated_rows + max_rows - 1) // max_rows)
            pages += table_pages

        return pages

    def _estimate_pages_for_single_table(self, table_info: Any) -> int:
        """Estimate pages for a single table.

        Args:
            table_info: Table information object

        Returns:
            Estimated number of pages
        """
        if hasattr(table_info, "row_count") and isinstance(
            table_info.row_count, (int, float)
        ):
            max_rows = self.config.max_table_rows_per_page
            row_count = int(table_info.row_count)
            return max(1, (row_count + max_rows - 1) // max_rows)
        return 1  # Default to 1 page

    def _estimate_table_rows(self, table: Any) -> int:
        """Estimate number of rows in a table.

        Args:
            table: Table object

        Returns:
            Estimated row count
        """
        # This is a rough estimation - in a real implementation,
        # you might want to access the actual data
        try:
            if hasattr(table, "_arg") and table._arg:
                result = len(table._arg)
                return int(result) if isinstance(result, (int, float)) else 50
            return 50  # Default estimate
        except:
            return 50  # Default estimate

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the PDF generation process.

        Returns:
            Dictionary containing generation statistics
        """
        return {
            "config": {
                "page_size": self.config.page_size,
                "orientation": self.config.orientation,
                "optimize_for_notebooklm": self.config.optimize_for_notebooklm,
                "include_bookmarks": self.config.include_bookmarks,
                "include_metadata": self.config.include_metadata,
                "max_table_rows_per_page": self.config.max_table_rows_per_page,
            },
            "components": {
                "table_renderer": self.table_renderer is not None,
                "bookmark_manager": self.bookmark_manager is not None,
                "metadata_manager": self.metadata_manager is not None,
            },
            "current_page": self.current_page,
        }

    def validate_generation_requirements(self, sheet_data_list: List[Any]) -> List[str]:
        """Validate requirements for PDF generation.

        Args:
            sheet_data_list: List of sheet data objects

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        if not sheet_data_list:
            warnings.append("No sheet data provided")
            return warnings

        # Check for sheets with no data
        sheets_with_data = 0
        for sheet_data in sheet_data_list:
            if getattr(sheet_data, "has_data", False):
                sheets_with_data += 1

        if sheets_with_data == 0:
            warnings.append("No sheets contain data")
        elif sheets_with_data < len(sheet_data_list):
            warnings.append(
                f"Only {sheets_with_data} of {len(sheet_data_list)} sheets contain data"
            )

        # Check configuration consistency
        if self.config.include_bookmarks and not self.config.optimize_for_notebooklm:
            warnings.append("Bookmarks enabled but NotebookLM optimization disabled")

        if not self.config.include_metadata and self.config.optimize_for_notebooklm:
            warnings.append(
                "NotebookLM optimization enabled but metadata inclusion disabled"
            )

        if warnings:
            logger.warning(
                "PDF generation validation warnings",
                extra={"warning_count": len(warnings), "warnings": warnings},
            )

        return warnings
