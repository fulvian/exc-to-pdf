"""
PDF table rendering module with modern styling and performance optimization.

This module provides specialized table rendering capabilities for creating
professional-looking tables in PDF documents using ReportLab with modern
styling patterns and performance optimization for large datasets.
"""

from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Table, TableStyle, KeepTogether, LongTable
from typing import Union

import structlog

from .config.pdf_config import PDFConfig
from .exceptions import TableRenderingException

logger = structlog.get_logger()


class PDFTableRenderer:
    """Specialized PDF table rendering with modern styling and performance optimization."""

    def __init__(self, config: Optional[PDFConfig] = None) -> None:
        """Initialize PDF table renderer with configuration.

        Args:
            config: Optional PDF configuration. Uses default if None.

        Raises:
            ConfigurationException: If configuration validation fails
        """
        self.config = config or PDFConfig()
        self.page_width = A4[0] - (self.config.margin_left + self.config.margin_right)

        # Color palette based on configuration
        self.colors = {
            "header_bg": colors.HexColor(self.config.header_background),
            "header_text": colors.HexColor(self.config.header_text_color),
            "row_even": (
                colors.whitesmoke if self.config.alternate_rows else colors.white
            ),
            "row_odd": colors.white,
            "alternate_row": (
                colors.HexColor(self.config.alternate_row_color)
                if self.config.alternate_rows
                else None
            ),
            "border": colors.HexColor("#CCCCCC"),
            "grid": colors.HexColor("#CCCCCC"),
            "text_primary": colors.HexColor("#212529"),
            "text_secondary": colors.HexColor("#6C757D"),
        }

    def render_table(
        self,
        table_data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
    ) -> Union[Table, LongTable]:
        """Render data as ReportLab Table with modern styling.

        Args:
            table_data: Table data rows (excluding headers)
            headers: Column headers
            title: Optional table title

        Returns:
            Formatted ReportLab Table object (Table or LongTable for large tables)

        Raises:
            TableRenderingException: If table rendering fails
        """
        try:
            if not table_data and not headers:
                raise ValueError("Both table_data and headers cannot be empty")

            # Check if table is too large and needs splitting
            max_rows = self.config.max_table_rows_per_page
            total_rows = len(table_data)
            expected_rows = total_rows + (1 if headers else 0)

            if expected_rows > max_rows:
                logger.debug(
                    "Table too large, splitting across pages",
                    extra={
                        "total_rows": expected_rows,
                        "max_rows": max_rows,
                        "chunk_count": (expected_rows + max_rows - 1) // max_rows,
                    },
                )

                if self.config.enable_table_splitting:
                    # Create LongTable for large tables when splitting is enabled
                    return self._create_long_table(table_data, headers, title)
                else:
                    # Return first chunk as a single table for now (splitting disabled)
                    chunk_size = max_rows - 1 if headers else max_rows
                    chunk = table_data[:chunk_size]
                    return self._create_single_table(chunk, headers, title)
            else:
                return self._create_single_table(table_data, headers, title)

        except Exception as e:
            logger.error(
                "Table rendering failed",
                extra={
                    "data_rows": len(table_data) if table_data else 0,
                    "headers": len(headers) if headers else 0,
                    "error": str(e),
                },
            )
            raise TableRenderingException("Failed to render table") from e

    def _create_single_table(
        self,
        table_data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
    ) -> Table:
        """Create a single table with the given data.

        Args:
            table_data: Table data (list of lists)
            headers: Column headers
            title: Optional table title

        Returns:
            Formatted ReportLab Table object
        """
        # Prepare full table data with headers
        if headers:
            full_data = [headers] + table_data
        else:
            full_data = table_data

        # Calculate column widths
        col_widths = self.calculate_column_widths(table_data, headers, self.page_width)

        # Create table
        table = Table(full_data, colWidths=col_widths, repeatRows=1)

        # Apply styling
        table.setStyle(
            self._create_table_style(
                header_rows=1 if headers else 0, is_long_table=False
            )
        )

        logger.info(
            "Table rendered successfully",
            extra={
                "rows": len(table_data),
                "columns": len(headers) if headers else 0,
                "has_title": title is not None,
                "is_long_table": False,
            },
        )

        return table

    def _create_long_table(
        self,
        table_data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
    ) -> LongTable:
        """Create a LongTable for large tables that can span multiple pages.

        Args:
            table_data: Table data (list of lists)
            headers: Column headers
            title: Optional table title

        Returns:
            Formatted ReportLab LongTable object
        """
        # Prepare full table data with headers
        if headers:
            full_data = [headers] + table_data
        else:
            full_data = table_data

        # Calculate column widths
        col_widths = self.calculate_column_widths(table_data, headers, self.page_width)

        # Create LongTable
        long_table = LongTable(full_data, colWidths=col_widths)

        # Apply styling with long table flag
        long_table.setStyle(
            self._create_table_style(
                header_rows=1 if headers else 0, is_long_table=True
            )
        )

        logger.info(
            "Table rendered successfully",
            extra={
                "rows": len(table_data),
                "columns": len(headers) if headers else 0,
                "has_title": title is not None,
                "is_long_table": True,
            },
        )

        return long_table

    def handle_large_table(
        self, data: List[List[Any]], headers: List[str]
    ) -> List[Table]:
        """Split large tables across multiple pages.

        Args:
            data: Complete table data
            headers: Column headers

        Returns:
            List of Table objects, one per page

        Raises:
            TableRenderingException: If table splitting fails
        """
        try:
            if not data and not headers:
                raise ValueError("Both data and headers cannot be empty")

            tables = []
            max_rows = self.config.max_table_rows_per_page
            has_headers = bool(headers)

            # Calculate chunk size (account for header row)
            chunk_size = max_rows - 1 if has_headers else max_rows

            if chunk_size <= 0:
                raise ValueError(f"max_table_rows_per_page too small: {max_rows}")

            # Split data into chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]

                # Create table for this chunk
                if i == 0 and has_headers:
                    # First table includes headers
                    chunk_table = self.render_table(chunk, headers)
                else:
                    # Subsequent tables may or may not include headers based on config
                    if has_headers:
                        chunk_table = self.render_table(chunk, headers)
                    else:
                        chunk_table = self.render_table(chunk, [])

                tables.append(chunk_table)

            logger.info(
                "Large table split successfully",
                extra={
                    "total_rows": len(data),
                    "chunks": len(tables),
                    "max_rows_per_page": max_rows,
                },
            )

            return tables

        except Exception as e:
            logger.error(
                "Table splitting failed",
                extra={
                    "data_rows": len(data) if data else 0,
                    "chunk_size": chunk_size if "chunk_size" in locals() else 0,
                    "error": str(e),
                },
            )
            raise TableRenderingException("Failed to split large table") from e

    def calculate_column_widths(
        self, data: List[List[Any]], headers: List[str], page_width: float
    ) -> List[float]:
        """Calculate optimal column widths based on content and page width.

        Args:
            data: Table data for content analysis
            headers: Column headers
            page_width: Available page width

        Returns:
            List of column widths in points

        Raises:
            TableRenderingException: If width calculation fails
        """
        try:
            if not headers and not data:
                return [page_width]  # Single column if no headers or data

            num_cols = len(headers) if headers else (len(data[0]) if data else 1)
            if num_cols == 0:
                return []

            # Calculate content-based widths
            content_widths: List[float] = [0.0] * num_cols
            font_name = "Helvetica"
            font_size = self.config.font_size
            header_font_size = self.config.header_font_size
            padding = 12  # Total padding per column

            # Analyze header widths
            if headers:
                for col_idx, header in enumerate(headers):
                    if col_idx < num_cols:
                        header_width = stringWidth(
                            str(header), font_name, header_font_size
                        )
                        content_widths[col_idx] = max(
                            content_widths[col_idx], header_width
                        )

            # Analyze data content widths (sample first 100 rows for performance)
            sample_size = min(100, len(data))
            for row in data[:sample_size]:
                for col_idx, cell_content in enumerate(row):
                    if col_idx < num_cols:
                        content_width = stringWidth(
                            str(cell_content), font_name, font_size
                        )
                        content_widths[col_idx] = max(
                            content_widths[col_idx], content_width
                        )

            # Add padding to content widths
            padded_widths = [width + padding for width in content_widths]

            # Sanitize and format cell content for better rendering
            sanitized_data = []
            for row in data:
                sanitized_row = []
                for cell_content in row:
                    if cell_content is None:
                        sanitized_row.append("")
                    else:
                        # Clean and format cell content
                        clean_content = str(cell_content).strip()

                        # Handle very long content that might cause rendering issues
                        if len(clean_content) > 200:  # Characters longer than 200 might indicate malformed content
                            # Truncate and add indicator
                            clean_content = clean_content[:200] + "..."

                        # Handle content with newlines that should be preserved
                        if '\n' in clean_content and len(clean_content) < 1000:
                            # Keep reasonable newlines, replace excessive ones
                            clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)

                        sanitized_row.append(clean_content)
                sanitized_data.append(sanitized_row)

            # Apply minimum and maximum width constraints
            min_width = 0.5 * inch  # 36 points minimum
            max_width = 3 * inch  # 216 points maximum

            constrained_widths: List[float] = []
            for width in padded_widths:
                constrained_width = max(min_width, min(max_width, width))
                constrained_widths.append(constrained_width)

            # Adjust to fit total page width
            total_width = sum(constrained_widths)

            if total_width <= page_width:
                # Table fits, distribute extra space proportionally
                extra_space = page_width - total_width
                if extra_space > 0 and num_cols > 0:
                    extra_per_col = extra_space / num_cols
                    constrained_widths = [w + extra_per_col for w in constrained_widths]
            else:
                # Table too wide, scale down proportionally
                scale_factor = page_width / total_width
                constrained_widths = [w * scale_factor for w in constrained_widths]

            # Ensure final widths sum to page_width (account for floating point precision)
            final_total = sum(constrained_widths)
            if abs(final_total - page_width) > 0.1:  # More than 0.1 point difference
                if num_cols > 0:
                    constrained_widths[-1] = constrained_widths[-1] + (
                        page_width - final_total
                    )

            logger.debug(
                "Column widths calculated",
                extra={
                    "columns": num_cols,
                    "page_width": page_width,
                    "total_width": sum(constrained_widths),
                    "average_width": (
                        sum(constrained_widths) / num_cols if num_cols > 0 else 0
                    ),
                },
            )

            return constrained_widths

        except Exception as e:
            logger.error(
                "Column width calculation failed",
                extra={
                    "data_rows": len(data) if data else 0,
                    "headers": len(headers) if headers else 0,
                    "page_width": page_width,
                    "error": str(e),
                },
            )
            raise TableRenderingException("Failed to calculate column widths") from e

    def _create_table_style(
        self, header_rows: int = 0, is_long_table: bool = False
    ) -> TableStyle:
        """Create modern table styling based on configuration.

        Args:
            header_rows: Number of header rows in the table
            is_long_table: Whether this table will be split across pages (affects styling choices)

        Returns:
            Configured TableStyle object
        """
        style_elements = []

        # Header styling
        if header_rows > 0:
            style_elements.extend(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, header_rows - 1),
                        self.colors["header_bg"],
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, header_rows - 1),
                        self.colors["header_text"],
                    ),
                    ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, header_rows - 1),
                        self.config.header_font_size,
                    ),
                    ("ALIGN", (0, 0), (-1, header_rows - 1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, header_rows - 1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, header_rows - 1), 12),
                    ("TOPPADDING", (0, 0), (-1, header_rows - 1), 12),
                ]
            )

        # Data rows styling
        if header_rows > 0:
            data_start_row = header_rows
            style_elements.extend(
                [
                    ("FONTNAME", (0, data_start_row), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, data_start_row), (-1, -1), self.config.font_size),
                    ("ALIGN", (0, data_start_row), (-1, -1), "CENTER"),
                    ("VALIGN", (0, data_start_row), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, data_start_row), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, data_start_row), (-1, -1), 8),
                ]
            )

            # Alternating row colors - AVOID ROWBACKGROUNDS for LongTable due to ReportLab bug
            if self.config.alternate_rows:
                if is_long_table:
                    # For LongTable, use simple background color to avoid rowpositions bug
                    style_elements.append(
                        (
                            "BACKGROUND",
                            (0, data_start_row),
                            (-1, -1),
                            self.colors["row_even"],
                        )
                    )
                else:
                    # For regular tables, ROWBACKGROUNDS is safe and more efficient
                    style_elements.append(
                        (
                            "ROWBACKGROUNDS",
                            (0, data_start_row),
                            (-1, -1),
                            [self.colors["row_even"], self.colors["alternate_row"]],
                        )  # type: ignore[arg-type]
                    )
            else:
                style_elements.append(
                    (
                        "BACKGROUND",
                        (0, data_start_row),
                        (-1, -1),
                        self.colors["row_even"],
                    )
                )
        else:
            # No headers - style all rows as data
            style_elements.extend(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), self.config.font_size),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors["row_even"]),
                ]
            )

            # Alternating row colors for no-header tables - AVOID ROWBACKGROUNDS for LongTable
            if self.config.alternate_rows:
                if is_long_table:
                    # For LongTable, use simple background to avoid bug
                    style_elements.append(
                        ("BACKGROUND", (0, 0), (-1, -1), self.colors["row_even"])
                    )
                else:
                    # For regular tables, ROWBACKGROUNDS is safe
                    style_elements.append(
                        (
                            "ROWBACKGROUNDS",
                            (0, 0),
                            (-1, -1),
                            [self.colors["row_even"], self.colors["alternate_row"]],
                        )  # type: ignore[arg-type]
                    )

        # Grid and borders
        style_elements.extend(
            [
                ("GRID", (0, 0), (-1, -1), 1, self.colors["grid"]),  # type: ignore
                ("BOX", (0, 0), (-1, -1), 2, self.colors["border"]),  # type: ignore
            ]
        )

        # Thicker line under headers if present
        if header_rows > 0:
            style_elements.append(
                ("LINEBELOW", (0, 0), (-1, header_rows - 1), 2, self.colors["header_bg"])  # type: ignore
            )

        return TableStyle(style_elements)

    def create_wrapped_table(
        self,
        table_data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
    ) -> KeepTogether:
        """Create a wrapped table that won't be split across pages.

        Args:
            table_data: Table data rows (excluding headers)
            headers: Column headers
            title: Optional table title

        Returns:
            KeepTogether flowable containing the table
        """
        table = self.render_table(table_data, headers, title)
        return KeepTogether([table])

    def _is_problematic_data(
        self, table_data: List[List[Any]]
    ) -> bool:
        """Check if table data appears problematic for rendering.

        Args:
            table_data: Table data to analyze

        Returns:
            True if data appears problematic
        """
        if not table_data:
            return False

        # Check for excessively long text content
        max_cell_length = 200
        for row in table_data:
            for cell in row:
                if cell and len(str(cell)) > max_cell_length:
                    return True
        return False

    def _create_info_table(
        self,
        table_data: List[List[Any]],
        title: Optional[str] = None,
    ) -> Table:
        """Create a simple info table instead of failing completely.

        Args:
            table_data: Table data
            title: Optional table title

        Returns:
            Simple Table with info about data issues
        """
        # Check if table has too many rows and needs splitting
            max_rows = self.config.max_table_rows_per_page
            if len(table_data) > max_rows:
                logger.debug(
                    "Table too large, splitting across pages",
                    extra={
                        "total_rows": len(table_data),
                        "max_rows": max_rows,
                        "chunk_count": (len(table_data) + max_rows - 1) // max_rows,
                    },
                )
                # Split data into chunks for processing
                chunks = []
                for i in range(0, len(table_data), max_rows):
                    chunk = table_data[i : i + max_rows]
                    chunks.append(chunk)
                return self.handle_large_table(chunks, headers, title)
            else:
                return self.create_wrapped_table(table_data, headers, title)

            # Create simple table with basic info
            table = Table(limited_data)
            table.setStyle([
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])

            if title:
                table.setStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ])

            logger.warning("Created safe info table instead of failing")
            return table

    def get_table_info(
        self, table_data: List[List[Any]], headers: List[str]
    ) -> Dict[str, Any]:
        """Get information about a table for optimization decisions.

        Args:
            table_data: Table data rows
            headers: Column headers

        Returns:
            Dictionary containing table metadata
        """
        return {
            "row_count": len(table_data),
            "column_count": (
                len(headers) if headers else (len(table_data[0]) if table_data else 0)
            ),
            "estimated_width": sum(
                self.calculate_column_widths(table_data, headers, self.page_width)
            ),
            "requires_splitting": len(table_data) > self.config.max_table_rows_per_page,
            "has_headers": bool(headers),
            "estimated_pages": max(
                1,
                (len(table_data) + self.config.max_table_rows_per_page - 1)
                // self.config.max_table_rows_per_page,
            ),
        }
