"""
PDF table rendering module with modern styling and performance optimization.

This module provides specialized table rendering capabilities for creating
professional-looking tables in PDF documents using ReportLab with modern
styling patterns and performance optimization for large datasets.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import KeepTogether, LongTable, Paragraph, Table, TableStyle
from xml.sax.saxutils import escape

import structlog

from .config.pdf_config import PDFConfig
from .exceptions import TableRenderingException

logger = structlog.get_logger()


@dataclass
class TableContent:
    """Prepared table content ready for width calculation and rendering."""

    measurement_headers: List[str]
    display_headers: List[str]
    measurement_rows: List[List[str]]
    display_rows: List[List[str]]
    numeric_columns: List[bool]
    num_columns: int


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
        self.page_width = max(self.config.get_available_width(), 1.0)
        self.min_column_width = max(0.45 * inch, self.config.font_size * 1.8)
        self.max_column_width = 3.2 * inch
        self.column_padding = 12.0

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

        # Pre-compute paragraph styles for consistent rendering
        self.header_style = ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=self.config.header_font_size,
            alignment=TA_CENTER,
            leading=self.config.header_font_size + 2,
        )
        self.data_style_left = ParagraphStyle(
            name="TableCellLeft",
            fontName="Helvetica",
            fontSize=self.config.font_size,
            alignment=TA_LEFT,
            leading=self.config.font_size + 2,
        )
        self.data_style_right = ParagraphStyle(
            name="TableCellRight",
            parent=self.data_style_left,
            alignment=TA_RIGHT,
        )

    def render_table(
        self,
        table_data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
    ) -> Union[Table, LongTable, List[Union[Table, LongTable]]]:
        """Render data as ReportLab Table with modern styling.

        Args:
            table_data: Table data rows (excluding headers)
            headers: Column headers
            title: Optional table title

        Returns:
            Formatted ReportLab Table object or list of tables when column chunking is required.

        Raises:
            TableRenderingException: If table rendering fails
        """
        try:
            if not table_data and not headers:
                raise ValueError("Both table_data and headers cannot be empty")

            content = self._prepare_table_content(table_data, headers)
            if content.num_columns == 0:
                raise ValueError("No columns detected for rendering")

            page_width = max(self.config.get_available_width(), 1.0)
            natural_widths = self._calculate_natural_widths(
                content.measurement_rows, content.measurement_headers
            )
            column_chunks = self._chunk_columns(natural_widths, page_width)

            header_paragraphs: List[Paragraph] = []
            if headers:
                header_paragraphs = [
                    Paragraph(text or "&nbsp;", self.header_style)
                    for text in content.display_headers
                ]

            paragraph_rows: List[List[Paragraph]] = []
            for row in content.display_rows:
                paragraph_row = []
                for col_idx, text in enumerate(row):
                    style = (
                        self.data_style_right
                        if content.numeric_columns[col_idx]
                        else self.data_style_left
                    )
                    paragraph_row.append(Paragraph(text or "&nbsp;", style))
                paragraph_rows.append(paragraph_row)

            flowables: List[Union[Table, LongTable]] = []
            for start, end in column_chunks:
                chunk_headers = header_paragraphs[start:end] if headers else []
                chunk_rows = [row[start:end] for row in paragraph_rows]
                chunk_widths = self._fit_chunk_widths(
                    natural_widths[start:end], page_width
                )

                flowable = self._build_table_flowable(
                    chunk_headers, chunk_rows, chunk_widths, bool(chunk_headers)
                )
                flowables.append(flowable)

            logger.info(
                "Table rendered successfully",
                extra={
                    "rows": len(table_data),
                    "columns": content.num_columns,
                    "has_title": title is not None,
                    "is_long_table": any(isinstance(f, LongTable) for f in flowables),
                    "chunks": len(flowables),
                },
            )

            if len(flowables) == 1:
                return flowables[0]
            return flowables

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

    def handle_large_table(
        self, data: List[List[Any]], headers: List[str]
    ) -> List[Table]:
        """Split large tables across multiple pages.

        Args:
            data: Complete table data
            headers: Column headers

        Returns:
            List of table flowables, one per chunk

        Raises:
            TableRenderingException: If table splitting fails
        """
        try:
            if not data and not headers:
                raise ValueError("Both data and headers cannot be empty")

            tables: List[Union[Table, LongTable]] = []
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

                if isinstance(chunk_table, list):
                    tables.extend(chunk_table)
                else:
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
            content = self._prepare_table_content(data, headers)
            if content.num_columns == 0:
                return [page_width]

            resolved_page_width = max(page_width, 1.0)
            natural_widths = self._calculate_natural_widths(
                content.measurement_rows, content.measurement_headers
            )
            adjusted_widths = self._fit_widths_to_page(natural_widths, resolved_page_width)

            logger.debug(
                "Column widths calculated",
                extra={
                    "columns": content.num_columns,
                    "page_width": resolved_page_width,
                    "total_width": sum(adjusted_widths),
                    "average_width": (
                        sum(adjusted_widths) / content.num_columns
                        if content.num_columns > 0
                        else 0
                    ),
                },
            )

            return adjusted_widths

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

    def _prepare_table_content(
        self, data: List[List[Any]], headers: List[str]
    ) -> TableContent:
        """Normalize and sanitize table content for width calculation and rendering."""
        num_cols = max(len(headers), max((len(row) for row in data), default=0))

        measurement_headers: List[str] = []
        display_headers: List[str] = []

        if num_cols == 0 and data:
            num_cols = len(data[0])

        if num_cols == 0:
            return TableContent(
                measurement_headers=[],
                display_headers=[],
                measurement_rows=[],
                display_rows=[],
                numeric_columns=[],
                num_columns=0,
            )

        if headers:
            for idx in range(num_cols):
                header_value = headers[idx] if idx < len(headers) else f"Column {idx+1}"
                measurement, display = self._normalize_cell(header_value)
                measurement_headers.append(measurement)
                display_headers.append(display)
        else:
            measurement_headers = ["" for _ in range(num_cols)]
            display_headers = ["" for _ in range(num_cols)]

        numeric_columns: List[bool] = [True] * num_cols
        measurement_rows: List[List[str]] = []
        display_rows: List[List[str]] = []

        for row in data:
            row_list = list(row) if isinstance(row, (list, tuple)) else [row]
            measurement_row: List[str] = []
            display_row: List[str] = []

            for idx in range(num_cols):
                value = row_list[idx] if idx < len(row_list) else ""
                measurement_text, display_text = self._normalize_cell(value)
                measurement_row.append(measurement_text)
                display_row.append(display_text)

                if self._has_content(value) and not self._is_numeric(value):
                    numeric_columns[idx] = False

            measurement_rows.append(measurement_row)
            display_rows.append(display_row)

        return TableContent(
            measurement_headers=measurement_headers,
            display_headers=display_headers,
            measurement_rows=measurement_rows,
            display_rows=display_rows,
            numeric_columns=numeric_columns,
            num_columns=num_cols,
        )

    def _calculate_natural_widths(
        self, data: List[List[str]], headers: List[str]
    ) -> List[float]:
        """Calculate natural column widths before fitting to the page."""
        num_cols = max(len(headers), max((len(row) for row in data), default=0))
        if num_cols == 0:
            return []

        content_widths: List[float] = [0.0] * num_cols

        header_font_name = "Helvetica-Bold"
        data_font_name = "Helvetica"
        header_font_size = self.config.header_font_size
        data_font_size = self.config.font_size

        for idx in range(min(len(headers), num_cols)):
            header_text = headers[idx] or ""
            header_width = stringWidth(header_text, header_font_name, header_font_size)
            content_widths[idx] = max(content_widths[idx], header_width)

        sample_size = min(200, len(data))
        for row in data[:sample_size]:
            for idx in range(min(len(row), num_cols)):
                cell_text = row[idx] or ""
                cell_width = stringWidth(cell_text, data_font_name, data_font_size)
                content_widths[idx] = max(content_widths[idx], cell_width)

        widths: List[float] = []
        for width in content_widths:
            padded = width + self.column_padding
            constrained = max(self.min_column_width, min(self.max_column_width, padded))
            widths.append(constrained)

        return widths

    def _chunk_columns(
        self, widths: List[float], page_width: float
    ) -> List[Tuple[int, int]]:
        """Split columns into chunks that fit within the page width."""
        if not widths:
            return []

        chunks: List[Tuple[int, int]] = []
        start = 0
        running_width = 0.0

        adjusted_page_width = max(page_width, self.min_column_width)

        for idx, width in enumerate(widths):
            column_width = max(width, self.min_column_width)

            if column_width >= adjusted_page_width:
                if running_width > 0:
                    chunks.append((start, idx))
                chunks.append((idx, idx + 1))
                start = idx + 1
                running_width = 0.0
                continue

            if running_width == 0.0:
                running_width = column_width
                continue

            if running_width + column_width > adjusted_page_width:
                chunks.append((start, idx))
                start = idx
                running_width = column_width
            else:
                running_width += column_width

        if start < len(widths):
            chunks.append((start, len(widths)))

        return chunks

    def _fit_chunk_widths(
        self, widths: List[float], page_width: float
    ) -> List[float]:
        """Scale or expand column widths to fill the available page width."""
        if not widths:
            return []

        adjusted_page_width = max(page_width, self.min_column_width)
        total_width = sum(widths)

        if total_width <= 0:
            even_width = adjusted_page_width / len(widths)
            return [even_width for _ in widths]

        if total_width < adjusted_page_width:
            extra = (adjusted_page_width - total_width) / len(widths)
            adjusted = [w + extra for w in widths]
        else:
            scale = adjusted_page_width / total_width
            adjusted = [w * scale for w in widths]

        diff = adjusted_page_width - sum(adjusted)
        if abs(diff) > 0.1:
            adjusted[-1] += diff

        return adjusted

    def _fit_widths_to_page(
        self, widths: List[float], page_width: float
    ) -> List[float]:
        """Public helper to align widths with page width."""
        adjusted_page_width = max(page_width, self.min_column_width)
        return self._fit_chunk_widths(widths, adjusted_page_width)

    def _build_table_flowable(
        self,
        headers: List[Paragraph],
        rows: List[List[Paragraph]],
        column_widths: List[float],
        has_headers: bool,
    ) -> Union[Table, LongTable]:
        """Create a table or long table flowable from prepared content."""
        header_rows = 1 if has_headers else 0
        data_row_count = len(rows)
        expected_rows = data_row_count + header_rows
        use_long_table = (
            self.config.enable_table_splitting
            and expected_rows > self.config.max_table_rows_per_page
        )

        full_data: List[List[Paragraph]] = []
        if has_headers:
            full_data.append(headers)
        full_data.extend(rows)

        table_cls = LongTable if use_long_table else Table
        table = table_cls(
            full_data,
            colWidths=column_widths,
            repeatRows=header_rows if has_headers else 0,
        )
        table.hAlign = "LEFT"
        table.setStyle(
            self._create_table_style(
                header_rows=header_rows,
                is_long_table=use_long_table,
            )
        )
        return table

    def _normalize_cell(self, value: Any) -> Tuple[str, str]:
        """Normalize a cell value for width measurement and rendering."""
        if value is None:
            return "", ""

        text = str(value)
        text = text.strip()

        if not text:
            return "", ""

        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        measurement_text = re.sub(r"\s+", " ", text).replace("\n", " ").strip()
        display_text = escape(text).replace("\n", "<br/>")

        return measurement_text, display_text

    def _is_numeric(self, value: Any) -> bool:
        """Determine if a value should be treated as numeric for alignment."""
        if value is None or isinstance(value, bool):
            return False
        return isinstance(value, (int, float, Decimal))

    def _has_content(self, value: Any) -> bool:
        """Check if a cell value carries meaningful content."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

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
                    ("ALIGN", (0, data_start_row), (-1, -1), "LEFT"),
                    ("VALIGN", (0, data_start_row), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, data_start_row), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, data_start_row), (-1, -1), 8),
                    ("LEFTPADDING", (0, data_start_row), (-1, -1), 6),
                    ("RIGHTPADDING", (0, data_start_row), (-1, -1), 6),
                    ("WORDWRAP", (0, data_start_row), (-1, -1), "CJK"),
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
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
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
        rendered = self.render_table(table_data, headers, title)
        if isinstance(rendered, list):
            return KeepTogether(rendered)
        return KeepTogether([rendered])

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
        content = self._prepare_table_content(table_data, headers)
        page_width = max(self.config.get_available_width(), 1.0)
        natural_widths = self._calculate_natural_widths(
            content.measurement_rows, content.measurement_headers
        )
        column_chunks = self._chunk_columns(natural_widths, page_width)

        if column_chunks:
            estimated_chunk_widths = [
                sum(natural_widths[start:end]) for start, end in column_chunks
            ]
            estimated_width = max(estimated_chunk_widths)
        else:
            estimated_width = sum(natural_widths)

        max_rows = self.config.max_table_rows_per_page
        row_chunks = max(1, (len(table_data) + max_rows - 1) // max_rows)
        column_chunk_count = max(1, len(column_chunks) or 1)

        return {
            "row_count": len(table_data),
            "column_count": content.num_columns,
            "estimated_width": estimated_width,
            "requires_splitting": len(table_data) > max_rows,
            "has_headers": bool(headers),
            "estimated_pages": row_chunks * column_chunk_count,
            "requires_column_chunking": column_chunk_count > 1,
        }
