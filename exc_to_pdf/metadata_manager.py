"""
PDF metadata management module for AI optimization.

This module provides metadata management functionality for creating
AI-optimized PDF metadata, specifically designed for tools like
NotebookLM to enhance content analysis and understanding.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

import structlog

from .config.pdf_config import PDFConfig
from .exceptions import PDFGenerationException

logger = structlog.get_logger()


class MetadataManager:
    """Manages PDF metadata optimized for AI analysis (NotebookLM)."""

    def __init__(self, config: Optional[PDFConfig] = None) -> None:
        """Initialize metadata manager with configuration.

        Args:
            config: Optional PDF configuration. Uses default if None.
        """
        self.config = config or PDFConfig()

    def create_pdf_metadata(self, sheet_data_list: List[Any],
                          source_file: str) -> Dict[str, Any]:
        """Create comprehensive PDF metadata.

        Args:
            sheet_data_list: List of sheet data objects from P2 processing
            source_file: Original Excel file path

        Returns:
            Dictionary of PDF metadata fields

        Raises:
            PDFGenerationException: If metadata creation fails
        """
        try:
            if not sheet_data_list:
                logger.warning("No sheet data provided for metadata creation")
                sheet_data_list = []

            # Extract basic metadata
            base_metadata = self._extract_base_metadata(source_file)

            # Add content analysis metadata
            content_metadata = self._analyze_content_metadata(sheet_data_list)

            # Add structural metadata
            structural_metadata = self._create_structural_metadata(sheet_data_list)

            # Combine all metadata
            metadata = {**base_metadata, **content_metadata, **structural_metadata}

            # Add AI optimization tags if enabled
            if self.config.optimize_for_notebooklm:
                table_info_list = self._extract_table_info(sheet_data_list)
                metadata = self.add_ai_optimization_tags(metadata, table_info_list)

            logger.info(
                "PDF metadata created successfully",
                extra={
                    "source_file": source_file,
                    "sheet_count": len(sheet_data_list),
                    "ai_optimized": self.config.optimize_for_notebooklm,
                    "metadata_fields": len(metadata)
                }
            )

            return metadata

        except Exception as e:
            logger.error(
                "Failed to create PDF metadata",
                extra={
                    "source_file": source_file,
                    "sheet_count": len(sheet_data_list) if sheet_data_list else 0,
                    "error": str(e)
                }
            )
            raise PDFGenerationException("Failed to create PDF metadata") from e

    def add_ai_optimization_tags(self, metadata: Dict[str, Any],
                               tables: List[Any]) -> Dict[str, Any]:
        """Add AI-optimization tags for NotebookLM compatibility.

        Args:
            metadata: Base metadata dictionary
            tables: List of table information objects

        Returns:
            Enhanced metadata with AI tags

        Raises:
            PDFGenerationException: If AI optimization fails
        """
        try:
            # Add AI-specific tags
            ai_tags = {
                'ai_optimized': True,
                'notebooklm_compatible': True,
                'content_type': 'structured_data',
                'data_format': 'excel_derived',
                'generation_timestamp': datetime.now().isoformat(),
                'analysis_priority': 'high',
                'semantic_structure': 'tabular',
                'language': 'en',
                'encoding': 'utf-8'
            }

            # Add content summary for AI understanding
            if tables:
                ai_tags.update({
                    'table_count': len(tables),
                    'total_data_points': self._estimate_data_points(tables),
                    'data_completeness': self._assess_data_completeness(tables),
                    'content_density': self._calculate_content_density(tables)
                })

            # Add processing information
            ai_tags.update({
                'processing_tool': 'exc-to-pdf v2.2.0',
                'processing_method': 'hybrid_table_detection',
                'quality_score': self._calculate_quality_score(metadata, tables)
            })

            # Add search optimization keywords
            ai_tags['search_keywords'] = self._generate_search_keywords(metadata, tables)

            # Combine with existing metadata
            enhanced_metadata = {**metadata, **ai_tags}

            logger.debug(
                "AI optimization tags added",
                extra={
                    "table_count": len(tables),
                    "optimization_fields": len(ai_tags),
                    "notebooklm_compatible": True
                }
            )

            return enhanced_metadata

        except Exception as e:
            logger.error(
                "Failed to add AI optimization tags",
                extra={
                    "metadata_keys": len(metadata) if metadata else 0,
                    "table_count": len(tables) if tables else 0,
                    "error": str(e)
                }
            )
            raise PDFGenerationException("Failed to add AI optimization tags") from e

    def _extract_base_metadata(self, source_file: str) -> Dict[str, Any]:
        """Extract basic PDF metadata from source file.

        Args:
            source_file: Original Excel file path

        Returns:
            Dictionary with basic metadata fields
        """
        from pathlib import Path

        file_path = Path(source_file)

        return {
            'title': file_path.stem if file_path.name else 'Excel Data Analysis',
            'author': 'exc-to-pdf converter',
            'subject': 'Structured data for AI analysis',
            'creator': 'exc-to-pdf v2.2.0',
            'producer': 'ReportLab PDF Library',
            'creation_date': datetime.now().strftime('%Y%m%d%H%M%S+00\'00\''),
            'mod_date': datetime.now().strftime('%Y%m%d%H%M%S+00\'00\''),
            'source_file': str(file_path.name),
            'source_format': 'Microsoft Excel',
            'conversion_timestamp': datetime.now().isoformat()
        }

    def _analyze_content_metadata(self, sheet_data_list: List[Any]) -> Dict[str, Any]:
        """Analyze sheet data to extract content metadata.

        Args:
            sheet_data_list: List of sheet data objects

        Returns:
            Dictionary with content analysis metadata
        """
        if not sheet_data_list:
            return {
                'total_sheets': 0,
                'total_tables': 0,
                'total_rows': 0,
                'has_data': False,
                'content_summary': 'Empty document'
            }

        total_sheets = len(sheet_data_list)
        total_tables = sum(
            len(sheet.tables) if hasattr(sheet, 'tables') else 0
            for sheet in sheet_data_list
        )
        total_rows = sum(
            sheet.row_count if hasattr(sheet, 'row_count') and isinstance(sheet.row_count, (int, float)) else 0
            for sheet in sheet_data_list
        )
        has_data = any(
            sheet.has_data if hasattr(sheet, 'has_data') else False
            for sheet in sheet_data_list
        )

        # Extract sheet names for metadata
        sheet_names = [
            sheet.sheet_name if hasattr(sheet, 'sheet_name') else f'Sheet_{i+1}'
            for i, sheet in enumerate(sheet_data_list)
        ]

        content_summary = self._generate_content_summary(sheet_data_list)

        return {
            'total_sheets': total_sheets,
            'total_tables': total_tables,
            'total_rows': total_rows,
            'has_data': has_data,
            'sheet_names': sheet_names,
            'content_summary': content_summary,
            'data_structure': 'multi_sheet_tabular'
        }

    def _create_structural_metadata(self, sheet_data_list: List[Any]) -> Dict[str, Any]:
        """Create structural metadata about PDF organization.

        Args:
            sheet_data_list: List of sheet data objects

        Returns:
            Dictionary with structural metadata
        """
        structure_info: Dict[str, Any] = {
            'document_type': 'excel_derived_report',
            'organization': 'sheet_based',
            'navigation': 'bookmarks_enabled' if self.config.include_bookmarks else 'linear',
            'table_rendering': 'modern_styled',
            'page_orientation': self.config.orientation,
            'page_size': self.config.page_size
        }

        if self.config.include_bookmarks:
            structure_info['bookmark_structure'] = 'hierarchical'
            structure_info['bookmark_levels'] = 2  # Sheets + Tables

        return structure_info

    def _extract_table_info(self, sheet_data_list: List[Any]) -> List[Any]:
        """Extract table information from sheet data.

        Args:
            sheet_data_list: List of sheet data objects

        Returns:
            List of table information objects
        """
        tables = []

        for sheet in sheet_data_list:
            if hasattr(sheet, 'tables') and sheet.tables:
                for table in sheet.tables:
                    tables.append(table)
            elif hasattr(sheet, 'raw_data') and sheet.raw_data:
                # Create basic table info from raw data
                tables.append({
                    'name': f"{sheet.sheet_name}_data" if hasattr(sheet, 'sheet_name') else 'data_table',
                    'row_count': len(sheet.raw_data),
                    'col_count': len(sheet.raw_data[0]) if sheet.raw_data and sheet.raw_data[0] else 0
                })

        return tables

    def _estimate_data_points(self, tables: List[Any]) -> int:
        """Estimate total number of data points.

        Args:
            tables: List of table information

        Returns:
            Estimated total data points
        """
        total = 0
        for table in tables:
            if isinstance(table, dict):
                rows = table.get('row_count', 0)
                cols = table.get('col_count', 0)
                # Ensure both are numbers
                if isinstance(rows, (int, float)) and isinstance(cols, (int, float)):
                    total += int(rows) * int(cols)
            elif hasattr(table, 'row_count') and hasattr(table, 'col_count'):
                rows = table.row_count
                cols = table.col_count
                # Ensure both are numbers
                if isinstance(rows, (int, float)) and isinstance(cols, (int, float)):
                    total += int(rows) * int(cols)

        return total

    def _assess_data_completeness(self, tables: List[Any]) -> str:
        """Assess data completeness across tables.

        Args:
            tables: List of table information

        Returns:
            Completeness assessment string
        """
        if not tables:
            return 'empty'

        total_cells = 0
        non_empty_cells = 0

        for table in tables:
            if isinstance(table, dict):
                rows = table.get('row_count', 0)
                cols = table.get('col_count', 0)
                # Ensure both are numbers
                if isinstance(rows, (int, float)) and isinstance(cols, (int, float)):
                    total_cells += int(rows) * int(cols)
                    # Estimate non-empty cells (assume 80% fill rate for Excel data)
                    non_empty_cells += int(rows * cols * 0.8)

        if total_cells == 0:
            return 'empty'

        completeness_ratio = non_empty_cells / total_cells

        if completeness_ratio > 0.9:
            return 'complete'
        elif completeness_ratio > 0.7:
            return 'mostly_complete'
        elif completeness_ratio > 0.5:
            return 'partially_complete'
        else:
            return 'sparse'

    def _calculate_content_density(self, tables: List[Any]) -> str:
        """Calculate content density rating.

        Args:
            tables: List of table information

        Returns:
            Content density string
        """
        total_data_points = self._estimate_data_points(tables)

        if total_data_points > 10000:
            return 'very_high'
        elif total_data_points > 5000:
            return 'high'
        elif total_data_points > 1000:
            return 'medium'
        elif total_data_points > 100:
            return 'low'
        else:
            return 'very_low'

    def _calculate_quality_score(self, metadata: Dict[str, Any], tables: List[Any]) -> float:
        """Calculate overall quality score for the data.

        Args:
            metadata: Current metadata
            tables: List of table information

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.5  # Base score

        # Add points for having data
        if metadata.get('has_data', False):
            score += 0.2

        # Add points for table structure
        if tables and len(tables) > 0:
            score += 0.1

        # Add points for multiple sheets
        if metadata.get('total_sheets', 0) > 1:
            score += 0.1

        # Add points for data completeness
        completeness = self._assess_data_completeness(tables)
        if completeness == 'complete':
            score += 0.1
        elif completeness == 'mostly_complete':
            score += 0.05

        return min(1.0, score)

    def _generate_search_keywords(self, metadata: Dict[str, Any], tables: List[Any]) -> List[str]:
        """Generate search keywords for AI systems.

        Args:
            metadata: Current metadata
            tables: List of table information

        Returns:
            List of search keywords
        """
        keywords = [
            'excel', 'table', 'data', 'analysis', 'report',
            'spreadsheet', 'structured', 'tabular'
        ]

        # Add content-specific keywords
        if metadata.get('total_sheets', 0) > 1:
            keywords.append('multi-sheet')

        if metadata.get('total_tables', 0) > 1:
            keywords.append('multi-table')

        # Add density-specific keywords
        density = self._calculate_content_density(tables)
        if density in ['high', 'very_high']:
            keywords.append('comprehensive')
            keywords.append('detailed')

        # Add completeness keywords
        completeness = self._assess_data_completeness(tables)
        if completeness in ['complete', 'mostly_complete']:
            keywords.append('complete_data')

        # Add tool-specific keywords
        keywords.extend(['notebooklm', 'ai_analysis', 'data_visualization'])

        return list(set(keywords))  # Remove duplicates

    def _generate_content_summary(self, sheet_data_list: List[Any]) -> str:
        """Generate a content summary for the document.

        Args:
            sheet_data_list: List of sheet data objects

        Returns:
            Content summary string
        """
        if not sheet_data_list:
            return 'Empty Excel document converted to PDF'

        sheet_count = len(sheet_data_list)
        table_count = sum(
            len(sheet.tables) if hasattr(sheet, 'tables') else 0
            for sheet in sheet_data_list
        )
        total_rows = sum(
            sheet.row_count if hasattr(sheet, 'row_count') and isinstance(sheet.row_count, (int, float)) else 0
            for sheet in sheet_data_list
        )

        summary_parts = []
        summary_parts.append(f"Excel document with {sheet_count} worksheet{'s' if sheet_count != 1 else ''}")

        if table_count > 0:
            summary_parts.append(f"{table_count} table{'s' if table_count != 1 else ''}")

        if total_rows > 0:
            summary_parts.append(f"{total_rows} total rows")

        # Add sheet names if available
        sheet_names = [
            sheet.sheet_name if hasattr(sheet, 'sheet_name') else f'Sheet_{i+1}'
            for i, sheet in enumerate(sheet_data_list)
        ]

        if sheet_names and len(sheet_names) <= 5:
            summary_parts.append(f"worksheets: {', '.join(sheet_names)}")
        elif sheet_names:
            summary_parts.append(f"worksheets: {', '.join(sheet_names[:3])} and {len(sheet_names) - 3} more")

        return ". ".join(summary_parts) + "."

    def get_metadata_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of metadata for logging/debugging.

        Args:
            metadata: Complete metadata dictionary

        Returns:
            Summary dictionary with key fields
        """
        return {
            'title': metadata.get('title', 'Unknown'),
            'total_sheets': metadata.get('total_sheets', 0),
            'total_tables': metadata.get('total_tables', 0),
            'total_rows': metadata.get('total_rows', 0),
            'has_data': metadata.get('has_data', False),
            'ai_optimized': metadata.get('ai_optimized', False),
            'notebooklm_compatible': metadata.get('notebooklm_compatible', False),
            'content_density': metadata.get('content_density', 'unknown'),
            'data_completeness': metadata.get('data_completeness', 'unknown'),
            'quality_score': metadata.get('quality_score', 0.0),
            'keyword_count': len(metadata.get('search_keywords', []))
        }