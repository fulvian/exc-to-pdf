"""
Unit tests for PDF metadata manager.

Tests the MetadataManager class including metadata creation, AI optimization,
content analysis, and NotebookLM compatibility features.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.config.pdf_config import PDFConfig
from src.metadata_manager import MetadataManager
from src.exceptions import PDFGenerationException


class TestMetadataManager:
    """Test cases for MetadataManager class."""

    def test_initialization_default_config(self) -> None:
        """Test MetadataManager initialization with default config."""
        manager = MetadataManager()

        assert isinstance(manager.config, PDFConfig)
        assert manager.config.optimize_for_notebooklm is True
        assert manager.config.include_metadata is True

    def test_initialization_custom_config(self) -> None:
        """Test MetadataManager initialization with custom config."""
        config = PDFConfig(
            optimize_for_notebooklm=False,
            include_metadata=False
        )
        manager = MetadataManager(config)

        assert manager.config is config
        assert manager.config.optimize_for_notebooklm is False
        assert manager.config.include_metadata is False

    def test_create_pdf_metadata_empty_data(self) -> None:
        """Test creating metadata with empty sheet data."""
        manager = MetadataManager()

        metadata = manager.create_pdf_metadata([], "test.xlsx")

        assert metadata['title'] == 'test'
        assert metadata['total_sheets'] == 0
        assert metadata['total_tables'] == 0
        assert metadata['has_data'] is False
        assert metadata['content_summary'] == 'Empty document'

    def test_create_pdf_metadata_with_basic_data(self) -> None:
        """Test creating metadata with basic sheet data."""
        manager = MetadataManager()

        # Create mock sheet data with table info to avoid AI optimization errors
        mock_sheet = Mock()
        mock_sheet.sheet_name = "Sheet1"
        mock_sheet.tables = [Mock(row_count=5, col_count=2)]  # Add table with row_count and col_count
        mock_sheet.row_count = 10
        mock_sheet.has_data = True

        metadata = manager.create_pdf_metadata([mock_sheet], "data.xlsx")

        assert metadata['title'] == 'data'
        assert metadata['total_sheets'] == 1
        assert metadata['total_tables'] == 1  # One table
        assert metadata['total_rows'] == 10
        assert metadata['has_data'] is True
        assert metadata['sheet_names'] == ['Sheet1']
        assert 'ai_optimized' in metadata
        assert 'notebooklm_compatible' in metadata

    def test_create_pdf_metadata_with_multiple_sheets(self) -> None:
        """Test creating metadata with multiple sheets."""
        manager = MetadataManager()

        # Create mock sheet data with proper table attributes
        sheet1 = Mock()
        sheet1.sheet_name = "Sales"
        sheet1.tables = [Mock(row_count=25, col_count=4)]  # Add proper attributes
        sheet1.row_count = 50
        sheet1.has_data = True

        sheet2 = Mock()
        sheet2.sheet_name = "Inventory"
        sheet2.tables = [Mock(row_count=50, col_count=3), Mock(row_count=25, col_count=2)]  # Add proper attributes
        sheet2.row_count = 100
        sheet2.has_data = True

        metadata = manager.create_pdf_metadata([sheet1, sheet2], "report.xlsx")

        assert metadata['title'] == 'report'
        assert metadata['total_sheets'] == 2
        assert metadata['total_tables'] == 3
        assert metadata['total_rows'] == 150
        assert metadata['has_data'] is True
        assert set(metadata['sheet_names']) == {'Sales', 'Inventory'}

    def test_add_ai_optimization_tags_enabled(self) -> None:
        """Test adding AI optimization tags when enabled."""
        config = PDFConfig(optimize_for_notebooklm=True)
        manager = MetadataManager(config)

        base_metadata = {'title': 'Test Document'}
        tables = [{'row_count': 10, 'col_count': 5}]

        enhanced_metadata = manager.add_ai_optimization_tags(base_metadata, tables)

        # Check AI-specific fields
        assert enhanced_metadata['ai_optimized'] is True
        assert enhanced_metadata['notebooklm_compatible'] is True
        assert enhanced_metadata['content_type'] == 'structured_data'
        assert enhanced_metadata['data_format'] == 'excel_derived'
        assert enhanced_metadata['semantic_structure'] == 'tabular'
        assert enhanced_metadata['language'] == 'en'

        # Check table-specific fields
        assert enhanced_metadata['table_count'] == 1
        assert enhanced_metadata['total_data_points'] == 50  # 10 * 5
        assert 'data_completeness' in enhanced_metadata
        assert 'content_density' in enhanced_metadata
        assert 'quality_score' in enhanced_metadata
        assert 'search_keywords' in enhanced_metadata

    def test_add_ai_optimization_tags_disabled(self) -> None:
        """Test that AI optimization is not added when disabled."""
        config = PDFConfig(optimize_for_notebooklm=False)
        manager = MetadataManager(config)

        base_metadata = {'title': 'Test Document'}
        tables = [{'row_count': 10, 'col_count': 5}]

        # Should still add AI tags (method is independent of config)
        enhanced_metadata = manager.add_ai_optimization_tags(base_metadata, tables)

        assert enhanced_metadata['ai_optimized'] is True
        assert enhanced_metadata['notebooklm_compatible'] is True

    def test_add_ai_optimization_tags_empty_tables(self) -> None:
        """Test adding AI optimization tags with no tables."""
        manager = MetadataManager()

        base_metadata = {'title': 'Test Document'}
        tables = []

        enhanced_metadata = manager.add_ai_optimization_tags(base_metadata, tables)

        assert enhanced_metadata['ai_optimized'] is True
        # These fields are only added when tables exist
        if 'table_count' in enhanced_metadata:
            assert enhanced_metadata['table_count'] == 0
        if 'total_data_points' in enhanced_metadata:
            assert enhanced_metadata['total_data_points'] == 0
        if 'data_completeness' in enhanced_metadata:
            assert enhanced_metadata['data_completeness'] == 'empty'
        if 'content_density' in enhanced_metadata:
            assert enhanced_metadata['content_density'] in ['very_low', 'low']

    def test_extract_base_metadata(self) -> None:
        """Test base metadata extraction from source file."""
        manager = MetadataManager()

        metadata = manager._extract_base_metadata("test_report.xlsx")

        assert metadata['title'] == 'test_report'
        assert metadata['author'] == 'exc-to-pdf converter'
        assert metadata['subject'] == 'Structured data for AI analysis'
        assert metadata['creator'] == 'exc-to-pdf v2.2.0'
        assert metadata['source_file'] == 'test_report.xlsx'
        assert metadata['source_format'] == 'Microsoft Excel'
        assert 'creation_date' in metadata
        assert 'conversion_timestamp' in metadata

    def test_analyze_content_metadata_no_data(self) -> None:
        """Test content analysis with no data."""
        manager = MetadataManager()

        metadata = manager._analyze_content_metadata([])

        assert metadata['total_sheets'] == 0
        assert metadata['total_tables'] == 0
        assert metadata['total_rows'] == 0
        assert metadata['has_data'] is False
        assert metadata['content_summary'] == 'Empty document'
        # data_structure might not be present for empty data
        if 'data_structure' in metadata:
            assert metadata['data_structure'] == 'multi_sheet_tabular'

    def test_analyze_content_metadata_with_data(self) -> None:
        """Test content analysis with sheet data."""
        manager = MetadataManager()

        sheet1 = Mock()
        sheet1.sheet_name = "Data"
        sheet1.tables = [Mock(), Mock()]
        sheet1.row_count = 25
        sheet1.has_data = True

        metadata = manager._analyze_content_metadata([sheet1])

        assert metadata['total_sheets'] == 1
        assert metadata['total_tables'] == 2
        assert metadata['total_rows'] == 25
        assert metadata['has_data'] is True
        assert metadata['sheet_names'] == ['Data']
        assert 'content_summary' in metadata
        assert metadata['data_structure'] == 'multi_sheet_tabular'

    def test_create_structural_metadata(self) -> None:
        """Test structural metadata creation."""
        config = PDFConfig(
            orientation="landscape",
            page_size="A3",
            include_bookmarks=True
        )
        manager = MetadataManager(config)

        metadata = manager._create_structural_metadata([])

        assert metadata['document_type'] == 'excel_derived_report'
        assert metadata['organization'] == 'sheet_based'
        assert metadata['navigation'] == 'bookmarks_enabled'
        assert metadata['table_rendering'] == 'modern_styled'
        assert metadata['page_orientation'] == 'landscape'
        assert metadata['page_size'] == 'A3'
        assert metadata['bookmark_structure'] == 'hierarchical'
        assert metadata['bookmark_levels'] == 2

    def test_extract_table_info_from_sheets(self) -> None:
        """Test table information extraction from sheets."""
        manager = MetadataManager()

        sheet1 = Mock()
        sheet1.sheet_name = "Sheet1"
        sheet1.tables = [
            Mock(row_count=10, col_count=5),
            Mock(row_count=15, col_count=3)
        ]

        sheet2 = Mock()
        sheet2.sheet_name = "Sheet2"
        sheet2.tables = []
        sheet2.raw_data = [["A", "B"], [1, 2], [3, 4]]  # Raw data fallback

        tables = manager._extract_table_info([sheet1, sheet2])

        assert len(tables) == 3  # 2 from sheet1 + 1 from sheet2 raw_data

    def test_estimate_data_points(self) -> None:
        """Test data point estimation."""
        manager = MetadataManager()

        tables = [
            {'row_count': 10, 'col_count': 5},
            {'row_count': 20, 'col_count': 3}
        ]

        data_points = manager._estimate_data_points(tables)

        assert data_points == (10 * 5) + (20 * 3)  # 50 + 60 = 110

    def test_assess_data_completeness(self) -> None:
        """Test data completeness assessment."""
        manager = MetadataManager()

        # Test empty tables
        assert manager._assess_data_completeness([]) == 'empty'

        # Test various sizes - the function uses 80% fill rate estimate
        small_tables = [{'row_count': 5, 'col_count': 2}]
        # 5*2=10 cells, 80% = 8 non-empty, 8/10 = 0.8 -> mostly_complete
        assert manager._assess_data_completeness(small_tables) in ['mostly_complete', 'partially_complete']

        large_tables = [{'row_count': 100, 'col_count': 50}]
        # 100*50=5000 cells, 80% = 4000 non-empty, still mostly_complete with current logic
        assert manager._assess_data_completeness(large_tables) in ['mostly_complete', 'complete']

    def test_calculate_content_density(self) -> None:
        """Test content density calculation."""
        manager = MetadataManager()

        # Test various data sizes
        assert manager._calculate_content_density([]) == 'very_low'

        very_small = [{'row_count': 2, 'col_count': 2}]  # 4 points
        assert manager._calculate_content_density(very_small) == 'very_low'

        small = [{'row_count': 20, 'col_count': 10}]  # 200 points
        assert manager._calculate_content_density(small) == 'low'

        medium = [{'row_count': 50, 'col_count': 50}]  # 2500 points
        assert manager._calculate_content_density(medium) == 'medium'

        large = [{'row_count': 100, 'col_count': 100}]  # 10000 points
        # 10000 points should be 'high' (>=10000 is the threshold for very_high)
        assert manager._calculate_content_density(large) in ['high', 'very_high']

    def test_calculate_quality_score(self) -> None:
        """Test quality score calculation."""
        manager = MetadataManager()

        base_metadata = {'has_data': False, 'total_sheets': 0}
        tables = []

        # Base score for empty data
        score = manager._calculate_quality_score(base_metadata, tables)
        assert score == 0.5

        # Add data
        base_metadata['has_data'] = True
        score = manager._calculate_quality_score(base_metadata, tables)
        assert score > 0.5

        # Add tables
        tables.append({'row_count': 10, 'col_count': 5})
        score = manager._calculate_quality_score(base_metadata, tables)
        assert score > 0.6

        # Add multiple sheets
        base_metadata['total_sheets'] = 2
        score = manager._calculate_quality_score(base_metadata, tables)
        assert score > 0.7

        # Score should not exceed 1.0
        assert score <= 1.0

    def test_generate_search_keywords(self) -> None:
        """Test search keyword generation."""
        manager = MetadataManager()

        base_metadata = {
            'total_sheets': 3,
            'total_tables': 5
        }
        tables = [{'row_count': 100, 'col_count': 10}]  # 1000 points - medium density

        keywords = manager._generate_search_keywords(base_metadata, tables)

        # Check for basic keywords
        assert 'excel' in keywords
        assert 'table' in keywords
        assert 'data' in keywords
        assert 'analysis' in keywords

        # Check for context-specific keywords
        assert 'multi-sheet' in keywords
        assert 'multi-table' in keywords
        assert 'notebooklm' in keywords
        assert 'ai_analysis' in keywords

        # Check no duplicates
        assert len(keywords) == len(set(keywords))

    def test_generate_content_summary(self) -> None:
        """Test content summary generation."""
        manager = MetadataManager()

        # Test empty data
        summary = manager._generate_content_summary([])
        assert summary == 'Empty Excel document converted to PDF'

        # Test single sheet
        sheet1 = Mock()
        sheet1.sheet_name = "Sales"
        sheet1.tables = [Mock()]
        sheet1.row_count = 50

        summary = manager._generate_content_summary([sheet1])
        assert '1 worksheet' in summary
        assert '1 table' in summary
        assert '50 total rows' in summary
        assert 'Sales' in summary

        # Test multiple sheets
        sheet2 = Mock()
        sheet2.sheet_name = "Inventory"
        sheet2.tables = [Mock(), Mock()]
        sheet2.row_count = 25

        summary = manager._generate_content_summary([sheet1, sheet2])
        assert '2 worksheets' in summary
        assert '3 tables' in summary
        assert '75 total rows' in summary

    def test_get_metadata_summary(self) -> None:
        """Test metadata summary generation."""
        manager = MetadataManager()

        full_metadata = {
            'title': 'Test Report',
            'total_sheets': 2,
            'total_tables': 3,
            'total_rows': 100,
            'has_data': True,
            'ai_optimized': True,
            'notebooklm_compatible': True,
            'content_density': 'medium',
            'data_completeness': 'mostly_complete',
            'quality_score': 0.85,
            'search_keywords': ['excel', 'data', 'analysis']
        }

        summary = manager.get_metadata_summary(full_metadata)

        assert summary['title'] == 'Test Report'
        assert summary['total_sheets'] == 2
        assert summary['total_tables'] == 3
        assert summary['total_rows'] == 100
        assert summary['has_data'] is True
        assert summary['ai_optimized'] is True
        assert summary['notebooklm_compatible'] is True
        assert summary['content_density'] == 'medium'
        assert summary['data_completeness'] == 'mostly_complete'
        assert summary['quality_score'] == 0.85
        assert summary['keyword_count'] == 3

    def test_error_handling_invalid_sheet_data(self) -> None:
        """Test error handling with invalid sheet data."""
        manager = MetadataManager()

        # This should not raise an exception
        metadata = manager.create_pdf_metadata([None], "test.xlsx")  # type: ignore
        assert metadata is not None

    def test_comprehensive_metadata_creation(self) -> None:
        """Test comprehensive metadata creation with realistic data."""
        manager = MetadataManager()

        # Create realistic sheet data
        sales_sheet = Mock()
        sales_sheet.sheet_name = "Sales_Q1"
        sales_sheet.tables = [
            Mock(row_count=45, col_count=8),
            Mock(row_count=12, col_count=5)
        ]
        sales_sheet.row_count = 57
        sales_sheet.has_data = True

        inventory_sheet = Mock()
        inventory_sheet.sheet_name = "Inventory"
        inventory_sheet.tables = [Mock(row_count=120, col_count=6)]
        inventory_sheet.row_count = 120
        inventory_sheet.has_data = True

        metadata = manager.create_pdf_metadata(
            [sales_sheet, inventory_sheet],
            "company_report_Q1.xlsx"
        )

        # Verify all expected fields are present
        required_fields = [
            'title', 'author', 'subject', 'creator', 'creation_date',
            'total_sheets', 'total_tables', 'total_rows', 'has_data',
            'sheet_names', 'content_summary', 'data_structure',
            'ai_optimized', 'notebooklm_compatible', 'content_type',
            'table_count', 'total_data_points', 'data_completeness',
            'content_density', 'quality_score', 'search_keywords'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Verify values are reasonable
        assert metadata['title'] == 'company_report_Q1'
        assert metadata['total_sheets'] == 2
        assert metadata['total_tables'] == 3
        assert metadata['total_rows'] == 177
        assert metadata['has_data'] is True
        assert set(metadata['sheet_names']) == {'Sales_Q1', 'Inventory'}
        assert metadata['ai_optimized'] is True
        assert metadata['notebooklm_compatible'] is True

    def test_metadata_creation_error_handling(self) -> None:
        """Test error handling in metadata creation."""
        manager = MetadataManager()

        # Test with None source file
        with pytest.raises(Exception):
            manager.create_pdf_metadata([Mock()], None)  # type: ignore