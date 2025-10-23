"""
Unit tests for chart extraction module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import openpyxl

from exc_to_pdf.chart.chart_extractor import ChartExtractor, ChartType, ChartInfo


class TestChartExtractor:
    """Test cases for ChartExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ChartExtractor()

    def test_init(self):
        """Test ChartExtractor initialization."""
        assert self.extractor is not None
        assert hasattr(self.extractor, '_chart_type_mapping')
        assert ChartType.BAR in self.extractor._chart_type_mapping.values()

    def test_determine_chart_type(self):
        """Test chart type determination."""
        # Test known chart types
        bar_chart = Mock()
        bar_chart.__class__.__name__ = 'BarChart'
        assert self.extractor._determine_chart_type(bar_chart) == ChartType.BAR

        line_chart = Mock()
        line_chart.__class__.__name__ = 'LineChart'
        assert self.extractor._determine_chart_type(line_chart) == ChartType.LINE

        # Test unknown chart type
        unknown_chart = Mock()
        unknown_chart.__class__.__name__ = 'UnknownChart'
        assert self.extractor._determine_chart_type(unknown_chart) == ChartType.UNKNOWN

    def test_col_letter_to_number(self):
        """Test column letter to number conversion."""
        assert self.extractor._col_letter_to_number('A') == 1
        assert self.extractor._col_letter_to_number('B') == 2
        assert self.extractor._col_letter_to_number('Z') == 26
        assert self.extractor._col_letter_to_number('AA') == 27
        assert self.extractor._col_letter_to_number('AB') == 28
        assert self.extractor._col_letter_to_number('AZ') == 52
        assert self.extractor._col_letter_to_number('BA') == 53

    def test_parse_range(self):
        """Test Excel range parsing."""
        # Test basic range
        result = self.extractor._parse_range('A1:B10')
        assert result == (None, 1, 1, 2, 10)

        # Test range with worksheet
        result = self.extractor._parse_range('Sheet1!$A$1:$B$10')
        assert result == ('Sheet1', 1, 1, 2, 10)

        # Test invalid range
        result = self.extractor._parse_range('invalid')
        assert result is None

    @patch('exc_to_pdf.chart.chart_extractor.logger')
    def test_extract_chart_info_with_mock_chart(self, mock_logger):
        """Test chart info extraction with mock chart."""
        # Create mock chart
        mock_chart = Mock()
        mock_chart.__class__.__name__ = 'BarChart'
        mock_chart.title = Mock()
        mock_chart.title.text = 'Test Chart'
        mock_chart.series = []
        mock_chart.legend = Mock()
        mock_chart.legend.visible = True
        mock_chart.legend.position = 'r'

        # Test extraction
        chart_info = self.extractor._extract_chart_info(mock_chart, 'TestSheet')

        assert chart_info is not None
        assert chart_info.chart_type == ChartType.BAR
        assert chart_info.title == 'Test Chart'
        assert chart_info.worksheet_name == 'TestSheet'
        assert isinstance(chart_info.data_ranges, list)
        assert isinstance(chart_info.series_data, list)
        assert isinstance(chart_info.styling, dict)

    def test_extract_charts_from_empty_worksheet(self):
        """Test extracting charts from empty worksheet."""
        mock_worksheet = Mock()
        mock_worksheet.title = 'EmptySheet'
        mock_worksheet._charts = []

        charts = self.extractor.extract_charts_from_worksheet(mock_worksheet)
        assert charts == []

    @patch('exc_to_pdf.chart.chart_extractor.logger')
    def test_extract_charts_from_worksheet_with_error(self, mock_logger):
        """Test error handling in chart extraction."""
        mock_worksheet = Mock()
        mock_worksheet.title = 'ErrorSheet'
        mock_worksheet._charts = Mock(side_effect=Exception('Test error'))

        charts = self.extractor.extract_charts_from_worksheet(mock_worksheet)
        assert charts == []
        mock_logger.error.assert_called()

    def test_detect_charts_in_empty_workbook(self):
        """Test detecting charts in empty workbook."""
        mock_workbook = Mock()
        mock_workbook.worksheets = []
        mock_workbook.chartsheets = []

        charts = self.extractor.detect_charts_in_workbook(mock_workbook)
        assert charts == {}

    @patch.object(ChartExtractor, 'extract_charts_from_worksheet')
    @patch.object(ChartExtractor, 'extract_charts_from_chartsheet')
    def test_detect_charts_in_workbook(self, mock_extract_chartsheet, mock_extract_worksheet):
        """Test detecting charts in workbook with charts."""
        # Setup mocks
        mock_workbook = Mock()
        mock_worksheet1 = Mock()
        mock_worksheet1.title = 'Sheet1'
        mock_worksheet2 = Mock()
        mock_worksheet2.title = 'Sheet2'
        mock_workbook.worksheets = [mock_worksheet1, mock_worksheet2]
        mock_workbook.chartsheets = []

        # Mock chart extraction
        chart_info = Mock()
        mock_extract_worksheet.side_effect = [
            [chart_info],  # Sheet1 has charts
            []             # Sheet2 has no charts
        ]

        # Test detection
        charts = self.extractor.detect_charts_in_workbook(mock_workbook)

        assert 'Sheet1' in charts
        assert 'Sheet2' not in charts
        assert len(charts['Sheet1']) == 1
        mock_extract_worksheet.assert_called()

    def test_chart_type_mapping_completeness(self):
        """Test that all ChartType enum values are mapped."""
        mapped_types = set(self.extractor._chart_type_mapping.values())
        expected_types = set(ChartType)

        # Should have mappings for all types except UNKNOWN
        assert mapped_types == expected_types - {ChartType.UNKNOWN}

    def test_extract_series_data_empty_series(self):
        """Test extracting series data from empty series."""
        mock_chart = Mock()
        mock_chart.series = []

        data_ranges, series_data = self.extractor._extract_series_data(mock_chart)

        assert data_ranges == []
        assert series_data == []

    def test_extract_series_data_with_series(self):
        """Test extracting series data with mock series."""
        # Create mock series
        mock_series = Mock()
        mock_series.title = Mock()
        mock_series.title.text = 'Series 1'
        mock_series.values = 'Sheet1!$A$1:$A$10'
        mock_series.categories = 'Sheet1!$B$1:$B$10'
        mock_series.graphicalProperties = {'solidFill': {'color': '#FF0000'}}

        mock_chart = Mock()
        mock_chart.series = [mock_series]

        data_ranges, series_data = self.extractor._extract_series_data(mock_chart)

        assert len(data_ranges) == 2
        assert 'Sheet1!$A$1:$A$10' in data_ranges
        assert 'Sheet1!$B$1:$B$10' in data_ranges
        assert len(series_data) == 1
        assert series_data[0]['title'] == 'Series 1'

    def test_extract_position_default(self):
        """Test default position extraction."""
        mock_chart = Mock()

        position = self.extractor._extract_position(mock_chart)

        assert position == (0, 0, 400, 300)

    def test_extract_styling_with_legend(self):
        """Test styling extraction with legend."""
        mock_chart = Mock()
        mock_chart.legend = Mock()
        mock_chart.legend.visible = True
        mock_chart.legend.position = 'r'
        mock_chart.display_blanks_as = 'gap'

        styling = self.extractor._extract_styling(mock_chart)

        assert 'legend' in styling
        assert styling['legend']['visible'] == True
        assert styling['legend']['position'] == 'r'
        assert styling['display_blanks_as'] == 'gap'

    def test_extract_styling_with_axes(self):
        """Test styling extraction with axes."""
        mock_chart = Mock()
        mock_chart.x_axis = Mock()
        mock_chart.x_axis.title = Mock()
        mock_chart.x_axis.title.text = 'X Axis'
        mock_chart.x_axis.visible = True
        mock_chart.y_axis = Mock()
        mock_chart.y_axis.title = Mock()
        mock_chart.y_axis.title.text = 'Y Axis'
        mock_chart.y_axis.visible = True

        styling = self.extractor._extract_styling(mock_chart)

        assert 'x_axis' in styling
        assert styling['x_axis']['title'] == 'X Axis'
        assert 'y_axis' in styling
        assert styling['y_axis']['title'] == 'Y Axis'

    @patch('exc_to_pdf.chart.chart_extractor.logger')
    def test_extract_charts_from_chartsheet(self, mock_logger):
        """Test extracting chart from chartsheet."""
        mock_chartsheet = Mock()
        mock_chartsheet.title = 'ChartSheet1'
        mock_chart = Mock()
        mock_chartsheet._charts = [mock_chart]

        # Mock the _extract_chart_info method
        with patch.object(self.extractor, '_extract_chart_info') as mock_extract:
            mock_extract.return_value = Mock()

            result = self.extractor.extract_charts_from_chartsheet(mock_chartsheet)

            mock_extract.assert_called_once_with(mock_chart, 'ChartSheet1')
            assert result is not None

    @patch('exc_to_pdf.chart.chart_extractor.logger')
    def test_extract_charts_from_chartsheet_empty(self, mock_logger):
        """Test extracting from empty chartsheet."""
        mock_chartsheet = Mock()
        mock_chartsheet.title = 'EmptyChartSheet'
        mock_chartsheet._charts = []

        result = self.extractor.extract_charts_from_chartsheet(mock_chartsheet)

        assert result is None

    @patch('exc_to_pdf.chart.chart_extractor.logger')
    def test_extract_charts_from_chartsheet_error(self, mock_logger):
        """Test error handling in chartsheet extraction."""
        mock_chartsheet = Mock()
        mock_chartsheet.title = 'ErrorChartSheet'
        mock_chartsheet._charts = Mock(side_effect=Exception('Test error'))

        result = self.extractor.extract_charts_from_chartsheet(mock_chartsheet)

        assert result is None
        mock_logger.error.assert_called()

    def test_chart_info_dataclass(self):
        """Test ChartInfo dataclass structure."""
        chart_info = ChartInfo(
            chart_type=ChartType.BAR,
            title='Test Chart',
            data_ranges=['Sheet1!A1:B10'],
            series_data=[{'title': 'Series 1'}],
            position=(0, 0, 400, 300),
            styling={'legend': {'visible': True}},
            worksheet_name='Sheet1'
        )

        assert chart_info.chart_type == ChartType.BAR
        assert chart_info.title == 'Test Chart'
        assert len(chart_info.data_ranges) == 1
        assert chart_info.worksheet_name == 'Sheet1'

        # Test immutability of dataclass
        with pytest.raises(AttributeError):
            chart_info.new_field = 'test'