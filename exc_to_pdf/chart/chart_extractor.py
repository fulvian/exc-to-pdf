"""
Chart extraction module for detecting and extracting chart information from Excel files.

This module provides capabilities to detect charts in Excel worksheets and extract
their data, styling, and configuration information for recreation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging

from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.chartsheet import Chartsheet

# Try to import chart classes - handle different openpyxl versions
try:
    from openpyxl.chart import Chart, BarChart, LineChart, PieChart, ScatterChart
except ImportError:
    # Handle newer openpyxl versions or different import paths
    try:
        from openpyxl.chart._chart import Chart
        from openpyxl.chart.bar_chart import BarChart
        from openpyxl.chart.line_chart import LineChart
        from openpyxl.chart.pie_chart import PieChart
        from openpyxl.chart.scatter_chart import ScatterChart
    except ImportError:
        # Final fallback - use generic Chart if available
        try:
            from openpyxl.chart import Chart
            # Create placeholder classes for missing chart types
            BarChart = LineChart = PieChart = ScatterChart = Chart
        except ImportError:
            # If even Chart is not available, define dummy classes
            Chart = object
            BarChart = LineChart = PieChart = ScatterChart = object

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    UNKNOWN = "unknown"


@dataclass
class ChartInfo:
    """Information about an extracted chart."""
    chart_type: ChartType
    title: str
    data_ranges: List[str]  # Cell ranges containing data
    series_data: List[Dict[str, Any]]  # Series information
    position: Tuple[int, int, int, int]  # (col, row, width, height)
    styling: Dict[str, Any]  # Chart styling information
    worksheet_name: str


class ChartExtractor:
    """Extracts chart information from Excel worksheets."""

    def __init__(self):
        """Initialize chart extractor."""
        self._chart_type_mapping = {
            'BarChart': ChartType.BAR,
            'LineChart': ChartType.LINE,
            'PieChart': ChartType.PIE,
            'ScatterChart': ChartType.SCATTER,
            'AreaChart': ChartType.AREA,
        }

    def extract_charts_from_worksheet(self, worksheet: Worksheet) -> List[ChartInfo]:
        """Extract all charts from a worksheet.

        Args:
            worksheet: Excel worksheet to extract charts from

        Returns:
            List of ChartInfo objects containing chart information
        """
        charts = []

        try:
            # Check for embedded charts in worksheet
            if hasattr(worksheet, '_charts'):
                for chart in worksheet._charts:
                    chart_info = self._extract_chart_info(chart, worksheet.title)
                    if chart_info:
                        charts.append(chart_info)

            logger.info(
                f"Extracted {len(charts)} charts from worksheet: {worksheet.title}"
            )

        except Exception as e:
            logger.error(
                f"Error extracting charts from worksheet {worksheet.title}: {e}"
            )

        return charts

    def extract_charts_from_chartsheet(self, chartsheet: Chartsheet) -> Optional[ChartInfo]:
        """Extract chart from a dedicated chartsheet.

        Args:
            chartsheet: Excel chartsheet containing a single chart

        Returns:
            ChartInfo object if chart found, None otherwise
        """
        try:
            chart = chartsheet._charts[0] if chartsheet._charts else None
            if chart:
                return self._extract_chart_info(chart, chartsheet.title)
        except Exception as e:
            logger.error(f"Error extracting chart from chartsheet {chartsheet.title}: {e}")

        return None

    def _extract_chart_info(self, chart: Chart, worksheet_name: str) -> Optional[ChartInfo]:
        """Extract information from a single chart object.

        Args:
            chart: openpyxl Chart object
            worksheet_name: Name of the containing worksheet

        Returns:
            ChartInfo object with extracted information
        """
        try:
            # Determine chart type
            chart_type = self._determine_chart_type(chart)

            # Extract chart title
            title = getattr(chart.title, 'text', '') if hasattr(chart, 'title') else ''

            # Extract data ranges and series information
            data_ranges, series_data = self._extract_series_data(chart)

            # Extract position information (if available)
            position = self._extract_position(chart)

            # Extract styling information
            styling = self._extract_styling(chart)

            return ChartInfo(
                chart_type=chart_type,
                title=title,
                data_ranges=data_ranges,
                series_data=series_data,
                position=position,
                styling=styling,
                worksheet_name=worksheet_name
            )

        except Exception as e:
            logger.error(f"Error extracting chart info: {e}")
            return None

    def _determine_chart_type(self, chart: Chart) -> ChartType:
        """Determine the type of chart.

        Args:
            chart: openpyxl Chart object

        Returns:
            ChartType enum value
        """
        chart_class_name = chart.__class__.__name__
        return self._chart_type_mapping.get(chart_class_name, ChartType.UNKNOWN)

    def _extract_series_data(self, chart: Chart) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract data ranges and series information from chart.

        Args:
            chart: openpyxl Chart object

        Returns:
            Tuple of (data_ranges, series_data)
        """
        data_ranges = []
        series_data = []

        try:
            for series in chart.series:
                series_info = {
                    'title': getattr(series.title, 'text', '') if hasattr(series, 'title') else '',
                    'values': str(series.values) if hasattr(series, 'values') else '',
                    'categories': str(series.categories) if hasattr(series, 'categories') else '',
                    'color': getattr(series, 'graphicalProperties', {}).get('solidFill', {}).get('color', None)
                }

                series_data.append(series_info)

                # Collect data ranges
                if hasattr(series, 'values') and series.values:
                    data_ranges.append(str(series.values))
                if hasattr(series, 'categories') and series.categories:
                    data_ranges.append(str(series.categories))

        except Exception as e:
            logger.error(f"Error extracting series data: {e}")

        return list(set(data_ranges)), series_data

    def _extract_position(self, chart: Chart) -> Tuple[int, int, int, int]:
        """Extract chart position information.

        Args:
            chart: openpyxl Chart object

        Returns:
            Tuple of (col, row, width, height)
        """
        # openpyxl chart positioning is limited - provide defaults
        return (0, 0, 400, 300)  # Default position and size

    def _extract_styling(self, chart: Chart) -> Dict[str, Any]:
        """Extract styling information from chart.

        Args:
            chart: openpyxl Chart object

        Returns:
            Dictionary containing styling information
        """
        styling = {}

        try:
            # Extract basic styling properties
            if hasattr(chart, 'legend'):
                styling['legend'] = {
                    'position': getattr(chart.legend, 'position', 'r'),
                    'visible': chart.legend.visible if hasattr(chart.legend, 'visible') else True
                }

            if hasattr(chart, 'display_blanks_as'):
                styling['display_blanks_as'] = chart.display_blanks_as

            # Extract axis information if available
            if hasattr(chart, 'x_axis'):
                styling['x_axis'] = {
                    'title': getattr(chart.x_axis.title, 'text', '') if hasattr(chart.x_axis, 'title') else '',
                    'visible': chart.x_axis.visible if hasattr(chart.x_axis, 'visible') else True
                }

            if hasattr(chart, 'y_axis'):
                styling['y_axis'] = {
                    'title': getattr(chart.y_axis.title, 'text', '') if hasattr(chart.y_axis, 'title') else '',
                    'visible': chart.y_axis.visible if hasattr(chart.y_axis, 'visible') else True
                }

        except Exception as e:
            logger.error(f"Error extracting styling: {e}")

        return styling

    def detect_charts_in_workbook(self, workbook: Workbook) -> Dict[str, List[ChartInfo]]:
        """Detect all charts in an Excel workbook.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            Dictionary mapping worksheet names to lists of ChartInfo objects
        """
        all_charts = {}

        try:
            # Check regular worksheets for embedded charts
            for worksheet in workbook.worksheets:
                charts = self.extract_charts_from_worksheet(worksheet)
                if charts:
                    all_charts[worksheet.title] = charts

            # Check for dedicated chartsheets
            if hasattr(workbook, 'chartsheets'):
                for chartsheet in workbook.chartsheets:
                    chart = self.extract_charts_from_chartsheet(chartsheet)
                    if chart:
                        all_charts[chartsheet.title] = [chart]

            total_charts = sum(len(charts) for charts in all_charts.values())
            logger.info(f"Detected {total_charts} charts across {len(all_charts)} worksheets")

        except Exception as e:
            logger.error(f"Error detecting charts in workbook: {e}")

        return all_charts