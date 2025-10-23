"""
Chart to PDF rendering module.

This module provides capabilities to render matplotlib charts as PDF-compatible
images for integration into PDF documents.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging
import tempfile
import io

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image

from .chart_recreator import ChartRecreator
from .chart_extractor import ChartInfo

logger = logging.getLogger(__name__)


class ChartRenderer:
    """Renders charts for PDF integration."""

    def __init__(self, dpi: int = 300, format_type: str = 'png'):
        """Initialize chart renderer.

        Args:
            dpi: Resolution for chart rendering
            format_type: Output format ('png', 'pdf', 'svg')
        """
        self.dpi = dpi
        self.format_type = format_type.lower()

        if self.format_type not in ['png', 'pdf', 'svg']:
            raise ValueError(f"Unsupported format: {format_type}. Use 'png', 'pdf', or 'svg'")

    def render_chart_to_image(self, figure: Figure) -> Optional[ImageReader]:
        """Render a matplotlib figure to a ReportLab-compatible image.

        Args:
            figure: matplotlib Figure object

        Returns:
            ReportLab ImageReader object or None if rendering failed
        """
        try:
            # Create temporary file for chart
            with tempfile.NamedTemporaryFile(suffix=f'.{self.format_type}', delete=False) as tmp_file:
                temp_path = tmp_file.name

            # Save figure with high quality
            figure.savefig(
                temp_path,
                format=self.format_type,
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                transparent=False
            )

            # Create ReportLab ImageReader
            image_reader = ImageReader(temp_path)

            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

            logger.debug(f"Successfully rendered chart to {self.format_type.upper()}")
            return image_reader

        except Exception as e:
            logger.error(f"Error rendering chart to image: {e}")
            # Clean up temporary file if it exists
            if 'temp_path' in locals():
                Path(temp_path).unlink(missing_ok=True)
            return None

    def render_chart_to_bytes(self, figure: Figure) -> Optional[bytes]:
        """Render a matplotlib figure to bytes.

        Args:
            figure: matplotlib Figure object

        Returns:
            Bytes data or None if rendering failed
        """
        try:
            # Create in-memory bytes buffer
            buffer = io.BytesIO()

            # Save figure to buffer
            figure.savefig(
                buffer,
                format=self.format_type,
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                transparent=False
            )

            # Get bytes data
            buffer.seek(0)
            image_data = buffer.getvalue()
            buffer.close()

            logger.debug(f"Successfully rendered chart to bytes ({len(image_data)} bytes)")
            return image_data

        except Exception as e:
            logger.error(f"Error rendering chart to bytes: {e}")
            if 'buffer' in locals():
                buffer.close()
            return None

    def render_chart_to_reportlab_image(self, figure: Figure,
                                      width: float = None,
                                      height: float = None) -> Optional[Image]:
        """Render a matplotlib figure directly to a ReportLab Image.

        Args:
            figure: matplotlib Figure object
            width: Width in points (optional)
            height: Height in points (optional)

        Returns:
            ReportLab Image object or None if rendering failed
        """
        try:
            # Render to bytes first
            image_data = self.render_chart_to_bytes(figure)
            if not image_data:
                return None

            # Create in-memory file-like object
            image_buffer = io.BytesIO(image_data)

            # Create ReportLab Image
            reportlab_image = Image(image_buffer)

            # Set dimensions if provided
            if width is not None:
                reportlab_image.drawWidth = width
            if height is not None:
                reportlab_image.drawHeight = height

            # Maintain aspect ratio if only one dimension is provided
            if width is not None and height is None:
                aspect_ratio = reportlab_image.imageHeight / reportlab_image.imageWidth
                reportlab_image.drawHeight = width * aspect_ratio
            elif height is not None and width is None:
                aspect_ratio = reportlab_image.imageWidth / reportlab_image.imageHeight
                reportlab_image.drawWidth = height * aspect_ratio

            logger.debug("Successfully rendered chart to ReportLab Image")
            return reportlab_image

        except Exception as e:
            logger.error(f"Error rendering chart to ReportLab Image: {e}")
            if 'image_buffer' in locals():
                image_buffer.close()
            return None

    def render_charts_from_excel(self, excel_file_path: str,
                               charts_info: List[ChartInfo]) -> List[Image]:
        """Render charts directly from Excel file.

        Args:
            excel_file_path: Path to Excel file
            charts_info: List of ChartInfo objects

        Returns:
            List of ReportLab Image objects
        """
        images = []

        try:
            # Create chart recreator
            recreator = ChartRecreator(excel_file_path)

            # Recreate and render each chart
            for chart_info in charts_info:
                # Recreate chart
                figure = recreator.recreate_chart(chart_info)
                if not figure:
                    logger.warning(f"Failed to recreate chart: {chart_info.title}")
                    continue

                # Render to ReportLab Image
                image = self.render_chart_to_reportlab_image(figure)
                if image:
                    images.append(image)
                    logger.info(f"Successfully rendered chart: {chart_info.title}")
                else:
                    logger.warning(f"Failed to render chart: {chart_info.title}")

                # Close figure to free memory
                plt.close(figure)

        except Exception as e:
            logger.error(f"Error rendering charts from Excel: {e}")

        logger.info(f"Successfully rendered {len(images)} out of {len(charts_info)} charts")
        return images

    def calculate_optimal_size(self, figure: Figure,
                             max_width: float = 400,
                             max_height: float = 300) -> tuple[float, float]:
        """Calculate optimal size for chart rendering.

        Args:
            figure: matplotlib Figure object
            max_width: Maximum width in points
            max_height: Maximum height in points

        Returns:
            Tuple of (width, height) in points
        """
        try:
            # Get figure size in inches
            fig_width_inch, fig_height_inch = figure.get_size_inches()

            # Convert to points (1 inch = 72 points)
            fig_width_pts = fig_width_inch * 72
            fig_height_pts = fig_height_inch * 72

            # Calculate scaling factor to fit within max dimensions
            width_scale = max_width / fig_width_pts
            height_scale = max_height / fig_height_pts
            scale = min(width_scale, height_scale, 1.0)  # Don't upscale

            # Calculate final dimensions
            final_width = fig_width_pts * scale
            final_height = fig_height_pts * scale

            return final_width, final_height

        except Exception as e:
            logger.error(f"Error calculating optimal size: {e}")
            return max_width, max_height

    def render_chart_with_size(self, figure: Figure,
                             max_width: float = 400,
                             max_height: float = 300) -> Optional[Image]:
        """Render chart with automatic size calculation.

        Args:
            figure: matplotlib Figure object
            max_width: Maximum width in points
            max_height: Maximum height in points

        Returns:
            ReportLab Image object with optimal dimensions
        """
        try:
            # Calculate optimal size
            width, height = self.calculate_optimal_size(figure, max_width, max_height)

            # Render with calculated size
            return self.render_chart_to_reportlab_image(figure, width, height)

        except Exception as e:
            logger.error(f"Error rendering chart with size: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        # Clean up any matplotlib resources
        plt.close('all')