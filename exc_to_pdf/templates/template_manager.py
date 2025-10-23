"""
Template manager for advanced template system.

This module provides template loading, inheritance, and management capabilities.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging

from .template_config import TemplateConfig
from .style_presets import StylePresets

logger = logging.getLogger(__name__)


class Template:
    """Represents a loaded template with its configuration."""

    def __init__(self, config: TemplateConfig):
        """Initialize template with configuration.

        Args:
            config: Template configuration
        """
        self.config = config
        self.name = config.template_name
        self.version = config.template_version

    def apply_to_pdf_config(self, pdf_config) -> None:
        """Apply template settings to PDF configuration.

        Args:
            pdf_config: PDF configuration to modify
        """
        # Copy relevant settings from template to PDF config
        pdf_config.header_background = self.config.header_background
        pdf_config.header_text_color = self.config.header_text_color
        pdf_config.alternate_rows = self.config.alternate_rows
        pdf_config.alternate_row_color = self.config.alternate_row_color
        pdf_config.font_family = self.config.font_family
        pdf_config.font_size = self.config.font_size
        pdf_config.header_font_size = self.config.header_font_size


class TemplateManager:
    """Manages template loading and inheritance."""

    def __init__(self):
        """Initialize template manager."""
        self._loaded_templates: Dict[str, Template] = {}
        self._template_cache: Dict[str, TemplateConfig] = {}

    def load_template(
        self, template_name: str, base_template: Optional[str] = None
    ) -> Template:
        """Load a template by name.

        Args:
            template_name: Name of the template to load
            base_template: Optional base template for inheritance

        Returns:
            Loaded Template object
        """
        # Check cache first
        cache_key = f"{template_name}_{base_template or 'none'}"
        if cache_key in self._template_cache:
            return Template(self._template_cache[cache_key])

        try:
            # Load base template configuration
            if template_name in StylePresets.list_available_styles():
                config = StylePresets.get_preset(template_name)
            else:
                # Try to load from file (future implementation)
                raise ValueError(f"Unknown template: {template_name}")

            # Apply inheritance if base template specified
            if base_template and base_template != template_name:
                base_config = StylePresets.get_preset(base_template)
                config = config.merge_with_base(base_config)

            # Cache the configuration
            self._template_cache[cache_key] = config

            template = Template(config)
            self._loaded_templates[template_name] = template

            logger.info(f"Loaded template: {template_name}")
            return template

        except Exception as e:
            logger.error(f"Error loading template '{template_name}': {e}")
            raise

    def get_available_templates(self) -> list[str]:
        """Get list of available template names.

        Returns:
            List of available template names
        """
        return StylePresets.list_available_locales()

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists.

        Args:
            template_name: Name of the template to check

        Returns:
            True if template exists, False otherwise
        """
        return template_name in StylePresets.list_available_styles()

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template.

        Args:
            template_name: Name of the template

        Returns:
            Dictionary with template information
        """
        if not self.template_exists(template_name):
            return {}

        try:
            config = StylePresets.get_preset(template_name)
            return {
                "name": config.template_name,
                "version": config.template_version,
                "description": StylePresets.get_style_description(template_name),
                "features": {
                    "charts": config.include_charts,
                    "branding": config.branding is not None,
                    "dynamic_styling": config.dynamic_column_width,
                },
            }
        except Exception as e:
            logger.error(f"Error getting template info for '{template_name}': {e}")
            return {}
