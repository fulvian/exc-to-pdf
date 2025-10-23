"""
Advanced template system for exc-to-pdf.

This module provides comprehensive template management with inheritance,
branding, and dynamic styling capabilities.
"""

from .template_manager import TemplateManager, Template
from .template_config import TemplateConfig, BrandingConfig
from .style_presets import StylePresets, TemplateStyle

__all__ = [
    'TemplateManager',
    'Template',
    'TemplateConfig',
    'BrandingConfig',
    'StylePresets',
    'TemplateStyle'
]