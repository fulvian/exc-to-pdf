"""
DevStream Installation Module

This module provides comprehensive project detection and installation support
for universal DevStream deployment across different project types and environments.
"""

from .project_detector import ProjectDetector, ProjectType, detect_project_type, calculate_project_score

__all__ = [
    "ProjectDetector",
    "ProjectType",
    "detect_project_type",
    "calculate_project_score",
]