"""
Project Detection Framework for DevStream Installation

This module implements a comprehensive project type detection system that analyzes
directories to determine project types, programming languages, frameworks, and
development environments. It uses multi-factor analysis based on configuration files,
documentation patterns, git repositories, and source code characteristics.

Based on Context7 research and STRUCT project management patterns for robust
project structure detection and classification.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


class ProjectType(Enum):
    """Enumeration of supported project types and their characteristics."""

    # Programming Languages
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    C_SHARP = "c_sharp"
    CPP = "cpp"

    # Frameworks and Platforms
    NODE_JS = "node_js"
    REACT = "react"
    ANGULAR = "angular"
    VUE = "vue"
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    SPRING = "spring"
    DOTNET = "dotnet"

    # Build Systems and Package Managers
    MAVEN = "maven"
    GRADLE = "gradle"
    NPM = "npm"
    YARN = "yarn"
    PIP = "pip"
    POETRY = "poetry"
    CONDA = "conda"
    CARGO = "cargo"

    # Containerization and DevOps
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"

    # Data Science
    JUPYTER = "jupyter"
    R = "r"

    # General Types
    UNKNOWN = "unknown"
    EMPTY = "empty"
    MIXED = "mixed"


@dataclass
class ProjectIndicator:
    """Represents a detected project indicator with its weight."""

    name: str
    value: Any
    weight: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate indicator values."""
        if self.weight < 0 or self.weight > 1.0:
            raise ValueError("Weight must be between 0 and 1.0")
        if self.confidence < 0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0 and 1.0")


@dataclass
class ProjectAnalysis:
    """Comprehensive project analysis results."""

    project_path: str
    project_type: ProjectType
    indicators: List[ProjectIndicator] = field(default_factory=list)
    confidence_score: float = 0.0
    detected_languages: Set[str] = field(default_factory=set)
    detected_frameworks: Set[str] = field(default_factory=set)
    build_systems: Set[str] = field(default_factory=set)
    package_managers: Set[str] = field(default_factory=set)
    has_git_repo: bool = False
    has_documentation: bool = False
    is_empty: bool = False
    file_count: int = 0
    directory_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "project_path": str(self.project_path),
            "project_type": self.project_type.value,
            "confidence_score": self.confidence_score,
            "detected_languages": list(self.detected_languages),
            "detected_frameworks": list(self.detected_frameworks),
            "build_systems": list(self.build_systems),
            "package_managers": list(self.package_managers),
            "has_git_repo": self.has_git_repo,
            "has_documentation": self.has_documentation,
            "is_empty": self.is_empty,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "indicators": [
                {
                    "name": ind.name,
                    "value": str(ind.value),
                    "weight": ind.weight,
                    "confidence": ind.confidence
                }
                for ind in self.indicators
            ]
        }


class ProjectDetector:
    """
    Multi-factor project type detector.

    Implements sophisticated project detection using configuration files,
    documentation patterns, source code analysis, and build system indicators.
    Based on Context7 research and STRUCT project management best practices.
    """

    # Configuration file patterns with weights
    CONFIG_PATTERNS = {
        # Package managers and build systems
        "package.json": {"type": ProjectType.NODE_JS, "weight": 0.9, "manager": "npm"},
        "requirements.txt": {"type": ProjectType.PYTHON, "weight": 0.8, "manager": "pip"},
        "pyproject.toml": {"type": ProjectType.PYTHON, "weight": 0.9, "manager": "poetry"},
        "Pipfile": {"type": ProjectType.PYTHON, "weight": 0.8, "manager": "pipenv"},
        "environment.yml": {"type": ProjectType.PYTHON, "weight": 0.7, "manager": "conda"},
        "Cargo.toml": {"type": ProjectType.RUST, "weight": 0.9, "manager": "cargo"},
        "go.mod": {"type": ProjectType.GO, "weight": 0.9, "manager": "go"},
        "go.sum": {"type": ProjectType.GO, "weight": 0.7, "manager": "go"},
        "pom.xml": {"type": ProjectType.JAVA, "weight": 0.9, "manager": "maven"},
        "build.gradle": {"type": ProjectType.JAVA, "weight": 0.9, "manager": "gradle"},
        "build.gradle.kts": {"type": ProjectType.JAVA, "weight": 0.9, "manager": "gradle"},
        "composer.json": {"type": ProjectType.PHP, "weight": 0.9, "manager": "composer"},
        "Gemfile": {"type": ProjectType.RUBY, "weight": 0.9, "manager": "bundler"},
        "csproj": {"type": ProjectType.C_SHARP, "weight": 0.9, "manager": "dotnet"},
        "sln": {"type": ProjectType.C_SHARP, "weight": 0.8, "manager": "dotnet"},
        "CMakeLists.txt": {"type": ProjectType.CPP, "weight": 0.8, "manager": "cmake"},
        "Makefile": {"type": ProjectType.UNKNOWN, "weight": 0.6, "manager": "make"},

        # Docker and containerization
        "Dockerfile": {"type": ProjectType.DOCKER, "weight": 0.8, "manager": "docker"},
        "docker-compose.yml": {"type": ProjectType.DOCKER, "weight": 0.8, "manager": "docker"},
        "docker-compose.yaml": {"type": ProjectType.DOCKER, "weight": 0.8, "manager": "docker"},

        # DevOps and Infrastructure
        "terraform.tf": {"type": ProjectType.TERRAFORM, "weight": 0.9, "manager": "terraform"},
        "main.tf": {"type": ProjectType.TERRAFORM, "weight": 0.9, "manager": "terraform"},

        # Jupyter and Data Science
        ".ipynb_checkpoints": {"type": ProjectType.JUPYTER, "weight": 0.7, "manager": "jupyter"},

        # TypeScript specific
        "tsconfig.json": {"type": ProjectType.TYPESCRIPT, "weight": 0.8, "manager": "typescript"},
        "package-lock.json": {"type": ProjectType.NODE_JS, "weight": 0.6, "manager": "npm"},
        "yarn.lock": {"type": ProjectType.NODE_JS, "weight": 0.6, "manager": "yarn"},

        # Python specific files
        "setup.py": {"type": ProjectType.PYTHON, "weight": 0.8, "manager": "pip"},
        "setup.cfg": {"type": ProjectType.PYTHON, "weight": 0.7, "manager": "pip"},
        "tox.ini": {"type": ProjectType.PYTHON, "weight": 0.6, "manager": "pip"},
        "poetry.lock": {"type": ProjectType.PYTHON, "weight": 0.8, "manager": "poetry"},

        # Java specific
        "gradle.properties": {"type": ProjectType.JAVA, "weight": 0.7, "manager": "gradle"},
        "settings.gradle": {"type": ProjectType.JAVA, "weight": 0.7, "manager": "gradle"},

        # JavaScript/TypeScript frameworks
        "angular.json": {"type": ProjectType.ANGULAR, "weight": 0.9, "manager": "angular"},
        "vue.config.js": {"type": ProjectType.VUE, "weight": 0.8, "manager": "vue"},
        "next.config.js": {"type": ProjectType.REACT, "weight": 0.8, "manager": "next"},
        "nuxt.config.js": {"type": ProjectType.VUE, "weight": 0.8, "manager": "nuxt"},
    }

    # Source code file extensions with language mapping
    SOURCE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c_sharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".sql": "sql",
        ".sh": "shell",
        ".bat": "batch",
        ".ps1": "powershell",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".xml": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".ipynb": "jupyter",
        ".r": "r",
        ".R": "r",
    }

    # Documentation patterns
    DOC_PATTERNS = [
        "README", "readme", "Readme", "README.md", "readme.md",
        "doc", "docs", "Doc", "Docs", "documentation", "Documentation"
    ]

    # Framework-specific patterns
    FRAMEWORK_PATTERNS = {
        "django": {
            "files": ["manage.py", "wsgi.py", "asgi.py"],
            "dirs": ["django", "django_project"],
            "weight": 0.9
        },
        "flask": {
            "files": ["app.py", "flask_app.py"],
            "dirs": ["templates", "static"],
            "weight": 0.8
        },
        "fastapi": {
            "files": ["main.py", "app.py"],
            "dirs": ["routers", "models"],
            "weight": 0.8
        },
        "spring": {
            "files": ["Application.java"],
            "dirs": ["src/main/java"],
            "weight": 0.9
        },
        "react": {
            "files": ["package.json"],
            "content_patterns": ["react", "jsx", "tsx"],
            "weight": 0.8
        },
    }

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize project detector.

        Args:
            min_confidence: Minimum confidence threshold for detection
        """
        self.min_confidence = min_confidence

    def detect_project_type(self, project_path: str) -> ProjectAnalysis:
        """
        Detect project type using multi-factor analysis.

        Args:
            project_path: Path to the project directory

        Returns:
            ProjectAnalysis with comprehensive detection results

        Raises:
            ValueError: If project_path is invalid or inaccessible
            OSError: If directory cannot be accessed
        """
        logger.info("detecting_project_type", project_path=project_path)

        try:
            path = Path(project_path).resolve()

            # Validate project path
            if not path.exists():
                raise ValueError(f"Project path does not exist: {project_path}")
            if not path.is_dir():
                raise ValueError(f"Project path is not a directory: {project_path}")

            # Initialize analysis
            analysis = ProjectAnalysis(
                project_path=str(path),
                project_type=ProjectType.UNKNOWN,
                confidence_score=0.0
            )

            # Check if directory is empty
            self._analyze_directory_structure(analysis)
            if analysis.is_empty:
                analysis.project_type = ProjectType.EMPTY
                analysis.confidence_score = 1.0
                return analysis

            # Perform multi-factor detection
            self._detect_git_repository(analysis)
            self._detect_documentation(analysis)
            self._detect_configuration_files(analysis)
            self._detect_source_code(analysis)
            self._detect_frameworks(analysis)

            # Calculate final confidence score
            analysis.confidence_score = calculate_project_score(analysis)

            # Determine primary project type
            analysis.project_type = self._determine_primary_type(analysis)

            logger.info(
                "project_detection_completed",
                project_path=project_path,
                project_type=analysis.project_type.value,
                confidence=analysis.confidence_score
            )

            return analysis

        except Exception as e:
            logger.error("project_detection_failed", project_path=project_path, error=str(e))
            raise

    def _analyze_directory_structure(self, analysis: ProjectAnalysis) -> None:
        """Analyze basic directory structure and statistics."""
        path = Path(analysis.project_path)

        try:
            items = list(path.iterdir())
            analysis.file_count = len([item for item in items if item.is_file()])
            analysis.directory_count = len([item for item in items if item.is_dir()])
            analysis.is_empty = len(items) == 0

        except PermissionError:
            logger.warning("directory_access_denied", path=analysis.project_path)
            analysis.is_empty = True

    def _detect_git_repository(self, analysis: ProjectAnalysis) -> None:
        """Detect if directory is a git repository."""
        git_dir = Path(analysis.project_path) / ".git"
        analysis.has_git_repo = git_dir.exists() and git_dir.is_dir()

        if analysis.has_git_repo:
            analysis.indicators.append(ProjectIndicator(
                name="git_repository",
                value=True,
                weight=0.5
            ))

    def _detect_documentation(self, analysis: ProjectAnalysis) -> None:
        """Detect documentation presence."""
        path = Path(analysis.project_path)

        for pattern in self.DOC_PATTERNS:
            # Check for files
            if (path / pattern).exists():
                analysis.has_documentation = True
                analysis.indicators.append(ProjectIndicator(
                    name="documentation",
                    value=pattern,
                    weight=0.3
                ))
                break

            # Check for directories
            doc_dir = path / pattern
            if doc_dir.exists() and doc_dir.is_dir():
                analysis.has_documentation = True
                analysis.indicators.append(ProjectIndicator(
                    name="documentation_dir",
                    value=pattern,
                    weight=0.4
                ))
                break

    def _detect_configuration_files(self, analysis: ProjectAnalysis) -> None:
        """Detect and analyze configuration files."""
        path = Path(analysis.project_path)

        for config_file, config_info in self.CONFIG_PATTERNS.items():
            config_path = path / config_file

            # Check exact file match
            if config_path.exists() and config_path.is_file():
                analysis.indicators.append(ProjectIndicator(
                    name="config_file",
                    value=config_file,
                    weight=config_info["weight"]
                ))

                # Add to detected sets
                project_type = config_info["type"]
                analysis.detected_languages.add(project_type.value)
                if "manager" in config_info:
                    analysis.package_managers.add(config_info["manager"])

            # Check for patterns (e.g., *.csproj files)
            elif "*" in config_file:
                pattern = config_file.replace("*", "*")
                matching_files = list(path.glob(pattern))
                if matching_files:
                    analysis.indicators.append(ProjectIndicator(
                        name="config_pattern",
                        value=f"{len(matching_files)} {config_file} files",
                        weight=config_info["weight"] * 0.8  # Slightly lower weight for patterns
                    ))

                    project_type = config_info["type"]
                    analysis.detected_languages.add(project_type.value)
                    if "manager" in config_info:
                        analysis.package_managers.add(config_info["manager"])

    def _detect_source_code(self, analysis: ProjectAnalysis) -> None:
        """Detect source code files and determine languages."""
        path = Path(analysis.project_path)
        language_counts: Dict[str, int] = {}

        try:
            for file_path in path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    ext = file_path.suffix.lower()
                    if ext in self.SOURCE_EXTENSIONS:
                        language = self.SOURCE_EXTENSIONS[ext]
                        language_counts[language] = language_counts.get(language, 0) + 1
                        analysis.detected_languages.add(language)

            # Add indicators for dominant languages
            total_files = sum(language_counts.values())
            if total_files > 0:
                for language, count in language_counts.items():
                    if count / total_files > 0.1:  # At least 10% of files
                        analysis.indicators.append(ProjectIndicator(
                            name="source_language",
                            value=f"{language} ({count} files)",
                            weight=min(0.6, count / total_files)
                        ))

        except PermissionError:
            logger.warning("source_code_scan_permission_denied", path=analysis.project_path)

    def _detect_frameworks(self, analysis: ProjectAnalysis) -> None:
        """Detect specific frameworks based on file patterns."""
        path = Path(analysis.project_path)

        for framework, framework_info in self.FRAMEWORK_PATTERNS.items():
            framework_score = 0.0

            # Check files
            for file_pattern in framework_info.get("files", []):
                if (path / file_pattern).exists():
                    framework_score += framework_info["weight"] * 0.5

            # Check directories
            for dir_pattern in framework_info.get("dirs", []):
                if (path / dir_pattern).exists() and (path / dir_pattern).is_dir():
                    framework_score += framework_info["weight"] * 0.3

            # Check content patterns (simplified - would require file reading)
            for content_pattern in framework_info.get("content_patterns", []):
                # For now, just check if pattern appears in any filename
                matching_files = list(path.rglob(f"*{content_pattern}*"))
                if matching_files:
                    framework_score += framework_info["weight"] * 0.2

            if framework_score >= 0.3:  # Minimum threshold
                analysis.indicators.append(ProjectIndicator(
                    name="framework",
                    value=framework,
                    weight=framework_score
                ))
                analysis.detected_frameworks.add(framework)

    def _determine_primary_type(self, analysis: ProjectAnalysis) -> ProjectType:
        """Determine the primary project type from all indicators."""
        type_scores: Dict[ProjectType, float] = {}

        # Score based on indicators
        for indicator in analysis.indicators:
            if "config_file" in indicator.name or "config_pattern" in indicator.name:
                config_value = str(indicator.value)
                for config_file, config_info in self.CONFIG_PATTERNS.items():
                    if config_file in config_value:
                        project_type = config_info["type"]
                        type_scores[project_type] = type_scores.get(project_type, 0) + indicator.weight

        # Score based on detected frameworks
        for framework in analysis.detected_frameworks:
            framework_type = ProjectType(framework) if hasattr(ProjectType, framework.upper()) else ProjectType.UNKNOWN
            if framework_type != ProjectType.UNKNOWN:
                type_scores[framework_type] = type_scores.get(framework_type, 0) + 0.8

        # Score based on source code languages
        for language in analysis.detected_languages:
            try:
                lang_type = ProjectType(language)
                type_scores[lang_type] = type_scores.get(lang_type, 0) + 0.4
            except ValueError:
                pass  # Language not in ProjectType enum

        # Determine primary type
        if type_scores:
            primary_type = max(type_scores, key=type_scores.get)
            return primary_type

        # Fallback heuristics
        if analysis.is_empty:
            return ProjectType.EMPTY
        if len(analysis.detected_languages) > 2:
            return ProjectType.MIXED

        return ProjectType.UNKNOWN


def calculate_project_score(analysis: ProjectAnalysis) -> float:
    """
    Calculate confidence score for project detection.

    Args:
        analysis: Project analysis results

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if analysis.is_empty:
        return 1.0

    # Base score from indicators
    indicator_score = sum(ind.weight * ind.confidence for ind in analysis.indicators)

    # Normalize by maximum possible score
    max_possible_score = len(analysis.indicators) if analysis.indicators else 1
    normalized_score = indicator_score / max_possible_score if max_possible_score > 0 else 0

    # Bonus factors
    bonus_score = 0.0

    # Git repository bonus
    if analysis.has_git_repo:
        bonus_score += 0.1

    # Documentation bonus
    if analysis.has_documentation:
        bonus_score += 0.1

    # File count bonus (more files = more confidence)
    if analysis.file_count > 10:
        bonus_score += 0.1
    elif analysis.file_count > 5:
        bonus_score += 0.05

    # Framework detection bonus
    if analysis.detected_frameworks:
        bonus_score += 0.15

    # Final score with bonus, capped at 1.0
    final_score = min(1.0, normalized_score + bonus_score)

    return round(final_score, 3)


# Convenience functions
def detect_project_type(project_path: str, min_confidence: float = 0.3) -> ProjectAnalysis:
    """
    Convenience function for project type detection.

    Args:
        project_path: Path to the project directory
        min_confidence: Minimum confidence threshold

    Returns:
        ProjectAnalysis with detection results
    """
    detector = ProjectDetector(min_confidence)
    return detector.detect_project_type(project_path)


def calculate_project_score(indicators: Dict[str, Any]) -> float:
    """
    Legacy function for backward compatibility.

    Args:
        indicators: Dictionary of project indicators

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Convert legacy format to ProjectAnalysis
    analysis = ProjectAnalysis(
        project_path=indicators.get("project_path", ""),
        project_type=ProjectType.UNKNOWN
    )

    # Convert indicators
    for name, value in indicators.items():
        if name in ["has_git_repo", "has_documentation"]:
            weight = 0.5 if value else 0
        else:
            weight = 0.3

        analysis.indicators.append(ProjectIndicator(
            name=name,
            value=value,
            weight=weight
        ))

    return calculate_project_score(analysis)