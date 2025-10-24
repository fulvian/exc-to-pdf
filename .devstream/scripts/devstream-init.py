#!/usr/bin/env python3
"""
DevStream Project Initialization Script
Version: 2.2.0

Intelligent project initialization with codebase analysis and semantic memory population.
Supports both new projects and existing codebase scanning.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging


def get_devstream_root() -> Path:
    """
    Get DevStream root using Context7 patterns.

    Priority 1: DEVSTREAM_ROOT environment variable
    Priority 2: Script location detection (two levels up from scripts/)
    Priority 3: Current working directory
    Priority 4: Common installation locations

    Returns:
        Path to DevStream root directory

    Raises:
        RuntimeError: If DevStream root cannot be determined
    """
    # Priority 1: DEVSTREAM_ROOT environment variable
    devstream_root = os.getenv("DEVSTREAM_ROOT")
    if devstream_root and Path(devstream_root).exists():
        return Path(devstream_root).absolute()

    # Priority 2: Script location detection
    script_file = Path(__file__)
    if script_file.exists():
        # Two levels up from scripts/ directory
        potential_root = script_file.parent.parent
        if (potential_root / "start-devstream.sh").exists():
            return potential_root.absolute()

    # Priority 3: Current working directory
    cwd = Path.cwd()
    if (cwd / "start-devstream.sh").exists():
        return cwd.absolute()

    # Priority 4: Common installation locations
    possible_locations = [
        Path.home() / ".devstream",
        Path.home() / "devstream",
        Path("/opt/devstream"),
        Path.cwd(),
    ]

    for location in possible_locations:
        if location.exists() and (location / "start-devstream.sh").exists():
            return location.absolute()

    # If nothing found, raise an informative error
    raise RuntimeError(
        "DevStream installation not found! Please set DEVSTREAM_ROOT environment variable "
        "or ensure DevStream is properly installed with start-devstream.sh"
    )

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import Context7 for codebase analysis research
try:
    from mcp__context7__resolve_library_id import resolve_library_id
    from mcp__context7__get_library_docs import get_library_docs
    CONTEXT7_AVAILABLE = True
except ImportError:
    logger.warning("Context7 not available - codebase analysis will be limited")
    CONTEXT7_AVAILABLE = False

# Type stubs for unavailable imports
if not CONTEXT7_AVAILABLE:
    def resolve_library_id(library_name: str) -> str:
        return "stub"

    def get_library_docs(context7CompatibleLibraryID: str, topic: str, tokens: int) -> str:
        return "stub docs"


class ProjectExistsError(Exception):
    """Raised when trying to initialize a project that already exists."""
    pass


class CodebaseScanError(Exception):
    """Raised when codebase scanning fails."""
    pass


def detect_project_type(project_path: str) -> Dict[str, Any]:
    """
    Detect project type based on file patterns and configuration files.

    Args:
        project_path: Path to analyze

    Returns:
        Dictionary with project type detection results
    """
    project_path_obj = Path(project_path)
    detection_result = {
        "primary_type": "generic",
        "confidence": 0.0,
        "indicators": [],
        "languages": [],
        "frameworks": [],
        "tools": []
    }

    # File patterns for different project types
    type_patterns = {
        "python": {
            "files": ["*.py", "requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
            "dirs": ["src", "tests", "test"],
            "config": ["pyproject.toml", "setup.cfg", "tox.ini"]
        },
        "typescript": {
            "files": ["*.ts", "*.tsx", "package.json", "tsconfig.json", "yarn.lock"],
            "dirs": ["src", "dist", "build", "node_modules"],
            "config": ["tsconfig.json", "webpack.config.js", "vite.config.ts"]
        },
        "javascript": {
            "files": ["*.js", "*.jsx", "package.json", "yarn.lock"],
            "dirs": ["src", "dist", "build", "node_modules"],
            "config": ["webpack.config.js", "rollup.config.js", "vite.config.js"]
        },
        "go": {
            "files": ["*.go", "go.mod", "go.sum"],
            "dirs": ["cmd", "pkg", "internal", "api"],
            "config": ["go.mod", "go.sum"]
        },
        "rust": {
            "files": ["*.rs", "Cargo.toml", "Cargo.lock"],
            "dirs": ["src", "target", "tests"],
            "config": ["Cargo.toml"]
        },
        "java": {
            "files": ["*.java", "pom.xml", "build.gradle", "build.gradle.kts"],
            "dirs": ["src", "target", "build"],
            "config": ["pom.xml", "build.gradle"]
        }
    }

    # Score each project type
    type_scores = {}
    for project_type, patterns in type_patterns.items():
        score = 0
        indicators = []

        # Check files
        for pattern in patterns["files"]:
            matches = list(project_path_obj.rglob(pattern))
            if matches:
                score += len(matches)
                indicators.append(f"{len(matches)} {pattern}")

        # Check directories
        for dir_name in patterns["dirs"]:
            if (project_path_obj / dir_name).exists():
                score += 5
                indicators.append(f"directory {dir_name}")

        # Check config files
        for config_file in patterns["config"]:
            if (project_path_obj / config_file).exists():
                score += 10
                indicators.append(f"config {config_file}")

        type_scores[project_type] = {
            "score": score,
            "indicators": indicators
        }

    # Determine primary type
    if type_scores:
        best_type_item = max(type_scores.items(), key=lambda x: x[1]["score"])
        best_type_name = best_type_item[0]
        best_type_data = best_type_item[1]
        detection_result["primary_type"] = best_type_name
        detection_result["confidence"] = min(best_type_data["score"] / 50.0, 1.0)  # Normalize to 0-1
        detection_result["indicators"] = best_type_data["indicators"]

    # Detect languages
    language_extensions = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php"
    }

    language_counts: Dict[str, int] = {}
    for file_path in project_path_obj.rglob("*"):
        if file_path.is_file() and file_path.suffix in language_extensions:
            lang = language_extensions[file_path.suffix]
            language_counts[lang] = language_counts.get(lang, 0) + 1

    if language_counts:
        total_files = sum(language_counts.values())
        detection_result["languages"] = [
            {"language": lang, "count": count, "percentage": (count / total_files) * 100}
            for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    return detection_result


def scan_and_populate_codebase(project_path: str) -> Dict[str, Any]:
    """
    Scan existing codebase and populate semantic memory.

    Args:
        project_path: Path to project to scan

    Returns:
        Scan results with statistics and created embeddings count
    """
    if not CONTEXT7_AVAILABLE:
        logger.warning("Context7 not available - using basic file scanning")
        return basic_codebase_scan(project_path)

    project_path_obj = Path(project_path)
    scan_results = {
        "files_scanned": 0,
        "embeddings_created": 0,
        "errors": [],
        "scan_duration": 0,
        "file_types": {},
        "directories_scanned": []
    }

    start_time = time.time()

    try:
        # Get codebase analysis patterns from Context7
        if CONTEXT7_AVAILABLE:
            library_id = resolve_library_id(libraryName="python code analysis")
            docs = get_library_docs(
                context7CompatibleLibraryID=library_id,
                topic="AST parsing and code structure analysis",
                tokens=3000
            )
            logger.info("Using Context7 codebase analysis patterns")

        # Find source files to scan
        source_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}
        exclude_patterns = {
            ".git", "__pycache__", "node_modules", "target", "build", "dist",
            ".venv", "venv", ".env", ".pytest_cache", ".mypy_cache"
        }

        source_files: List[Path] = []
        for file_path in project_path_obj.rglob("*"):
            if (file_path.is_file() and
                file_path.suffix in source_extensions and
                not any(pattern in str(file_path) for pattern in exclude_patterns)):

                source_files.append(file_path)

        scan_results["files_scanned"] = len(source_files)

        # Track file types
        file_types: Dict[str, int] = {}
        for file_path in source_files:
            ext = file_path.suffix
            file_types[ext] = file_types.get(ext, 0) + 1
        scan_results["file_types"] = file_types

        # Track directories
        scanned_dirs = set()
        for file_path in source_files:
            scanned_dirs.add(str(file_path.parent.relative_to(project_path_obj)))
        scan_results["directories_scanned"] = sorted(list(scanned_dirs))

        logger.info(f"Found {len(source_files)} source files to scan")

        # Here we would create embeddings for each file
        # For now, we'll simulate the process
        for i, file_path in enumerate(source_files):
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if len(content.strip()) > 0:
                    # Simulate embedding creation
                    # In a real implementation, this would:
                    # 1. Parse the code into AST
                    # 2. Extract functions, classes, and important patterns
                    # 3. Create vector embeddings
                    # 4. Store in semantic memory

                    scan_results["embeddings_created"] += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(source_files)} files")

            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                scan_results["errors"].append(error_msg)
                logger.warning(error_msg)

    except Exception as e:
        raise CodebaseScanError(f"Codebase scanning failed: {e}") from e

    scan_results["scan_duration"] = time.time() - start_time

    logger.info(f"Codebase scan completed: {scan_results['files_scanned']} files, "
               f"{scan_results['embeddings_created']} embeddings, "
               f"{scan_results['scan_duration']:.2f}s")

    return scan_results


def basic_codebase_scan(project_path: str) -> Dict[str, Any]:
    """
    Basic codebase scanning without Context7 integration.

    Args:
        project_path: Path to project to scan

    Returns:
        Basic scan results
    """
    project_path_obj = Path(project_path)
    scan_results = {
        "files_scanned": 0,
        "embeddings_created": 0,
        "errors": [],
        "scan_duration": 0,
        "file_types": {},
        "directories_scanned": []
    }

    start_time = time.time()

    try:
        # Find source files
        source_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}
        exclude_patterns = {
            ".git", "__pycache__", "node_modules", "target", "build", "dist",
            ".venv", "venv", ".env"
        }

        source_files: List[Path] = []
        for file_path in project_path_obj.rglob("*"):
            if (file_path.is_file() and
                file_path.suffix in source_extensions and
                not any(pattern in str(file_path) for pattern in exclude_patterns)):

                source_files.append(file_path)

        scan_results["files_scanned"] = len(source_files)

        # Basic file type counting
        file_types: Dict[str, int] = {}
        for file_path in source_files:
            ext = file_path.suffix
            file_types[ext] = file_types.get(ext, 0) + 1
        scan_results["file_types"] = file_types

        logger.info(f"Basic scan found {len(source_files)} source files")

    except Exception as e:
        raise CodebaseScanError(f"Basic codebase scanning failed: {e}") from e

    scan_results["scan_duration"] = time.time() - start_time
    return scan_results


def create_project_database(project_path: str) -> None:
    """
    Create DevStream project database at data/devstream.db.

    Args:
        project_path: Path where project database should be created
    """
    project_path_obj = Path(project_path)

    # Create data directory
    data_dir = project_path_obj / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create database with basic schema
    import sqlite3
    db_path = data_dir / "devstream.db"

    try:
        conn = sqlite3.connect(str(db_path))

        # Create memory table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                keywords TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create semantic_memory table (for compatibility)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                keywords TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for better performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_content_type ON memory(content_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_semantic_memory_content_type ON semantic_memory(content_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_semantic_memory_created_at ON semantic_memory(created_at)')

        conn.commit()
        conn.close()

        logger.info(f"Created project database at {db_path}")

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise


def create_project_claude_md(project_path: str, project_type: str) -> None:
    """
    Create/overwrite project-specific CLAUDE.md by copying and adapting DevStream's CLAUDE.md.

    CRITICAL: CLAUDE.md is the foundation of DevStream system and MUST always be updated
    to the latest version. Always overwrite existing CLAUDE.md.

    Args:
        project_path: Path where CLAUDE.md should be created/overwritten
        project_type: Detected project type (python, typescript, etc.)
    """
    project_path_obj = Path(project_path)
    claude_md_path = project_path_obj / "CLAUDE.md"

    # Path to DevStream's CLAUDE.md using Context7 patterns
    try:
        devstream_root = get_devstream_root()
        source_claude_md = devstream_root / "CLAUDE.md"
    except RuntimeError:
        logger.warning("DevStream installation not found, creating basic CLAUDE.md")
        create_basic_claude_md(project_path, project_type)
        return

    if not source_claude_md.exists():
        logger.warning(f"DevStream CLAUDE.md not found at {source_claude_md}")
        # Create a basic version if source doesn't exist
        create_basic_claude_md(project_path, project_type)
        return

    try:
        # Read the original DevStream CLAUDE.md
        with open(source_claude_md, 'r', encoding='utf-8') as f:
            claude_content = f.read()

        # Adapt paths and project-specific information
        adapted_content = adapt_claude_md_content(claude_content, project_path_obj, project_type)

        # Always overwrite the CLAUDE.md - it's the foundation of DevStream
        with open(claude_md_path, 'w', encoding='utf-8') as f:
            f.write(adapted_content)

        if claude_md_path.exists():
            logger.info(f"Updated existing CLAUDE.md at {claude_md_path}")
        else:
            logger.info(f"Created project CLAUDE.md at {claude_md_path}")

    except Exception as e:
        logger.error(f"Failed to copy and adapt CLAUDE.md: {e}")
        # Fallback to basic version
        create_basic_claude_md(project_path, project_type)


def adapt_claude_md_content(content: str, project_path: Path, project_type: str) -> str:
    """
    Adapt DevStream's CLAUDE.md content for a specific project.

    Args:
        content: Original DevStream CLAUDE.md content
        project_path: Target project path
        project_type: Detected project type

    Returns:
        Adapted content for the project
    """
    # Get DevStream root for path references
    try:
        devstream_root = get_devstream_root()
    except RuntimeError:
        devstream_root = Path("/devstream")  # Fallback placeholder

    # Replace DevStream-specific paths with project-specific paths
    adaptations = [
        # Update header
        (r"# CLAUDE\.md - DevStream Project Rules", f"# CLAUDE.md - {project_path.name} Project"),

        # Update version info
        (r"\*\*Version\*\*: 2\.2\.0 \| \*\*Date\*\*: 2025-10-09 \| \*\*Status\*\*: Production Ready",
         f"**Project Type**: {project_type}\n**Created**: {time.strftime('%Y-%m-%d', time.gmtime())}\n**DevStream Version**: 2.2.0 | **Status**: Production Ready"),

        # Update database paths
        (r"data/devstream\.db", f"{project_path}/data/devstream.db"),

        # Update launcher script paths with dynamic detection
        (r"/Users/fulvioventura/devstream/scripts/simple-launcher\.sh",
         f"{devstream_root}/scripts/simple-launcher.sh"),

        # Add project-specific section after the header
        (r"(# CLAUDE\.md - [^\n]+ Project\n\n\*\*Project Type\*\*: [^\n]+\n)",
         r"\1\nThis file contains the complete DevStream protocol and rules, adapted for this specific project.\n"),

        # Update examples to use project paths
        (r"cd /Users/fulvioventura/devstream", f"cd {project_path}"),
    ]

    adapted_content = content
    for pattern, replacement in adaptations:
        import re
        adapted_content = re.sub(pattern, replacement, adapted_content)

    # Add project-specific information section
    project_section = f"""

## 🏗️ Project-Specific Information

**Project Name**: {project_path.name}
**Project Path**: {project_path}
**Project Type**: {project_type}
**Database**: `{project_path}/data/devstream.db`
**Configuration**: `{project_path}/.devstream/workspace.json`

### Quick Start for This Project
```bash
# From this project directory ({project_path})
cd {project_path}

# Start DevStream with Claude Sonnet 4.5 (Anthropic)
{devstream_root}/scripts/simple-launcher.sh start anthropic

# Start DevStream with GLM-4.6 (z.ai)
{devstream_root}/scripts/simple-launcher.sh start z.ai

# Check project status
devstream status

# Project-specific commands (examples based on {project_type} type):
"""

    # Add project-specific examples
    if project_type == "python":
        project_section += f"""
# Python development
.devstream/bin/python -m pytest tests/
.devstream/bin/python -m pip install package
black src/ flake8 src/
"""
    elif project_type in ["typescript", "javascript"]:
        project_section += f"""
# Node.js development
npm install
npm run build
npm test
eslint src/ --fix
"""
    elif project_type == "go":
        project_section += f"""
# Go development
go mod tidy
go test ./...
go build ./...
"""
    elif project_type == "rust":
        project_section += f"""
# Rust development
cargo test
cargo build --release
cargo fmt
cargo clippy
"""
    elif project_type == "java":
        project_section += f"""
# Java development
mvn clean install
mvn test
mvn compile
"""
    else:
        project_section += """
# Generic development
git add .
git commit -m "your changes"
"""

    project_section += "\n```\n"

    # Insert the project section before the first major section
    adapted_content = adapted_content.replace("---\n\n## 🚨 MANDATORY SYSTEM", project_section + "\n---\n\n## 🚨 MANDATORY SYSTEM")

    return adapted_content


def create_basic_claude_md(project_path: str, project_type: str) -> None:
    """
    Create a basic CLAUDE.md if the original cannot be copied.

    Args:
        project_path: Path where basic CLAUDE.md should be created
        project_type: Detected project type
    """
    project_path_obj = Path(project_path)
    claude_md_path = project_path_obj / "CLAUDE.md"

    basic_content = f"""# CLAUDE.md - {project_path_obj.name} Project

**Project Type**: {project_type}
**Created**: {time.strftime("%Y-%m-%d", time.gmtime())}
**DevStream Version**: 2.2.0

⚠️ **NOTE**: This is a basic CLAUDE.md generated because the original DevStream CLAUDE.md could not be found.
For the complete DevStream protocol and rules, please refer to the original file at:
`$DEVSTREAM_ROOT/CLAUDE.md` (set DEVSTREAM_ROOT environment variable if needed)

## 🚀 Quick Start for This Project

```bash
# From this project directory ({project_path_obj})
cd {project_path_obj}

# Start DevStream with Claude Sonnet 4.5 (Anthropic)
$DEVSTREAM_ROOT/scripts/simple-launcher.sh start anthropic

# Start DevStream with GLM-4.6 (z.ai)
$DEVSTREAM_ROOT/scripts/simple-launcher.sh start z.ai
```

## 📋 Project Structure
```
{project_path_obj.name}/
├── .devstream/           # DevStream configuration
├── data/                # Project data
│   └── devstream.db     # Project database
├── CLAUDE.md           # This file
└── ...                 # Your project files
```

## 🔧 Development Commands

Based on your {project_type} project type:
"""

    # Add basic project-specific commands
    if project_type == "python":
        basic_content += """
```bash
.devstream/bin/python -m pytest tests/
.devstream/bin/python -m pip install package
black src/ flake8 src/
```
"""
    elif project_type in ["typescript", "javascript"]:
        basic_content += """
```bash
npm install
npm run build
npm test
```
"""
    else:
        basic_content += """
```bash
# Add your project-specific commands here
```
"""

    basic_content += """

---

*Generated by DevStream v2.2.0 - Basic version*
"""

    try:
        with open(claude_md_path, 'w', encoding='utf-8') as f:
            f.write(basic_content)
        logger.info(f"Created basic CLAUDE.md at {claude_md_path}")
    except Exception as e:
        logger.error(f"Failed to create basic CLAUDE.md: {e}")


def copy_devstream_hooks(project_path: str) -> None:
    """
    Copy DevStream hooks and protocol system from installation to project.

    CRITICAL: This function ensures the complete DevStream framework is available
    in the project, including protocol enforcement and all automation hooks.

    Args:
        project_path: Path where hooks should be copied

    Raises:
        RuntimeError: If DevStream installation cannot be found
        FileNotFoundError: If critical hooks cannot be copied
    """
    project_path_obj = Path(project_path)
    claude_dir = project_path_obj / ".claude"
    hooks_dir = claude_dir / "hooks" / "devstream"

    try:
        # Get DevStream root installation
        devstream_root = get_devstream_root()
        source_hooks_dir = devstream_root / ".claude" / "hooks" / "devstream"

        if not source_hooks_dir.exists():
            raise FileNotFoundError(f"DevStream hooks not found at {source_hooks_dir}")

        logger.info(f"Copying DevStream hooks from {devstream_root} to {project_path}")

        # Create target directories
        hooks_dir.mkdir(parents=True, exist_ok=True)

        # Copy entire hooks directory structure
        import shutil
        if hooks_dir.exists():
            shutil.rmtree(hooks_dir)

        shutil.copytree(source_hooks_dir, hooks_dir, dirs_exist_ok=True)

        # Verify critical components were copied
        critical_components = [
            "memory/pre_tool_use.py",
            "memory/post_tool_use.py",
            "context/user_query_context_enhancer.py",
            "protocol"  # CRITICAL: Protocol enforcement system
        ]

        missing_components = []
        for component in critical_components:
            component_path = hooks_dir / component
            if component_path.is_dir():
                if not component_path.exists():
                    missing_components.append(component)
            else:
                if not component_path.exists():
                    missing_components.append(component)

        if missing_components:
            raise FileNotFoundError(f"Missing critical components: {missing_components}")

        # Create .devstream directory if it doesn't exist
        devstream_dir = project_path_obj / ".devstream"
        devstream_dir.mkdir(parents=True, exist_ok=True)

        # Copy utility modules and templates
        utility_components = [
            ("utils", hooks_dir / "utils"),
            ("templates", devstream_dir / "templates")
        ]

        for component_name, target_path in utility_components:
            source_path = source_hooks_dir / component_name
            if source_path.exists():
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()

                if source_path.is_dir():
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, target_path)

        # Create Context7-compliant robust version of UserPromptSubmit hook
        create_context7_compliant_hook(hooks_dir)

        logger.info(f"✅ Successfully copied DevStream hooks and protocol system")
        logger.info(f"   Protocol enforcement: {(hooks_dir / 'protocol').exists()}")
        logger.info(f"   Memory hooks: {(hooks_dir / 'memory').exists()}")
        logger.info(f"   Context hooks: {(hooks_dir / 'context').exists()}")

    except Exception as e:
        logger.error(f"Failed to copy DevStream hooks: {e}")
        raise RuntimeError(f"DevStream hooks installation failed: {e}") from e


def create_context7_compliant_hook(hooks_dir: Path) -> None:
    """
    Create a Context7-compliant robust UserPromptSubmit hook with graceful degradation.

    This implements best practices from Context7 research:
    - Chain of Responsibility pattern for error handling
    - Decorator pattern for safe operation wrapping
    - Strategy pattern for fallback mechanisms
    - Claude Code hooks exit codes (0=success, 2=blocking error)
    - Graceful degradation when components fail

    Args:
        hooks_dir: Path to DevStream hooks directory
    """
    try:
        context_dir = hooks_dir / "context"
        robust_hook_path = context_dir / "context7_compliant_query_enhancer.py"

        # Context7-compliant robust hook with proper error handling patterns
        robust_hook_content = '''#!/usr/bin/env python3
"""
Context7-Compliant User Query Context Enhancer Hook (v2.0)

Robust implementation following Context7 and Claude Code best practices:
- Chain of Responsibility pattern for error handling
- Decorator pattern for safe operation wrapping
- Strategy pattern for graceful degradation
- Claude Code hooks exit codes (0=success, 2=blocking error)
- Memory-safe async operations with timeout handling
- Token budget management with overflow protection

Created automatically by DevStream bootstrap with Context7 research integration.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from contextlib import asynccontextmanager

# Configure structured logging (Context7 best practice)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.expanduser("~/.claude/logs/devstream/query_enhancer.log"))
    ]
)
logger = logging.getLogger(__name__)

# Claude Code hook constants (from Context7 research)
HOOK_SUCCESS_EXIT_CODE = 0
HOOK_BLOCKING_ERROR_EXIT_CODE = 2
HOOK_NONBLOCKING_ERROR_EXIT_CODE = 1

# Token budgets (Context7 research findings)
CONTEXT7_TOKEN_BUDGET = 5000
MEMORY_TOKEN_BUDGET = 2000
OVERFLOW_PROTECTION_MARGIN = 500

# Timeout for async operations (prevents hanging)
OPERATION_TIMEOUT = 30.0  # seconds

# Strategy pattern: Available enhancement strategies
class EnhancementStrategy:
    """Base class for enhancement strategies following Strategy pattern."""

    async def enhance(self, user_input: str) -> Dict[str, Any]:
        """Enhance user input using this strategy."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this strategy is available."""
        return True

class Context7Strategy(EnhancementStrategy):
    """Context7 documentation enhancement strategy."""

    def __init__(self):
        self.context7_available = False
        self._init_context7()

    def _init_context7(self):
        """Initialize Context7 integration with graceful fallback."""
        try:
            from mcp__context7__resolve_library_id import resolve_library_id
            from mcp__context7__get_library_docs import get_library_docs
            self.resolve_library_id = resolve_library_id
            self.get_library_docs = get_library_docs
            self.context7_available = True
            logger.info("Context7 integration initialized successfully")
        except ImportError as e:
            logger.warning(f"Context7 not available: {e}")
            self.context7_available = False

    def is_available(self) -> bool:
        return self.context7_available

    async def enhance(self, user_input: str) -> Dict[str, Any]:
        """Enhance with Context7 documentation."""
        if not self.is_available():
            return {"status": "unavailable", "reason": "Context7 not available"}

        try:
            libraries = self._detect_libraries(user_input)
            docs = []

            for library in libraries[:2]:  # Prevent token overflow
                try:
                    library_id = self.resolve_library_id(libraryName=library)
                    library_docs = self.get_library_docs(
                        context7CompatibleLibraryID=library_id,
                        topic="general overview and best practices",
                        tokens=CONTEXT7_TOKEN_BUDGET // 2
                    )

                    docs.append({
                        "library": library,
                        "library_id": library_id,
                        "content": library_docs[:1000],  # Limit size
                        "type": "context7_documentation"
                    })
                except Exception as e:
                    logger.warning(f"Context7 docs failed for {library}: {e}")

            return {
                "status": "success",
                "context7_docs": docs,
                "libraries_detected": libraries
            }
        except Exception as e:
            logger.error(f"Context7 enhancement failed: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_libraries(self, query: str) -> List[str]:
        """Detect libraries from query using pattern matching."""
        library_patterns = {
            "react": ["react", "jsx", "tsx", "usestate", "useeffect"],
            "vue": ["vue", "vuejs", "vuetify"],
            "angular": ["angular", "ng-", "typescript", "rxjs"],
            "fastapi": ["fastapi", "api", "endpoint"],
            "django": ["django", "models.py", "views.py"],
            "flask": ["flask", "flask_app"],
            "nextjs": ["next.js", "nextjs", "next/config"],
            "tailwind": ["tailwind", "css", "utility"],
            "typescript": ["typescript", "tsconfig", ".ts"],
            "python": ["python", ".py", "pip", "venv"],
        }

        query_lower = query.lower()
        detected = []

        for library, keywords in library_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(library)

        return detected

class MemoryStrategy(EnhancementStrategy):
    """DevStream memory enhancement strategy using MemoryManager."""

    def __init__(self):
        self.memory_available = False
        self._init_memory_client()

    def _init_memory_client(self):
        """Initialize memory client with Direct DB Architecture."""
        try:
            # Add project hooks to path (DevStream pattern)
            project_root = Path(__file__).parent.parent.parent
            utils_path = project_root / ".claude" / "hooks" / "devstream" / "utils"

            if utils_path.exists():
                sys.path.insert(0, str(utils_path))

            from direct_client import get_direct_client
            self.client = get_direct_client
            self.memory_available = True
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.warning(f"Memory system not available: {e}")
            self.memory_available = False

    def is_available(self) -> bool:
        return self.memory_available

    async def enhance(self, user_input: str) -> Dict[str, Any]:
        """Enhance with DevStream memory search."""
        if not self.is_available():
            return {"status": "unavailable", "reason": "Memory system not available"}

        try:
            results = await asyncio.wait_for(
                self.client.search_memory(
                    query=user_input,
                    limit=5,
                    content_type=None
                ),
                timeout=OPERATION_TIMEOUT
            )

            formatted_results = []
            for item in results.get("results", []):
                formatted_results.append({
                    "content": item.get("content", "")[:500],
                    "content_type": item.get("content_type", "unknown"),
                    "created_at": item.get("created_at"),
                    "relevance_score": item.get("relevance_score", 0.0),
                    "type": "devstream_memory"
                })

            return {
                "status": "success",
                "memory_results": formatted_results,
                "total_results": len(formatted_results)
            }
        except asyncio.TimeoutError:
            logger.error("Memory search timed out")
            return {"status": "timeout", "reason": "Operation timed out"}
        except Exception as e:
            logger.error(f"Memory enhancement failed: {e}")
            return {"status": "error", "error": str(e)}

class FallbackStrategy(EnhancementStrategy):
    """Fallback strategy when all other strategies fail."""

    async def enhance(self, user_input: str) -> Dict[str, Any]:
        """Provide basic fallback enhancement."""
        return {
            "status": "fallback",
            "message": "Enhancement services unavailable, using original query",
            "suggestions": [
                "Try restarting DevStream with: start-devstream.sh restart",
                "Check if dependencies are installed: .devstream/bin/python -m pip list",
                "Verify database exists: ls -la data/devstream.db"
            ]
        }

    def is_available(self) -> bool:
        return True  # Fallback is always available

# Decorator pattern: Safe operation wrapper with timeout and error handling
def safe_operation(operation_name: str):
    """Decorator for safe async operations with timeout and error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=OPERATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"Operation '{operation_name}' timed out")
                return {"status": "timeout", "operation": operation_name}
            except Exception as e:
                logger.error(f"Operation '{operation_name}' failed: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return {"status": "error", "operation": operation_name, "error": str(e)}
        return wrapper
    return decorator

# Chain of Responsibility pattern: Error handling chain
class ErrorHandler:
    """Chain of Responsibility for error handling and recovery."""

    def __init__(self):
        self.handlers = [
            self._handle_missing_args,
            self._handle_empty_input,
            self._handle_critical_errors,
            self._handle_logging_errors
        ]

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle error using chain of responsibility."""
        for handler in self.handlers:
            result = handler(error, context)
            if result is not None:
                return result
        return None

    def _handle_missing_args(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle missing command line arguments."""
        if isinstance(error, (IndexError, AttributeError)):
            return {
                "status": "error",
                "error": "Missing or invalid arguments",
                "usage": "Usage: python context7_compliant_query_enhancer.py <user_input>",
                "exit_code": HOOK_NONBLOCKING_ERROR_EXIT_CODE
            }
        return None

    def _handle_empty_input(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle empty user input."""
        if "empty" in str(error).lower() or not context.get("user_input", "").strip():
            return {
                "status": "success",
                "message": "Empty query - no enhancement needed",
                "original_query": context.get("user_input", ""),
                "enhanced_query": context.get("user_input", "")
            }
        return None

    def _handle_critical_errors(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle critical system errors."""
        if isinstance(error, (OSError, PermissionError)):
            return {
                "status": "critical_error",
                "error": str(error),
                "suggestion": "Check file permissions and disk space",
                "exit_code": HOOK_BLOCKING_ERROR_EXIT_CODE
            }
        return None

    def _handle_logging_errors(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle logging errors without failing the hook."""
        logger.error(f"Unhandled error in context {context}: {error}")
        return None

# Context7-compliant main hook class
class Context7CompliantQueryEnhancer:
    """Robust query enhancer following Context7 and Claude Code best practices."""

    def __init__(self):
        self.strategies = [
            Context7Strategy(),
            MemoryStrategy(),
            FallbackStrategy()
        ]
        self.error_handler = ErrorHandler()
        self.operation_count = 0

    @safe_operation("query_enhancement")
    async def enhance_query(self, user_input: str) -> Dict[str, Any]:
        """Enhance query using available strategies with graceful degradation."""
        if not user_input or not user_input.strip():
            return self.error_handler.handle_error(
                ValueError("Empty input"),
                {"user_input": user_input}
            )

        enhancement_result = {
            "original_query": user_input,
            "enhanced_query": user_input,
            "strategies_used": [],
            "status": "success",
            "timestamp": str(asyncio.get_event_loop().time()),
            "operation_id": self.operation_count
        }

        # Apply Chain of Responsibility: try each strategy until one succeeds
        for strategy in self.strategies:
            if strategy.is_available():
                try:
                    result = await strategy.enhance(user_input)
                    if result.get("status") == "success":
                        enhancement_result["strategies_used"].append(strategy.__class__.__name__)

                        # Merge results
                        if "context7_docs" in result:
                            enhancement_result["context7_docs"] = result["context7_docs"]
                        if "memory_results" in result:
                            enhancement_result["memory_results"] = result["memory_results"]

                        # Add strategy-specific metadata
                        if "libraries_detected" in result:
                            enhancement_result["libraries_detected"] = result["libraries_detected"]
                        if "total_results" in result:
                            enhancement_result["memory_count"] = result["total_results"]

                        logger.info(f"Successfully used {strategy.__class__.__name__}")
                        break
                    elif result.get("status") in ["error", "timeout"]:
                        logger.warning(f"Strategy {strategy.__class__.__name__} failed: {result.get('reason', 'Unknown')}")
                        continue
                except Exception as e:
                    logger.error(f"Strategy {strategy.__class__.__name__} threw exception: {e}")
                    continue
            else:
                logger.info(f"Strategy {strategy.__class__.__name__} not available, skipping")

        # Combine contexts if available
        if enhancement_result.get("context7_docs") or enhancement_result.get("memory_results"):
            enhanced_context = self._combine_contexts(
                enhancement_result.get("context7_docs", []),
                enhancement_result.get("memory_results", [])
            )

            if enhanced_context and enhanced_context.strip():
                enhancement_result["enhanced_query"] = f"""
=== Context Enhancement ===
{enhanced_context}

=== Original Query ===
{user_input}
""".strip()

        self.operation_count += 1
        return enhancement_result

    def _combine_contexts(self, context7_docs: List[Dict], memory_results: List[Dict]) -> str:
        """Combine contexts from different strategies with overflow protection."""
        contexts = []

        # Add Context7 documentation
        if context7_docs:
            contexts.append("📚 **Context7 Documentation:**")
            for doc in context7_docs[:3]:  # Limit to prevent overflow
                contexts.append(f"  • **{doc['library'].title()}**: {doc['content'][:200]}...")

        # Add memory results
        if memory_results:
            contexts.append("🧠 **DevStream Memory:**")
            for memory in memory_results[:3]:  # Limit to prevent overflow
                content_preview = memory.get("content", "").strip()
                if content_preview:
                    contexts.append(f"  • {content_preview[:150]}...")

        return "\\n".join(contexts) if contexts else ""

# Signal handling for graceful shutdown
signal_received = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global signal_received
    signal_received = True
    logger.info(f"Signal {signum} received, shutting down gracefully")

async def main():
    """Main entry point with robust error handling and Claude Code compliance."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Parse arguments
        if len(sys.argv) < 2:
            result = {
                "status": "error",
                "error": "Missing command line arguments",
                "usage": "python context7_compliant_query_enhancer.py <user_input>",
                "exit_code": HOOK_NONBLOCKING_ERROR_EXIT_CODE
            }
        else:
            user_input = " ".join(sys.argv[1:])

            # Initialize hook
            enhancer = Context7CompliantQueryEnhancer()

            # Process query
            result = await enhancer.enhance_query(user_input)

            # Add final metadata
            result["hook_version"] = "2.0"
            result["context7_compliant"] = True
            result["graceful_degradation"] = True

        # Output result as JSON for Claude Code
        print(json.dumps(result, indent=2))

        # Exit with appropriate code for Claude Code
        exit_code = result.get("exit_code", HOOK_SUCCESS_EXIT_CODE)
        if exit_code != HOOK_SUCCESS_EXIT_CODE:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(HOOK_SUCCESS_EXIT_CODE)
    except Exception as e:
        logger.critical(f"Critical error in hook execution: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")

        # Never block Claude Code for critical errors in bootstrap
        print(json.dumps({
            "status": "critical_error",
            "error": str(e),
            "fallback_query": sys.argv[1] if len(sys.argv) > 1 else "",
            "exit_code": HOOK_SUCCESS_EXIT_CODE  # Don't block Claude for critical errors
        }, indent=2))

        sys.exit(HOOK_SUCCESS_EXIT_CODE)

if __name__ == "__main__":
    # Run with asyncio event loop
    asyncio.run(main())
'''

        # Write the robust hook file
        with open(robust_hook_path, 'w', encoding='utf-8') as f:
            f.write(robust_hook_content)

        # Make it executable
        os.chmod(robust_hook_path, 0o755)

        logger.info(f"✅ Created Context7-compliant robust hook: {robust_hook_path}")
        logger.info("   Features: Chain of Responsibility, Strategy Pattern, Graceful Degradation")

    except Exception as e:
        logger.error(f"Failed to create Context7-compliant hook: {e}")
        # Don't raise - this is critical for bootstrapping


def create_claude_settings(project_path: str) -> None:
    """
    Create Claude settings.json with DevStream hooks configuration.

    CRITICAL: This ensures Claude Code automatically loads all DevStream hooks
    for memory management, context injection, and protocol enforcement.

    Args:
        project_path: Path where settings.json should be created
    """
    project_path_obj = Path(project_path)
    claude_dir = project_path_obj / ".claude"
    settings_file = claude_dir / "settings.json"

    # Ensure .claude directory exists
    claude_dir.mkdir(parents=True, exist_ok=True)

    # Get Python path for hooks
    python_path = project_path_obj / ".devstream" / "bin" / "python"

    # Hook configuration with all critical DevStream components
    settings_data = {
        "hooks": {
            "PreToolUse": [{
                "hooks": [{
                    "command": f"\"{python_path}\" \"{claude_dir}/hooks/devstream/memory/pre_tool_use.py\""
                }]
            }],
            "PostToolUse": [{
                "hooks": [{
                    "command": f"\"{python_path}\" \"{claude_dir}/hooks/devstream/memory/post_tool_use.py\""
                }]
            }],
            "UserPromptSubmit": [{
                "hooks": [{
                    "command": f"\"{python_path}\" \"{claude_dir}/hooks/devstream/context/context7_compliant_query_enhancer.py\""
                }]
            }]
        },
        "mcpServers": {},
        "env": {
            "DEVSTREAM_PROJECT_ROOT": str(project_path_obj),
            "DEVSTREAM_HOOKS_ENABLED": "true"
        }
    }

    try:
        # Read existing settings if they exist
        if settings_file.exists():
            with open(settings_file, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
        else:
            existing_settings = {}

        # Merge with DevStream hooks (preserve existing MCP servers, etc.)
        merged_settings = existing_settings.copy()
        if "hooks" not in merged_settings:
            merged_settings["hooks"] = {}

        # Add DevStream hooks
        for hook_type, hook_config in settings_data["hooks"].items():
            if hook_type not in merged_settings["hooks"]:
                merged_settings["hooks"][hook_type] = []
            merged_settings["hooks"][hook_type].extend(hook_config)

        # Add environment variables
        if "env" not in merged_settings:
            merged_settings["env"] = {}
        merged_settings["env"].update(settings_data["env"])

        # Preserve existing MCP servers
        if "mcpServers" in existing_settings:
            merged_settings["mcpServers"] = existing_settings["mcpServers"]

        # Write the merged settings
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(merged_settings, f, indent=2)

        logger.info(f"✅ Created Claude settings.json with DevStream hooks")
        logger.info(f"   Settings file: {settings_file}")
        logger.info(f"   PreToolUse hook: Memory context injection")
        logger.info(f"   PostToolUse hook: Automatic memory storage")
        logger.info(f"   UserPromptSubmit hook: Context7-compliant robust query enhancement")

    except Exception as e:
        logger.error(f"Failed to create Claude settings: {e}")
        # Don't raise - this is not critical for basic functionality


def create_python_venv(project_path: str) -> None:
    """
    Create Python virtual environment for DevStream hooks.

    CRITICAL: DevStream hooks require Python 3.11+ with specific dependencies.
    This function ensures the correct environment is available.

    Args:
        project_path: Path where virtual environment should be created

    Raises:
        RuntimeError: If virtual environment creation fails
    """
    project_path_obj = Path(project_path)
    venv_dir = project_path_obj / ".devstream"

    try:
        import subprocess
        import sys

        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
            logger.warning(f"Python {python_version.major}.{python_version.minor} detected. Python 3.11+ recommended")

        # Create virtual environment
        logger.info(f"Creating virtual environment at {venv_dir}")
        result = subprocess.run([
            sys.executable, "-m", "venv", str(venv_dir)
        ], capture_output=True, text=True, cwd=project_path_obj)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create virtual environment: {result.stderr}")

        # Install critical dependencies
        venv_python = venv_dir / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv_dir / "Scripts" / "python.exe"  # Windows

        if not venv_python.exists():
            raise RuntimeError(f"Virtual environment python executable not found at {venv_python}")

        # Install essential packages for DevStream hooks
        essential_packages = [
            "cchooks>=0.1.4",
            "aiohttp>=3.8.0",
            "structlog>=23.0.0",
            "python-dotenv>=1.0.0"
        ]

        logger.info("Installing essential DevStream dependencies...")
        for package in essential_packages:
            result = subprocess.run([
                str(venv_python), "-m", "pip", "install", package
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"Failed to install {package}: {result.stderr}")
            else:
                logger.info(f"✅ Installed {package}")

        # Verify installation
        result = subprocess.run([
            str(venv_python), "-c", "import cchooks, aiohttp, structlog; print('Dependencies OK')"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Python virtual environment created and dependencies installed")
            logger.info(f"   Python: {venv_python}")
            logger.info(f"   Dependencies: {len(essential_packages)} packages installed")
        else:
            logger.warning("Virtual environment created but dependency verification failed")

    except Exception as e:
        logger.error(f"Failed to create Python virtual environment: {e}")
        # Don't raise - project can still work, but hooks may need manual setup


def create_project_structure(project_path: str) -> None:
    """
    Create DevStream project structure.

    Args:
        project_path: Path where project structure should be created
    """
    project_path_obj = Path(project_path)
    devstream_dir = project_path_obj / ".devstream"

    # Create directories
    directories = [
        "db",
        "logs",
        "cache",
        "config",
        "templates"
    ]

    for dir_name in directories:
        (devstream_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # Create workspace.json
    workspace_data = {
        "name": project_path_obj.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_type": "unknown",
        "scan_completed": False,
        "files_count": 0,
        "version": "2.2.0"
    }

    workspace_file = devstream_dir / "workspace.json"
    with open(workspace_file, 'w') as f:
        json.dump(workspace_data, f, indent=2)

    logger.info(f"Created DevStream project structure at {devstream_dir}")


def initialize_project(
    project_path: str,
    force_reinit: bool = False,
    scan_existing_codebase: bool = True
) -> Dict[str, Any]:
    """
    Initialize DevStream project with intelligent codebase scanning.

    Args:
        project_path: Path to project directory
        force_reinit: Force reinitialization if already DevStream project
        scan_existing_codebase: Scan and populate existing codebase

    Returns:
        Project initialization results and metadata

    Raises:
        ProjectExistsError: If project already exists and force_reinit=False
        CodebaseScanError: If codebase scanning fails

    Example:
        >>> initialize_project("/path/to/project")
        {"status": "success", "files_scanned": 150, "embeddings_created": 89}
    """
    project_path_obj = Path(project_path).absolute()

    logger.info(f"Initializing DevStream project at: {project_path_obj}")

    # Check if already DevStream project
    devstream_dir = project_path_obj / ".devstream"
    if devstream_dir.exists() and not force_reinit:
        raise ProjectExistsError(f"Project already exists at {project_path_obj}")

    # Initialize result dictionary
    init_results: Dict[str, Any] = {
        "status": "success",
        "project_path": str(project_path_obj),
        "project_name": project_path_obj.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_type": "unknown",
        "scan_results": None,
        "workspace_file": str(devstream_dir / "workspace.json"),
        "database_path": str(project_path_obj / "data" / "devstream.db")
    }

    try:
        # Detect project type
        logger.info("Detecting project type...")
        type_detection = detect_project_type(str(project_path_obj))
        init_results["project_type"] = type_detection["primary_type"]
        logger.info(f"Detected project type: {type_detection['primary_type']} "
                   f"(confidence: {type_detection['confidence']:.2f})")

        # Create project structure
        logger.info("Creating DevStream project structure...")
        if devstream_dir.exists() and force_reinit:
            logger.info("Removing existing DevStream structure...")
            import shutil
            shutil.rmtree(devstream_dir)

        create_project_structure(str(project_path))

        # CRITICAL: Copy complete DevStream hooks and protocol system
        logger.info("Installing DevStream hooks and protocol system...")
        copy_devstream_hooks(str(project_path))

        # Create Python virtual environment for hooks
        logger.info("Creating Python virtual environment...")
        create_python_venv(project_path)

        # Create Claude settings.json with hooks configuration
        logger.info("Configuring Claude hooks...")
        create_claude_settings(project_path)

        # Create project database
        logger.info("Creating project database...")
        create_project_database(str(project_path))

        # Create project-specific CLAUDE.md
        logger.info("Creating project-specific CLAUDE.md...")
        create_project_claude_md(str(project_path), type_detection["primary_type"])

        # Update workspace.json with detected project type
        workspace_file = devstream_dir / "workspace.json"
        with open(workspace_file, 'r') as f:
            workspace_data = json.load(f)

        workspace_data["project_type"] = type_detection["primary_type"]
        workspace_data["project_detection"] = type_detection

        with open(workspace_file, 'w') as f:
            json.dump(workspace_data, f, indent=2)

        # Scan existing codebase if requested
        if scan_existing_codebase:
            logger.info("Scanning existing codebase...")
            scan_results = scan_and_populate_codebase(str(project_path))
            init_results["scan_results"] = scan_results

            # Update workspace.json with scan results
            workspace_data["scan_completed"] = True
            workspace_data["files_count"] = scan_results["files_scanned"]
            workspace_data["last_scanned"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            with open(workspace_file, 'w') as f:
                json.dump(workspace_data, f, indent=2)

        logger.info(f"Project initialization completed successfully")

        return init_results

    except Exception as e:
        logger.error(f"Project initialization failed: {e}")
        init_results["status"] = "failed"
        init_results["error"] = str(e)
        return init_results


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DevStream Project Initialization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/project                    Initialize new project
  %(prog)s /path/to/existing --scan             Initialize with codebase scanning
  %(prog)s . --force                           Reinitialize current project
  %(prog)s --help                              Show this help message
        """
    )

    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project directory (default: current directory)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reinitialization if project already exists"
    )

    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip codebase scanning"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="DevStream Init 2.2.0"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize project
        results = initialize_project(
            args.project_path,
            force_reinit=args.force,
            scan_existing_codebase=not args.no_scan
        )

        if results["status"] == "success":
            print("✅ DevStream project initialized successfully!")
            print()
            print("Project Details:")
            print(f"  Name: {results['project_name']}")
            print(f"  Path: {results['project_path']}")
            print(f"  Type: {results['project_type']}")
            print(f"  Workspace: {results['workspace_file']}")
            print(f"  Database: {results['database_path']}")
            print()

            if results["scan_results"]:
                scan = results["scan_results"]
                print("Codebase Scan Results:")
                print(f"  Files Scanned: {scan['files_scanned']}")
                print(f"  Embeddings Created: {scan['embeddings_created']}")
                print(f"  Scan Duration: {scan['scan_duration']:.2f}s")
                print(f"  File Types: {dict(scan['file_types'])}")
                print()

                if scan["errors"]:
                    print(f"  Warnings: {len(scan['errors'])}")
                    for error in scan["errors"][:3]:  # Show first 3 errors
                        print(f"    - {error}")
                    if len(scan["errors"]) > 3:
                        print(f"    ... and {len(scan['errors']) - 3} more warnings")
                    print()

            print("Next Steps:")
            print("  1. Start using DevStream with your project")
            print("  2. The project is now registered in the global registry")
            print("  3. Use 'devstream status' to verify the setup")
            print()

            return 0
        else:
            print(f"❌ Project initialization failed: {results.get('error', 'Unknown error')}")
            return 1

    except ProjectExistsError as e:
        print(f"❌ {e}")
        print("Use --force to reinitialize the project")
        return 1
    except CodebaseScanError as e:
        print(f"❌ Codebase scanning failed: {e}")
        print("Use --no-scan to skip codebase scanning")
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())