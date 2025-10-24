#!/usr/bin/env python3
"""
DevStream Installation Verification Script

Comprehensive validation of DevStream installation including:
- Python environment (venv, packages, versions)
- Database (schema, tables, triggers, indexes)
- MCP server (dependencies, build artifacts)
- Hook configuration (settings.json, hook scripts)
- Optional checks (Ollama, Git, environment)

Exit Codes:
    0: All checks passed
    1: Warnings present (installation may work with limitations)
    2: Critical failures (installation incomplete)

Usage:
    python scripts/verify-install.py [--verbose] [--json] [--skip-optional] [--fix]
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CheckStatus(Enum):
    """Status of individual check."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Result of an individual check."""
    category: str
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    remediation: Optional[str] = None


class VerificationReport:
    """Collection of check results with reporting capabilities."""

    def __init__(self) -> None:
        """Initialize empty report."""
        self.results: List[CheckResult] = []

    def add(
        self,
        category: str,
        name: str,
        status: CheckStatus,
        message: str,
        details: Optional[str] = None,
        remediation: Optional[str] = None
    ) -> None:
        """
        Add check result to report.

        Args:
            category: Check category (e.g., 'Python Environment')
            name: Check name (e.g., 'Python version')
            status: Check status (PASS/WARNING/FAIL/SKIP)
            message: Brief status message
            details: Optional detailed information
            remediation: Optional fix instructions
        """
        self.results.append(CheckResult(
            category=category,
            name=name,
            status=status,
            message=message,
            details=details,
            remediation=remediation
        ))

    def get_summary(self) -> Dict[str, int]:
        """
        Get summary counts by status.

        Returns:
            Dictionary with counts: {PASS: X, WARNING: Y, FAIL: Z, SKIP: W}
        """
        summary = {status: 0 for status in CheckStatus}
        for result in self.results:
            summary[result.status] += 1
        return summary

    def has_failures(self) -> bool:
        """Check if any checks failed."""
        return any(r.status == CheckStatus.FAIL for r in self.results)

    def has_warnings(self) -> bool:
        """Check if any checks have warnings."""
        return any(r.status == CheckStatus.WARNING for r in self.results)

    def to_json(self) -> str:
        """
        Export report as JSON string.

        Returns:
            JSON-formatted report
        """
        data = {
            "summary": {k.value: v for k, v in self.get_summary().items()},
            "results": [
                {
                    "category": r.category,
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "remediation": r.remediation
                }
                for r in self.results
            ]
        }
        return json.dumps(data, indent=2)


class DevStreamVerifier:
    """Main verification orchestrator."""

    def __init__(
        self,
        project_root: Path,
        verbose: bool = False,
        skip_optional: bool = False,
        fix: bool = False
    ) -> None:
        """
        Initialize verifier.

        Args:
            project_root: DevStream project root directory
            verbose: Enable verbose output
            skip_optional: Skip optional checks
            fix: Attempt auto-fix for issues
        """
        self.project_root = project_root
        self.verbose = verbose
        self.skip_optional = skip_optional
        self.fix = fix
        self.report = VerificationReport()

    def run_all_checks(self) -> VerificationReport:
        """
        Execute all verification checks.

        Returns:
            Verification report with all check results
        """
        self._check_python_environment()
        self._check_database()
        self._check_mcp_server()
        self._check_hook_configuration()

        if not self.skip_optional:
            self._check_optional_components()

        return self.report

    def _check_python_environment(self) -> None:
        """Check Python virtual environment and dependencies."""
        category = "Python Environment"

        # Check venv exists
        venv_python = self.project_root / ".devstream" / "bin" / "python"
        if not venv_python.exists():
            self.report.add(
                category,
                "Virtual Environment",
                CheckStatus.FAIL,
                "Virtual environment not found",
                details=f"Expected: {venv_python}",
                remediation="Run: python3.11 -m venv .devstream"
            )
            return  # Cannot continue without venv

        self.report.add(
            category,
            "Virtual Environment",
            CheckStatus.PASS,
            "Virtual environment exists",
            details=f"Location: {venv_python}"
        )

        # Check Python version
        try:
            result = subprocess.run(
                [str(venv_python), "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version_str = result.stdout.strip()
            if "Python 3.11" in version_str:
                self.report.add(
                    category,
                    "Python Version",
                    CheckStatus.PASS,
                    version_str,
                    details="Python 3.11.x detected"
                )
            else:
                self.report.add(
                    category,
                    "Python Version",
                    CheckStatus.WARNING,
                    version_str,
                    details="Expected Python 3.11.x",
                    remediation="Recreate venv with: python3.11 -m venv .devstream"
                )
        except subprocess.CalledProcessError as e:
            self.report.add(
                category,
                "Python Version",
                CheckStatus.FAIL,
                "Could not determine Python version",
                details=str(e),
                remediation="Recreate virtual environment"
            )

        # Check critical packages
        critical_packages = {
            "cchooks": "0.1.4",
            "aiohttp": "3.8.0",
            "structlog": "23.0.0",
            "python-dotenv": "1.0.0"
        }

        try:
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            installed = {
                pkg["name"]: pkg["version"]
                for pkg in json.loads(result.stdout)
            }

            for pkg_name, min_version in critical_packages.items():
                if pkg_name in installed:
                    installed_version = installed[pkg_name]
                    if self._version_compare(installed_version, min_version) >= 0:
                        self.report.add(
                            category,
                            f"Package: {pkg_name}",
                            CheckStatus.PASS,
                            f"Installed: {installed_version}",
                            details=f"Required: >={min_version}"
                        )
                    else:
                        self.report.add(
                            category,
                            f"Package: {pkg_name}",
                            CheckStatus.WARNING,
                            f"Outdated: {installed_version}",
                            details=f"Required: >={min_version}",
                            remediation=f"Run: .devstream/bin/python -m pip install --upgrade {pkg_name}"
                        )
                else:
                    self.report.add(
                        category,
                        f"Package: {pkg_name}",
                        CheckStatus.FAIL,
                        "Not installed",
                        details=f"Required: >={min_version}",
                        remediation=f"Run: .devstream/bin/python -m pip install {pkg_name}"
                    )

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.report.add(
                category,
                "Package Check",
                CheckStatus.FAIL,
                "Could not list installed packages",
                details=str(e),
                remediation="Check pip installation in venv"
            )

    def _check_database(self) -> None:
        """Check database schema and integrity."""
        category = "Database"

        db_path = self.project_root / "data" / "devstream.db"
        if not db_path.exists():
            self.report.add(
                category,
                "Database File",
                CheckStatus.FAIL,
                "Database file not found",
                details=f"Expected: {db_path}",
                remediation="Run: sqlite3 data/devstream.db < schema/schema.sql"
            )
            return

        # Check file permissions
        if not os.access(db_path, os.R_OK | os.W_OK):
            self.report.add(
                category,
                "Database File",
                CheckStatus.FAIL,
                "Database file not readable/writable",
                details=f"Permissions: {oct(db_path.stat().st_mode)}",
                remediation=f"Run: chmod 664 {db_path}"
            )
            return

        self.report.add(
            category,
            "Database File",
            CheckStatus.PASS,
            "Database file exists and accessible",
            details=f"Size: {db_path.stat().st_size / 1024:.1f} KB"
        )

        # Check schema
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check core tables (14 expected)
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            table_count = cursor.fetchone()[0]

            expected_tables = {
                "schema_version", "intervention_plans", "phases", "micro_tasks",
                "semantic_memory", "vec_semantic_memory", "fts_semantic_memory",
                "agents", "hooks", "hook_executions", "work_sessions",
                "context_injections", "learning_insights", "performance_metrics"
            }

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}

            missing_tables = expected_tables - existing_tables
            if missing_tables:
                self.report.add(
                    category,
                    "Core Tables",
                    CheckStatus.FAIL,
                    f"Missing {len(missing_tables)} table(s)",
                    details=f"Missing: {', '.join(sorted(missing_tables))}",
                    remediation="Recreate database from schema/schema.sql"
                )
            else:
                self.report.add(
                    category,
                    "Core Tables",
                    CheckStatus.PASS,
                    f"All {len(expected_tables)} core tables exist",
                    details=f"Tables: {', '.join(sorted(existing_tables))}"
                )

            # Check triggers (3 expected)
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
            )
            trigger_count = cursor.fetchone()[0]

            expected_triggers = {"sync_insert_memory", "sync_update_memory", "sync_delete_memory"}
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            existing_triggers = {row[0] for row in cursor.fetchall()}

            if existing_triggers == expected_triggers:
                self.report.add(
                    category,
                    "Triggers",
                    CheckStatus.PASS,
                    f"All {len(expected_triggers)} triggers exist",
                    details=f"Triggers: {', '.join(sorted(existing_triggers))}"
                )
            else:
                missing_triggers = expected_triggers - existing_triggers
                self.report.add(
                    category,
                    "Triggers",
                    CheckStatus.WARNING if existing_triggers else CheckStatus.FAIL,
                    f"Expected {len(expected_triggers)}, found {len(existing_triggers)}",
                    details=f"Missing: {', '.join(sorted(missing_triggers)) if missing_triggers else 'None'}",
                    remediation="Recreate database from schema/schema.sql"
                )

            # Check indexes (37 expected based on schema.sql)
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
            index_count = cursor.fetchone()[0]

            if index_count >= 30:  # Allow some variation
                self.report.add(
                    category,
                    "Indexes",
                    CheckStatus.PASS,
                    f"{index_count} indexes found",
                    details="Sufficient indexing for performance"
                )
            else:
                self.report.add(
                    category,
                    "Indexes",
                    CheckStatus.WARNING,
                    f"Only {index_count} indexes found (expected ~37)",
                    details="Performance may be degraded",
                    remediation="Recreate database from schema/schema.sql"
                )

            # Check virtual tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%VIRTUAL TABLE%'"
            )
            virtual_tables = {row[0] for row in cursor.fetchall()}
            expected_virtual = {"vec_semantic_memory", "fts_semantic_memory"}

            if expected_virtual.issubset(virtual_tables):
                self.report.add(
                    category,
                    "Virtual Tables",
                    CheckStatus.PASS,
                    "Vector search and FTS tables exist",
                    details=f"Virtual: {', '.join(sorted(virtual_tables))}"
                )
            else:
                missing_virtual = expected_virtual - virtual_tables
                self.report.add(
                    category,
                    "Virtual Tables",
                    CheckStatus.FAIL,
                    "Missing virtual table(s)",
                    details=f"Missing: {', '.join(sorted(missing_virtual))}",
                    remediation="Ensure sqlite-vec extension loaded, recreate database"
                )

            # Check schema version
            cursor.execute(
                "SELECT version, description FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            )
            version_row = cursor.fetchone()

            if version_row and version_row[0] == "2.1.0":
                self.report.add(
                    category,
                    "Schema Version",
                    CheckStatus.PASS,
                    f"Version: {version_row[0]}",
                    details=version_row[1] if version_row[1] else "Production schema"
                )
            else:
                self.report.add(
                    category,
                    "Schema Version",
                    CheckStatus.WARNING,
                    f"Version: {version_row[0] if version_row else 'Unknown'}",
                    details="Expected v2.1.0",
                    remediation="Update database schema"
                )

            conn.close()

        except sqlite3.Error as e:
            self.report.add(
                category,
                "Database Integrity",
                CheckStatus.FAIL,
                "Database error",
                details=str(e),
                remediation="Database may be corrupted, recreate from schema"
            )

    def _check_mcp_server(self) -> None:
        """Check MCP server installation and dependencies."""
        category = "MCP Server"

        mcp_dir = self.project_root / "mcp-devstream-server"
        if not mcp_dir.exists():
            self.report.add(
                category,
                "MCP Directory",
                CheckStatus.FAIL,
                "MCP server directory not found",
                details=f"Expected: {mcp_dir}",
                remediation="MCP server not installed"
            )
            return

        self.report.add(
            category,
            "MCP Directory",
            CheckStatus.PASS,
            "MCP server directory exists"
        )

        # Check dist directory
        dist_dir = mcp_dir / "dist"
        if not dist_dir.exists():
            self.report.add(
                category,
                "Build Artifacts",
                CheckStatus.FAIL,
                "dist/ directory not found",
                details="MCP server not built",
                remediation=f"Run: cd {mcp_dir} && npm run build"
            )
        else:
            index_js = dist_dir / "index.js"
            if index_js.exists():
                self.report.add(
                    category,
                    "Build Artifacts",
                    CheckStatus.PASS,
                    "MCP server built successfully",
                    details=f"Entry point: {index_js}"
                )
            else:
                self.report.add(
                    category,
                    "Build Artifacts",
                    CheckStatus.FAIL,
                    "dist/index.js not found",
                    remediation=f"Run: cd {mcp_dir} && npm run build"
                )

        # Check node_modules
        node_modules = mcp_dir / "node_modules"
        if not node_modules.exists():
            self.report.add(
                category,
                "Node Dependencies",
                CheckStatus.FAIL,
                "node_modules/ not found",
                details="Dependencies not installed",
                remediation=f"Run: cd {mcp_dir} && npm install"
            )
        else:
            # Check critical packages
            critical_deps = [
                "@modelcontextprotocol/sdk",
                "better-sqlite3",
                "ollama",
                "sqlite-vec"
            ]

            missing_deps = []
            for dep in critical_deps:
                dep_path = node_modules / dep.replace("/", os.sep)
                if not dep_path.exists():
                    missing_deps.append(dep)

            if missing_deps:
                self.report.add(
                    category,
                    "Node Dependencies",
                    CheckStatus.WARNING,
                    f"Missing {len(missing_deps)} package(s)",
                    details=f"Missing: {', '.join(missing_deps)}",
                    remediation=f"Run: cd {mcp_dir} && npm install"
                )
            else:
                self.report.add(
                    category,
                    "Node Dependencies",
                    CheckStatus.PASS,
                    f"All {len(critical_deps)} critical packages installed"
                )

        # Check package.json
        package_json = mcp_dir / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                    version = pkg_data.get("version", "unknown")
                    self.report.add(
                        category,
                        "Package Configuration",
                        CheckStatus.PASS,
                        f"Version: {version}",
                        details=f"Name: {pkg_data.get('name', 'unknown')}"
                    )
            except json.JSONDecodeError:
                self.report.add(
                    category,
                    "Package Configuration",
                    CheckStatus.WARNING,
                    "Invalid package.json",
                    remediation="Fix JSON syntax in package.json"
                )
        else:
            self.report.add(
                category,
                "Package Configuration",
                CheckStatus.FAIL,
                "package.json not found",
                remediation="MCP server missing configuration"
            )

    def _check_hook_configuration(self) -> None:
        """Check hook configuration in settings.json."""
        category = "Hook Configuration"

        settings_path = Path.home() / ".claude" / "settings.json"
        if not settings_path.exists():
            self.report.add(
                category,
                "Settings File",
                CheckStatus.FAIL,
                "~/.claude/settings.json not found",
                remediation="Create settings.json with hook configuration"
            )
            return

        self.report.add(
            category,
            "Settings File",
            CheckStatus.PASS,
            "Settings file exists",
            details=str(settings_path)
        )

        # Parse settings.json
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            hooks = settings.get("hooks", {})
            if not hooks:
                self.report.add(
                    category,
                    "Hook Configuration",
                    CheckStatus.FAIL,
                    "No hooks configured in settings.json",
                    remediation="Add hooks configuration to settings.json"
                )
                return

            # Check critical hooks
            required_hooks = {
                "PreToolUse": ".claude/hooks/devstream/memory/pre_tool_use.py",
                "PostToolUse": ".claude/hooks/devstream/memory/post_tool_use.py",
                "UserPromptSubmit": ".claude/hooks/devstream/context/user_query_context_enhancer.py"
            }

            for hook_name, expected_script in required_hooks.items():
                hook_configs = hooks.get(hook_name, [])
                if not hook_configs:
                    self.report.add(
                        category,
                        f"Hook: {hook_name}",
                        CheckStatus.FAIL,
                        "Not configured",
                        remediation=f"Add {hook_name} hook to settings.json"
                    )
                    continue

                # Check if any hook config uses .devstream/bin/python
                found_venv_python = False
                for config in hook_configs:
                    for hook_item in config.get("hooks", []):
                        command = hook_item.get("command", "")
                        if ".devstream/bin/python" in command and expected_script in command:
                            found_venv_python = True
                            break

                if found_venv_python:
                    # Check if script exists
                    script_path = self.project_root / expected_script
                    if script_path.exists():
                        self.report.add(
                            category,
                            f"Hook: {hook_name}",
                            CheckStatus.PASS,
                            "Configured correctly",
                            details=f"Script: {expected_script}"
                        )
                    else:
                        self.report.add(
                            category,
                            f"Hook: {hook_name}",
                            CheckStatus.FAIL,
                            "Script file not found",
                            details=f"Expected: {script_path}",
                            remediation="Restore hook script file"
                        )
                else:
                    self.report.add(
                        category,
                        f"Hook: {hook_name}",
                        CheckStatus.WARNING,
                        "Not using .devstream/bin/python",
                        details="May use wrong Python interpreter",
                        remediation="Update hook command to use .devstream/bin/python"
                    )

        except json.JSONDecodeError:
            self.report.add(
                category,
                "Settings File",
                CheckStatus.FAIL,
                "Invalid JSON in settings.json",
                remediation="Fix JSON syntax"
            )
        except Exception as e:
            self.report.add(
                category,
                "Settings File",
                CheckStatus.FAIL,
                "Error reading settings.json",
                details=str(e)
            )

    def _check_optional_components(self) -> None:
        """Check optional components (non-critical)."""
        category = "Optional Components"

        # Check Ollama
        try:
            with urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read())
                    models = [m["name"] for m in data.get("models", [])]
                    if "nomic-embed-text" in models:
                        self.report.add(
                            category,
                            "Ollama + nomic-embed-text",
                            CheckStatus.PASS,
                            "Running with required model",
                            details=f"Models: {len(models)}"
                        )
                    else:
                        self.report.add(
                            category,
                            "Ollama + nomic-embed-text",
                            CheckStatus.WARNING,
                            "nomic-embed-text model not found",
                            details="Semantic search unavailable",
                            remediation="Run: ollama pull nomic-embed-text"
                        )
                else:
                    self.report.add(
                        category,
                        "Ollama",
                        CheckStatus.WARNING,
                        "Ollama not responding",
                        remediation="Start Ollama service"
                    )
        except (URLError, TimeoutError):
            self.report.add(
                category,
                "Ollama",
                CheckStatus.WARNING,
                "Not running",
                details="Semantic search unavailable",
                remediation="Install and start Ollama: https://ollama.ai"
            )

        # Check Git repository
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            self.report.add(
                category,
                "Git Repository",
                CheckStatus.PASS,
                "Git repository initialized"
            )
        else:
            self.report.add(
                category,
                "Git Repository",
                CheckStatus.WARNING,
                "Not a Git repository",
                details="Version control unavailable",
                remediation="Run: git init"
            )

        # Check .env.devstream
        env_file = self.project_root / ".env.devstream"
        if env_file.exists():
            self.report.add(
                category,
                "Environment Configuration",
                CheckStatus.PASS,
                ".env.devstream exists"
            )
        else:
            self.report.add(
                category,
                "Environment Configuration",
                CheckStatus.WARNING,
                ".env.devstream not found",
                details="Using default configuration",
                remediation="Copy .env.example to .env.devstream and customize"
            )

    @staticmethod
    def _version_compare(v1: str, v2: str) -> int:
        """
        Compare two semantic version strings.

        Args:
            v1: First version string (e.g., "1.2.3")
            v2: Second version string (e.g., "1.2.0")

        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        def parse_version(v: str) -> List[int]:
            return [int(x) for x in v.split(".")[:3]]  # Only major.minor.patch

        try:
            parts1 = parse_version(v1)
            parts2 = parse_version(v2)

            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            return 0
        except (ValueError, IndexError):
            return 0  # Treat unparseable versions as equal


def print_report_rich(report: VerificationReport) -> None:
    """
    Print verification report using rich formatting.

    Args:
        report: Verification report to display
    """
    if not RICH_AVAILABLE:
        print_report_simple(report)
        return

    console = Console()

    # Title
    console.print("\n[bold cyan]DevStream Installation Verification[/bold cyan]\n")

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Message")

    # Add rows
    for result in report.results:
        status_styles = {
            CheckStatus.PASS: "[green]✅ PASS[/green]",
            CheckStatus.WARNING: "[yellow]⚠️  WARNING[/yellow]",
            CheckStatus.FAIL: "[red]❌ FAIL[/red]",
            CheckStatus.SKIP: "[dim]⊘ SKIP[/dim]"
        }

        table.add_row(
            result.category,
            result.name,
            status_styles[result.status],
            result.message
        )

    console.print(table)

    # Show details for failures and warnings
    failures = [r for r in report.results if r.status == CheckStatus.FAIL]
    warnings = [r for r in report.results if r.status == CheckStatus.WARNING]

    if failures:
        console.print("\n[bold red]Failures:[/bold red]")
        for result in failures:
            console.print(f"\n[red]❌ {result.category} - {result.name}[/red]")
            if result.details:
                console.print(f"   Details: {result.details}")
            if result.remediation:
                console.print(f"   [yellow]Fix:[/yellow] {result.remediation}")

    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for result in warnings:
            console.print(f"\n[yellow]⚠️  {result.category} - {result.name}[/yellow]")
            if result.details:
                console.print(f"   Details: {result.details}")
            if result.remediation:
                console.print(f"   [cyan]Suggestion:[/cyan] {result.remediation}")

    # Summary
    summary = report.get_summary()
    total = sum(summary.values())
    passed = summary[CheckStatus.PASS]

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total Checks: {total}")
    console.print(f"  [green]Passed: {summary[CheckStatus.PASS]}[/green]")
    console.print(f"  [yellow]Warnings: {summary[CheckStatus.WARNING]}[/yellow]")
    console.print(f"  [red]Failed: {summary[CheckStatus.FAIL]}[/red]")
    console.print(f"  [dim]Skipped: {summary[CheckStatus.SKIP]}[/dim]\n")


def print_report_simple(report: VerificationReport) -> None:
    """
    Print verification report using simple text formatting.

    Args:
        report: Verification report to display
    """
    print("\nDevStream Installation Verification")
    print("=" * 80)

    for result in report.results:
        status_marks = {
            CheckStatus.PASS: "✓",
            CheckStatus.WARNING: "⚠",
            CheckStatus.FAIL: "✗",
            CheckStatus.SKIP: "-"
        }

        print(f"\n[{status_marks[result.status]}] {result.category} - {result.name}")
        print(f"    Status: {result.status.value}")
        print(f"    Message: {result.message}")
        if result.details:
            print(f"    Details: {result.details}")
        if result.remediation:
            print(f"    Fix: {result.remediation}")

    # Summary
    summary = report.get_summary()
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total: {sum(summary.values())}")
    print(f"  Passed: {summary[CheckStatus.PASS]}")
    print(f"  Warnings: {summary[CheckStatus.WARNING]}")
    print(f"  Failed: {summary[CheckStatus.FAIL]}")
    print(f"  Skipped: {summary[CheckStatus.SKIP]}\n")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0=success, 1=warnings, 2=failures)
    """
    parser = argparse.ArgumentParser(
        description="Verify DevStream installation completeness"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed check results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional checks (Ollama, Git, etc.)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt auto-fix for issues (not yet implemented)"
    )

    args = parser.parse_args()

    # Determine project root (script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Run verification
    verifier = DevStreamVerifier(
        project_root=project_root,
        verbose=args.verbose,
        skip_optional=args.skip_optional,
        fix=args.fix
    )

    report = verifier.run_all_checks()

    # Output results
    if args.json:
        print(report.to_json())
    else:
        if RICH_AVAILABLE:
            print_report_rich(report)
        else:
            print_report_simple(report)

    # Determine exit code
    if report.has_failures():
        return 2  # Critical failures
    elif report.has_warnings():
        return 1  # Warnings present
    else:
        return 0  # All passed


if __name__ == "__main__":
    sys.exit(main())
