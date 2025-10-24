#!/usr/bin/env python3
"""
Setup script per development environment DevStream.

Questo script:
1. Verifica le dipendenze necessarie
2. Configura l'ambiente di sviluppo
3. Inizializza il database
4. Testa la connettività Ollama
5. Esegue test di base
"""

import asyncio
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from devstream.core.config import DevStreamConfig
from devstream.core.exceptions import DevStreamError

console = Console()
app = typer.Typer(help="DevStream Development Environment Setup")


def check_python_version() -> bool:
    """Verifica versione Python."""
    required_version = (3, 11)
    current_version = sys.version_info[:2]

    if current_version < required_version:
        console.print(
            f"❌ Python {required_version[0]}.{required_version[1]}+ required. "
            f"Current: {current_version[0]}.{current_version[1]}",
            style="red"
        )
        return False

    console.print(
        f"✅ Python {current_version[0]}.{current_version[1]} OK",
        style="green"
    )
    return True


def check_ollama_availability() -> bool:
    """Verifica disponibilità Ollama server."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            console.print("✅ Ollama server accessible", style="green")

            # Check for embedding models
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            embedding_models = ["nomic-embed-text", "embeddinggemma", "all-minilm"]
            available_embedding_models = [m for m in embedding_models if m in models]

            if available_embedding_models:
                console.print(
                    f"✅ Embedding models available: {', '.join(available_embedding_models)}",
                    style="green"
                )
            else:
                console.print(
                    "⚠️  No embedding models found. Install with:\n"
                    "   ollama pull nomic-embed-text",
                    style="yellow"
                )

            return True
        else:
            console.print(
                f"❌ Ollama server responded with status {response.status_code}",
                style="red"
            )
            return False

    except Exception as e:
        console.print(
            f"❌ Ollama server not accessible: {e}\n"
            "   Start with: ollama serve",
            style="red"
        )
        return False


def check_optional_dependencies() -> dict:
    """Verifica dipendenze opzionali."""
    optional_deps = {
        "sqlite-vss": "Vector search capabilities",
        "docker": "Container testing",
        "testcontainers": "Integration testing",
    }

    results = {}

    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace("-", "_"))
            console.print(f"✅ {dep}: {description}", style="green")
            results[dep] = True
        except ImportError:
            console.print(f"⚠️  {dep}: {description} (optional)", style="yellow")
            results[dep] = False

    return results


async def initialize_database(config: DevStreamConfig) -> bool:
    """Inizializza database con schema."""
    try:
        # Import here to avoid circular imports
        from devstream.database.connection import DatabaseManager

        db_manager = DatabaseManager(config.database)
        await db_manager.initialize()

        console.print("✅ Database initialized successfully", style="green")
        await db_manager.close()
        return True

    except Exception as e:
        console.print(f"❌ Database initialization failed: {e}", style="red")
        return False


async def run_basic_tests() -> bool:
    """Esegue test di base per verificare setup."""
    try:
        import subprocess

        # Run minimal test suite
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/unit", "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            console.print("✅ Basic tests passed", style="green")
            return True
        else:
            console.print("❌ Some tests failed:", style="red")
            console.print(result.stdout)
            console.print(result.stderr)
            return False

    except Exception as e:
        console.print(f"❌ Test execution failed: {e}", style="red")
        return False


@app.command()
def check(
    config_file: str = typer.Option(
        "config/development.yml",
        "--config",
        "-c",
        help="Configuration file to use"
    )
) -> None:
    """Verifica ambiente di sviluppo senza modifiche."""

    console.print(Panel.fit("🔍 DevStream Environment Check", style="blue"))

    # Check Python version
    console.print("\n📋 Checking Python version...")
    python_ok = check_python_version()

    # Load configuration
    console.print("\n⚙️  Loading configuration...")
    try:
        config_path = project_root / config_file
        config = DevStreamConfig.from_yaml(config_path)
        console.print(f"✅ Configuration loaded from {config_file}", style="green")
    except Exception as e:
        console.print(f"❌ Configuration loading failed: {e}", style="red")
        raise typer.Exit(1)

    # Check dependencies
    console.print("\n📦 Checking dependencies...")
    deps_ok = check_optional_dependencies()

    # Check Ollama
    console.print("\n🤖 Checking Ollama connectivity...")
    ollama_ok = check_ollama_availability()

    # Validate configuration
    console.print("\n🔍 Validating configuration...")
    issues = config.validate_dependencies()
    if issues:
        console.print("❌ Configuration issues found:", style="red")
        for issue in issues:
            console.print(f"   • {issue}", style="red")
    else:
        console.print("✅ Configuration validation passed", style="green")

    # Summary
    console.print("\n📊 Summary")
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")

    table.add_row("Python Version", "✅ OK" if python_ok else "❌ FAIL")
    table.add_row("Configuration", "✅ OK" if not issues else "❌ FAIL")
    table.add_row("Ollama Server", "✅ OK" if ollama_ok else "❌ FAIL")
    table.add_row("Vector Search", "✅ OK" if deps_ok.get("sqlite-vss") else "⚠️  Optional")
    table.add_row("Docker Testing", "✅ OK" if deps_ok.get("docker") else "⚠️  Optional")

    console.print(table)

    if not python_ok or issues:
        raise typer.Exit(1)


@app.command()
def setup(
    config_file: str = typer.Option(
        "config/development.yml",
        "--config",
        "-c",
        help="Configuration file to use"
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Skip running basic tests"
    )
) -> None:
    """Setup completo ambiente di sviluppo."""

    console.print(Panel.fit("🚀 DevStream Development Setup", style="blue"))

    # Run checks first
    console.print("Running environment checks...")
    try:
        check(config_file)
    except typer.Exit:
        console.print("❌ Environment checks failed. Please fix issues first.", style="red")
        raise

    # Load configuration
    config_path = project_root / config_file
    config = DevStreamConfig.from_yaml(config_path)

    # Setup with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Initialize database
        task = progress.add_task("Initializing database...", total=None)
        db_success = asyncio.run(initialize_database(config))
        progress.update(task, description="✅ Database initialized" if db_success else "❌ Database failed")

        if not db_success:
            raise typer.Exit(1)

        # Run tests if not skipped
        if not skip_tests:
            task = progress.add_task("Running basic tests...", total=None)
            test_success = asyncio.run(run_basic_tests())
            progress.update(task, description="✅ Tests passed" if test_success else "❌ Tests failed")

            if not test_success:
                console.print("\n⚠️  Tests failed, but setup continues...", style="yellow")

    # Final instructions
    console.print("\n🎉 Setup completed!", style="green")
    console.print("\nNext steps:")
    console.print("1. Activate virtual environment: poetry shell")
    console.print("2. Install pre-commit hooks: pre-commit install")
    console.print("3. Run full test suite: pytest")
    console.print("4. Start development: devstream --help")


@app.command()
def install_ollama_models() -> None:
    """Installa modelli Ollama raccomandati."""

    console.print(Panel.fit("📥 Installing Ollama Models", style="blue"))

    recommended_models = [
        ("nomic-embed-text", "137MB", "Recommended embedding model"),
        ("embeddinggemma", "149MB", "Alternative embedding model"),
        ("all-minilm", "23MB", "Lightweight embedding model"),
    ]

    console.print("\nRecommended models:")
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("Description", style="white")

    for model, size, desc in recommended_models:
        table.add_row(model, size, desc)

    console.print(table)

    install_all = typer.confirm("\nInstall all recommended models?")

    if install_all:
        import subprocess

        for model, _, _ in recommended_models:
            console.print(f"\n📥 Installing {model}...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", model],
                    check=True,
                    capture_output=True,
                    text=True
                )
                console.print(f"✅ {model} installed successfully", style="green")
            except subprocess.CalledProcessError as e:
                console.print(f"❌ Failed to install {model}: {e}", style="red")
            except FileNotFoundError:
                console.print("❌ Ollama CLI not found. Please install Ollama first.", style="red")
                break


if __name__ == "__main__":
    app()