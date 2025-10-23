---
title: Installation
description: Complete installation guide for exc-to-pdf
---

# Installation

This guide covers all installation methods for exc-to-pdf, from simple pip installation to development setup.

## 📋 System Requirements

### Minimum Requirements

* **Python**: 3.9 or higher
* **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
* **RAM**: 4GB minimum, 8GB recommended for large files
* **Storage**: 2x the size of your largest Excel file

### Recommended Setup

* **Python**: 3.11+ (for best performance)
* **RAM**: 16GB+ (for files >50MB)
* **SSD Storage**: For faster file processing

### Checking Python Version

```bash
# Check Python version
python --version

# Should show 3.9 or higher
Python 3.11.5
```

If you don't have Python installed:

=== "Windows"

    1. Download from [python.org](https://www.python.org/downloads/)
    2. Run the installer
    3. Check "Add Python to PATH" during installation
    4. Restart Command Prompt

=== "macOS"

    ```bash
    # Using Homebrew (recommended)
    brew install python@3.11

    # Or download from python.org
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-pip python3.11-venv
    ```

## 🚀 Installation Methods

### Method 1: pip install (Recommended)

For most users, the simplest installation method:

```bash
# Install from PyPI
pip install exc-to-pdf

# Verify installation
exc-to-pdf --version
```

**Pros:**
* Simple and fast
* Automatic dependency management
* Easy updates with `pip install --upgrade exc-to-pdf`

**Cons:**
* Latest features may not be available immediately
* Cannot modify source code

### Method 2: Development Installation

For developers or users who need the latest features:

```bash
# Clone the repository
git clone https://github.com/exc-to-pdf/exc-to-pdf.git
cd exc-to-pdf

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
exc-to-pdf --version
```

**Pros:**
* Access to latest features
* Can modify source code
* Includes development dependencies

**Cons:**
* More complex setup
* Requires git and virtual environment

### Method 3: From Source

Install directly from source without development dependencies:

```bash
# Clone the repository
git clone https://github.com/exc-to-pdf/exc-to-pdf.git
cd exc-to-pdf

# Install package
pip install .

# Verify installation
exc-to-pdf --version
```

## 🐍 Virtual Environment Setup

Using virtual environments is highly recommended to avoid conflicts.

### Creating a Virtual Environment

```bash
# Create virtual environment
python -m venv exc-to-pdf-env

# Activate environment
# On Windows
exc-to-pdf-env\Scripts\activate

# On macOS/Linux
source exc-to-pdf-env/bin/activate
```

### Installing in Virtual Environment

```bash
# Install exc-to-pdf
pip install exc-to-pdf

# Verify installation
exc-to-pdf --version

# Deactivate when done
deactivate
```

## 🔧 Dependencies

exc-to-pdf automatically installs these core dependencies:

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openpyxl` | ≥3.1.0 | Excel file reading |
| `pandas` | ≥2.0.0 | Data processing |
| `reportlab` | ≥4.0.0 | PDF generation |
| `Pillow` | ≥10.0.0 | Image handling |
| `click` | ≥8.0.0 | CLI interface |
| `structlog` | ≥23.0.0 | Structured logging |

### Optional Dependencies

For development installation:

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `black` | Code formatting |
| `mypy` | Type checking |
| `mkdocs` | Documentation |
| `mkdocs-material` | Documentation theme |

## ✅ Installation Verification

After installation, verify everything works correctly:

### Basic Commands

```bash
# Check version
exc-to-pdf --version

# Show help
exc-to-pdf --help

# Test conversion (create a simple test file first)
echo "Test,Data\n1,2\n3,4" > test.csv
# Convert to Excel first, then to PDF
```

### Python API Test

```python
# Create a simple test script
python -c "
import sys
try:
    from exc_to_pdf import PDFGenerator
    print('✅ exc-to-pdf imported successfully')
    print('✅ PDFGenerator class available')
    print('✅ Installation verified!')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"
```

### Dependency Check

```python
# Check all dependencies
python -c "
import sys
required_packages = ['openpyxl', 'pandas', 'reportlab', 'Pillow', 'click', 'structlog']
missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')

if missing:
    print(f'\\n❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('\\n✅ All dependencies satisfied!')
"
```

## 🔄 Updating exc-to-pdf

### pip Installation

```bash
# Update to latest version
pip install --upgrade exc-to-pdf

# Check version after update
exc-to-pdf --version
```

### Development Installation

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -e ".[dev]"

# Verify update
exc-to-pdf --version
```

## 🗑️ Uninstallation

### Standard Uninstallation

```bash
# Remove package
pip uninstall exc-to-pdf

# Verify removal
python -c "import exc_to_pdf" 2>/dev/null && echo "Still installed" || echo "Successfully removed"
```

### Development Uninstallation

```bash
# Remove package
pip uninstall exc-to-pdf

# Remove virtual environment (optional)
rm -rf exc-to-pdf-env

# Remove source code (optional)
rm -rf exc-to-pdf
```

## 🔍 Troubleshooting Installation

### Issue 1: Python Not Found

**Error**: `python: command not found`

**Solution**:
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Add Python to PATH during installation
3. Restart your terminal

### Issue 2: pip Not Found

**Error**: `pip: command not found`

**Solution**:
```bash
# Ensure pip is installed and up to date
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Issue 3: Permission Errors

**Error**: `Permission denied` during installation

**Solution**:
```bash
# Install in user directory (recommended)
pip install --user exc-to-pdf

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install exc-to-pdf
```

### Issue 4: Build Failures

**Error**: Compilation errors during installation

**Solution**:
```bash
# Install build tools first
# On Ubuntu/Debian
sudo apt install build-essential python3-dev

# On macOS
xcode-select --install

# Then retry installation
pip install exc-to-pdf
```

### Issue 5: Dependency Conflicts

**Error**: Version conflicts with existing packages

**Solution**:
```bash
# Use virtual environment to avoid conflicts
python -m venv clean-env
source clean-env/bin/activate
pip install exc-to-pdf
```

## 🌐 Offline Installation

For systems without internet access:

### Downloading Packages

```bash
# Download package and dependencies
pip download exc-to-pdf -d ./packages

# Transfer packages to offline system
# On offline system:
pip install exc-to-pdf --no-index --find-links ./packages
```

### Using Wheels

```bash
# Download wheel files
pip download --no-deps exc-to-pdf -d ./wheels

# Install from wheels on offline system
pip install exc-to-pdf --no-index --find-links ./wheels
```

## ✅ Post-Installation Checklist

After installation, verify:

* [ ] Python 3.9+ is available
* [ ] exc-to-pdf is installed (`exc-to-pdf --version`)
* [ ] Core dependencies are satisfied
* [ ] CLI commands work without errors
* [ ] Python API imports successfully
* [ ] Virtual environment is set up (recommended)

## 🎯 Next Steps

With exc-to-pdf installed, you're ready to:

* **[Start Converting Files](basic-usage.md)** - Your first conversion
* **[Explore Examples](examples.md)** - Practical use cases
* **[Read User Guide](../user-guide/index.md)** - Advanced features

!!! success "Installation Complete!"
    You have successfully installed exc-to-pdf! 🎉

    Ready to convert your first Excel file? **[Start here →](basic-usage.md)**