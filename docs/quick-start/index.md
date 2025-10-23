---
title: Quick Start
description: Get up and running with exc-to-pdf in minutes
---

# Quick Start Guide

Welcome to exc-to-pdf! This guide will help you get started quickly with converting Excel files to PDF format optimized for Google NotebookLM analysis.

## 🎯 What You'll Learn

In this guide, you'll learn how to:

* Install exc-to-pdf
* Convert your first Excel file to PDF
* Use basic command-line options
* Apply the Python API
* Troubleshoot common issues

## 📋 Prerequisites

Before you begin, ensure you have:

* **Python 3.9+** installed
* **Excel files** (.xlsx or .xls) to convert
* **Terminal/command line** access

!!! tip "System Requirements"
    * **RAM**: Minimum 4GB, recommended 8GB+
    * **Storage**: 2x the size of your Excel files
    * **OS**: Windows 10+, macOS 10.14+, or Linux

## 🚀 Installation

=== "pip install (Recommended)"

    ```bash
    # Install from PyPI
    pip install exc-to-pdf

    # Verify installation
    exc-to-pdf --version
    ```

=== "Development Installation"

    ```bash
    # Clone the repository
    git clone https://github.com/exc-to-pdf/exc-to-pdf.git
    cd exc-to-pdf

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install in development mode
    pip install -e ".[dev]"
    ```

!!! success "Installation Verification"
    After installation, verify everything works:

    ```bash
    exc-to-pdf --help
    ```
    You should see the help output with all available commands.

## 📄 Your First Conversion

Let's convert an Excel file to PDF format.

### Step 1: Prepare Your Excel File

Create or locate an Excel file you want to convert. For testing, you can create a simple file with:

* Multiple worksheets
* Tables with headers
* Some formatting

### Step 2: Basic Conversion

```bash
# Basic conversion
exc-to-pdf convert your-file.xlsx output.pdf

# See progress details
exc-to-pdf convert your-file.xlsx output.pdf --verbose
```

!!! tip "Output Location"
    If you don't specify an output file, exc-to-pdf will create one with the same name as your Excel file but with a `.pdf` extension.

### Step 3: Check the Results

Your PDF file should contain:
* All worksheets from your Excel file
* Formatted tables preserved
* Bookmarks for navigation
* Text that AI can analyze

## 🎨 Common Options

Here are the most useful options for everyday use:

### Template Styles

```bash
# Modern template (default)
exc-to-pdf convert data.xlsx output.pdf --template modern

# Classic template
exc-to-pdf convert data.xlsx output.pdf --template classic

# Minimal template
exc-to-pdf convert data.xlsx output.pdf --template minimal
```

### Page Orientation

```bash
# Portrait (default)
exc-to-pdf convert data.xlsx output.pdf --orientation portrait

# Landscape for wide tables
exc-to-pdf convert data.xlsx output.pdf --orientation landscape
```

### Worksheet Selection

```bash
# Convert specific worksheet
exc-to-pdf convert data.xlsx output.pdf --sheet "Sales Data"

# Convert all worksheets (default)
exc-to-pdf convert data.xlsx output.pdf
```

### Margins

```bash
# Custom margins (in points)
exc-to-pdf convert data.xlsx output.pdf \
  --margin-top 50 \
  --margin-bottom 50 \
  --margin-left 40 \
  --margin-right 40
```

## 🐍 Python API Usage

For programmatic use, exc-to-pdf provides a clean Python API.

### Basic Python Example

```python
from exc_to_pdf import PDFGenerator

# Create generator with default settings
generator = PDFGenerator()

# Convert file
generator.convert_excel_to_pdf(
    input_file="data.xlsx",
    output_file="output.pdf"
)

print("Conversion completed!")
```

### Advanced Python Example

```python
from exc_to_pdf import PDFGenerator
from exc_to_pdf.config import PDFConfig

# Custom configuration
config = PDFConfig()
config.table_style = "modern"
config.orientation = "landscape"
config.include_bookmarks = True
config.margin_top = 50
config.margin_bottom = 50

# Create generator with custom config
generator = PDFGenerator(config)

# Convert with options
generator.convert_excel_to_pdf(
    input_file="financial-report.xlsx",
    output_file="report.pdf",
    worksheet_name="Q4 Results"
)

print("Financial report converted successfully!")
```

## 🔍 Real-World Examples

### Example 1: Financial Reports

```bash
# Convert financial report with landscape orientation
exc-to-pdf convert financial-report.xlsx report.pdf \
  --template modern \
  --orientation landscape \
  --margin-top 60 \
  --margin-bottom 60
```

### Example 2: Academic Data

```bash
# Convert research data with specific worksheet
exc-to-pdf convert research-data.xlsx research.pdf \
  --sheet "Experimental Results" \
  --template classic \
  --verbose
```

### Example 3: Batch Processing

```python
import os
from exc_to_pdf import PDFGenerator

def batch_convert_excel_files(input_dir, output_dir):
    """Convert all Excel files in a directory"""
    generator = PDFGenerator()

    for filename in os.listdir(input_dir):
        if filename.endswith(('.xlsx', '.xls')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.pdf')

            try:
                generator.convert_excel_to_pdf(input_path, output_path)
                print(f"✅ Converted: {filename}")
            except Exception as e:
                print(f"❌ Failed: {filename} - {e}")

# Usage
batch_convert_excel_files("./excel_files/", "./pdf_outputs/")
```

## 🔧 Common Issues & Solutions

### Issue 1: "File not found" Error

**Problem**: The Excel file cannot be found.

**Solution**:
```bash
# Use absolute path
exc-to-pdf convert /full/path/to/file.xlsx output.pdf

# Or navigate to the directory first
cd /path/to/excel/files
exc-to-pdf convert file.xlsx output.pdf
```

### Issue 2: Memory Error with Large Files

**Problem**: Out of memory errors with large Excel files.

**Solution**:
```bash
# Process large files with verbose output for monitoring
exc-to-pdf convert large-file.xlsx output.pdf --verbose
```

For very large files (>50MB), consider:
* Closing other applications
* Using a machine with more RAM
* Processing files in smaller chunks

### Issue 3: Poor Formatting in PDF

**Problem**: Tables don't look good in the PDF.

**Solution**:
```bash
# Try different templates
exc-to-pdf convert data.xlsx output.pdf --template modern
exc-to-pdf convert data.xlsx output.pdf --template classic

# Adjust orientation for wide tables
exc-to-pdf convert data.xlsx output.pdf --orientation landscape
```

## ✅ Success Checklist

After your first conversion, verify:

* [ ] PDF file is created successfully
* [ ] All worksheets are included
* [ ] Tables are properly formatted
* [ ] Bookmarks work for navigation
* [ ] Text is selectable (not images)
* [ ] File size is reasonable

## 🎯 What's Next?

Now that you have exc-to-pdf working, explore:

* **[User Guide](../user-guide/index.md)** - Advanced configuration and options
* **[CLI Reference](../user-guide/cli-reference.md)** - Complete command-line documentation
* **[API Reference](../api/index.md)** - Full Python API documentation
* **[Examples](examples.md)** - More practical examples

!!! success "Congratulations!"
    You've successfully converted your first Excel file to PDF! 🎉

    Need help? Check our [troubleshooting guide](../user-guide/troubleshooting.md) or [open an issue](https://github.com/exc-to-pdf/exc-to-pdf/issues).