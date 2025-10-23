---
title: Basic Usage
description: Learn the fundamentals of using exc-to-pdf
---

# Basic Usage

This guide covers the fundamental operations you'll perform with exc-to-pdf, from simple conversions to common customization options.

## 📄 Converting Your First File

### Simple Conversion

The most basic usage is converting an Excel file to PDF:

```bash
# Basic syntax
exc-to-pdf convert <input.xlsx> [output.pdf]

# Example
exc-to-pdf convert financial-report.pdf report.pdf
```

!!! tip "Output File Naming"
    If you don't specify an output file, exc-to-pdf automatically creates one with the same name as your Excel file but with a `.pdf` extension:

    ```bash
    # This creates report.pdf automatically
    exc-to-pdf convert report.xlsx
    ```

### Understanding the Process

When you run exc-to-pdf, it performs these steps:

1. **Excel Analysis**: Reads and analyzes your Excel file
2. **Table Detection**: Identifies table structures and data relationships
3. **PDF Generation**: Creates a formatted PDF with proper tables
4. **Navigation Setup**: Adds bookmarks and metadata for AI analysis
5. **Output**: Saves the optimized PDF file

## 🎨 Template Selection

Choose from three professionally designed templates:

### Modern Template (Default)

```bash
exc-to-pdf convert data.xlsx output.pdf --template modern
```

**Features:**
* Clean, contemporary design
* Blue accent colors
* Optimized for business reports
* Excellent readability

### Classic Template

```bash
exc-to-pdf convert data.xlsx output.pdf --template classic
```

**Features:**
* Traditional, formal appearance
* Black and white design
* Perfect for academic or official documents
* Timeless formatting

### Minimal Template

```bash
exc-to-pdf convert data.xlsx output.pdf --template minimal
```

**Features:**
* Simple, uncluttered layout
* Minimal styling
* Focus on content
* Fast processing

## 📐 Page Configuration

### Page Orientation

```bash
# Portrait (default) - good for most data
exc-to-pdf convert data.xlsx output.pdf --orientation portrait

# Landscape - better for wide tables
exc-to-pdf convert data.xlsx output.pdf --orientation landscape
```

**When to use Landscape:**
* Tables with many columns
* Financial statements
* Datasets with wide data ranges
* Charts and graphics

### Custom Margins

Fine-tune the layout with custom margins (measured in points, 72 points = 1 inch):

```bash
# Default margins: 72 points (1 inch)
exc-to-pdf convert data.xlsx output.pdf

# Tighter margins for more content
exc-to-pdf convert data.xlsx output.pdf \
  --margin-top 50 \
  --margin-bottom 50 \
  --margin-left 40 \
  --margin-right 40

# Wide margins for professional look
exc-to-pdf convert data.xlsx output.pdf \
  --margin-top 90 \
  --margin-bottom 90 \
  --margin-left 72 \
  --margin-right 72
```

## 📋 Worksheet Management

### Convert Specific Worksheets

```bash
# Convert only one worksheet
exc-to-pdf convert workbook.xlsx output.pdf --sheet "Sales Data"

# Convert worksheet with spaces in name
exc-to-pdf convert workbook.xlsx output.pdf --sheet "Q4 Results"
```

### Convert All Worksheets

```bash
# Convert all worksheets (default behavior)
exc-to-pdf convert workbook.xlsx output.pdf

# Explicitly specify all worksheets
exc-to-pdf convert workbook.xlsx output.pdf --sheet all
```

### Finding Worksheet Names

```bash
# Use verbose mode to see available worksheets
exc-to-pdf convert workbook.xlsx output.pdf --verbose
```

This will show output like:
```
📄 Found 3 worksheets:
  1. Sales Data
  2. Customer List
  3. Product Catalog
🔄 Processing worksheet: Sales Data
```

## 🔍 Output Control

### Verbose Mode

Monitor the conversion process in detail:

```bash
# Show detailed progress
exc-to-pdf convert data.xlsx output.pdf --verbose
```

**Verbose output includes:**
* Worksheet detection
* Table analysis
* Processing progress
* File size information
* Success/failure details

### Quiet Mode

Suppress all output except errors:

```bash
# Silent operation
exc-to-pdf convert data.xlsx output.pdf --quiet
```

**Use cases:**
* Batch processing scripts
* Automated workflows
* CI/CD pipelines

## 🔧 Advanced Options

### Disable Bookmarks

If you don't want automatic bookmark generation:

```bash
exc-to-pdf convert data.xlsx output.pdf --no-bookmarks
```

**Why disable bookmarks?**
* Smaller file size
* Faster processing
* Simple documents don't need navigation

### Disable Metadata

Skip AI-optimized metadata:

```bash
exc-to-pdf convert data.xlsx output.pdf --no-metadata
```

**Why disable metadata?**
* Privacy concerns
* Smaller files
* Basic PDF creation

## 📊 Real-World Examples

### Example 1: Financial Report

```bash
# Convert financial report with professional styling
exc-to-pdf convert financial-report.xlsx report.pdf \
  --template modern \
  --orientation portrait \
  --margin-top 80 \
  --margin-bottom 80 \
  --verbose
```

### Example 2: Wide Data Table

```bash
# Convert wide table in landscape mode
exc-to-pdf convert wide-data.xlsx data.pdf \
  --template classic \
  --orientation landscape \
  --margin-left 30 \
  --margin-right 30 \
  --margin-top 50 \
  --margin-bottom 50
```

### Example 3: Specific Worksheet

```bash
# Convert only the summary worksheet
exc-to-pdf convert workbook.xlsx summary.pdf \
  --sheet "Executive Summary" \
  --template modern \
  --quiet
```

### Example 4: Minimal Processing

```bash
# Fast conversion with minimal features
exc-to-pdf convert data.pdf output.pdf \
  --template minimal \
  --no-bookmarks \
  --no-metadata \
  --quiet
```

## 🐍 Python API Basics

### Simple Conversion

```python
from exc_to_pdf import PDFGenerator

# Create generator
generator = PDFGenerator()

# Convert file
generator.convert_excel_to_pdf(
    input_file="data.xlsx",
    output_file="output.pdf"
)

print("Conversion completed!")
```

### Conversion with Options

```python
from exc_to_pdf import PDFGenerator

# Create generator
generator = PDFGenerator()

# Convert with worksheet selection
generator.convert_excel_to_pdf(
    input_file="workbook.xlsx",
    output_file="specific.pdf",
    worksheet_name="Sales Data"
)
```

### Batch Processing

```python
import os
from exc_to_pdf import PDFGenerator

def convert_directory(input_dir, output_dir):
    """Convert all Excel files in a directory"""
    generator = PDFGenerator()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.xlsx', '.xls')):
            input_path = os.path.join(input_dir, filename)
            output_name = filename.rsplit('.', 1)[0] + '.pdf'
            output_path = os.path.join(output_dir, output_name)

            try:
                generator.convert_excel_to_pdf(input_path, output_path)
                print(f"✅ Converted: {filename} → {output_name}")
            except Exception as e:
                print(f"❌ Failed: {filename} - {e}")

# Usage
convert_directory("./excel_files/", "./pdf_output/")
```

## 🔍 Troubleshooting Basic Issues

### File Not Found

```bash
# Error: No such file or directory
exc-to-pdf convert missing.xlsx output.pdf

# Solutions:
# 1. Check file path
ls -la /path/to/file.xlsx

# 2. Use absolute path
exc-to-pdf convert /full/path/to/file.xlsx output.pdf

# 3. Navigate to file directory
cd /path/to/directory
exc-to-pdf convert file.xlsx output.pdf
```

### Permission Denied

```bash
# Error: Permission denied
exc-to-pdf convert /protected/file.xlsx /protected/output.pdf

# Solutions:
# 1. Check permissions
ls -la /path/to/directory/

# 2. Use different output directory
exc-to-pdf convert input.xlsx ~/output.pdf

# 3. Fix permissions (if you have access)
chmod 644 input.xlsx
chmod 755 output_directory/
```

### Large File Processing

```bash
# For large files, use verbose mode to monitor progress
exc-to-pdf convert large-file.xlsx output.pdf --verbose

# Or process in smaller chunks if possible
# (requires splitting the Excel file beforehand)
```

## ✅ Success Verification

After conversion, verify your PDF:

### Manual Checks

```bash
# Check file size
ls -lh output.pdf

# Check file type
file output.pdf
```

### Programmatic Verification

```python
import os
from pathlib import Path

def verify_pdf(output_path):
    """Verify PDF was created successfully"""
    path = Path(output_path)

    if not path.exists():
        print("❌ PDF file was not created")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"✅ PDF created successfully")
    print(f"📊 File size: {size_mb:.2f} MB")

    if size_mb < 0.01:  # Very small file might indicate an error
        print("⚠️  Warning: File is very small, check conversion")

    return True

# Usage
verify_pdf("output.pdf")
```

## 🎯 Next Steps

Now that you understand basic usage:

* **[Examples](examples.md)** - More practical scenarios
* **[Configuration](../user-guide/configuration.md)** - Advanced configuration
* **[Performance](../user-guide/performance.md)** - Optimization tips

!!! success "Ready to advance?"
    You've mastered the basics! Try these advanced features:
    * Custom configuration files
    * Batch processing scripts
    * Performance optimization