---
title: User Guide
description: Comprehensive guide to using exc-to-pdf
---

# User Guide

Welcome to the comprehensive user guide for exc-to-pdf. This guide covers all aspects of using the tool, from basic operations to advanced configuration and optimization.

## 📚 Guide Overview

### Sections

1. **[CLI Reference](cli-reference.md)** - Complete command-line interface documentation
2. **[Configuration](configuration.md)** - Advanced configuration options and customization
3. **[Templates & Styling](templates.md)** - Template system and visual customization
4. **[Performance](performance.md)** - Performance optimization and resource management
5. **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🎯 Who This Guide Is For

This guide is designed for:

* **End Users** who want to convert Excel files to PDF
* **Power Users** who need advanced configuration options
* **Developers** integrating exc-to-pdf into their workflows
* **System Administrators** managing batch processing

## 🚀 Quick Reference

### Essential Commands

```bash
# Basic conversion
exc-to-pdf convert input.xlsx output.pdf

# With options
exc-to-pdf convert input.xlsx output.pdf \
  --template modern \
  --orientation landscape \
  --sheet "Specific Sheet" \
  --verbose

# Configuration management
exc-to-pdf config validate config.toml
exc-to-pdf config template --output my-config.toml
```

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--template` | PDF template style | `--template modern` |
| `--orientation` | Page orientation | `--orientation landscape` |
| `--sheet` | Specific worksheet | `--sheet "Sales Data"` |
| `--verbose` | Detailed output | `--verbose` |
| `--quiet` | Silent operation | `--quiet` |
| `--no-bookmarks` | Disable bookmarks | `--no-bookmarks` |
| `--no-metadata` | Disable metadata | `--no-metadata` |

## 📋 Usage Patterns

### Pattern 1: Single File Conversion

For converting individual Excel files:

```bash
# Quick conversion
exc-to-pdf convert report.xlsx report.pdf

# Professional report
exc-to-pdf convert financial-report.pdf report.pdf \
  --template modern \
  --orientation portrait \
  --margin-top 80 \
  --margin-bottom 80
```

### Pattern 2: Batch Processing

For processing multiple files:

```bash
#!/bin/bash
# Process all Excel files in directory
for file in *.xlsx; do
    exc-to-pdf convert "$file" "${file%.xlsx}.pdf" \
        --template modern \
        --quiet
done
```

### Pattern 3: Automated Workflows

For integration into automated systems:

```python
from exc_to_pdf import PDFGenerator
import os

def automated_conversion(input_dir, output_dir):
    """Automated conversion for CI/CD pipelines"""
    generator = PDFGenerator()

    for filename in os.listdir(input_dir):
        if filename.endswith('.xlsx'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.xlsx', '.pdf'))

            try:
                generator.convert_excel_to_pdf(input_path, output_path)
                print(f"✅ Converted: {filename}")
            except Exception as e:
                print(f"❌ Failed: {filename} - {e}")
                # In automated workflows, you might want to:
                # - Log the error
                # - Send notifications
                # - Fail the pipeline
                raise
```

## 🎨 Template Selection Guide

### Modern Template

**Best for:**
* Business reports
* Presentations
* Professional documents

**Features:**
* Clean, contemporary design
* Professional color scheme
* Excellent readability

```bash
exc-to-pdf convert data.xlsx output.pdf --template modern
```

### Classic Template

**Best for:**
* Academic papers
* Formal documents
* Traditional reports

**Features:**
* Timeless design
* Black and white color scheme
* Academic formatting

```bash
exc-to-pdf convert research.xlsx output.pdf --template classic
```

### Minimal Template

**Best for:**
* Data-focused documents
* Simple conversions
* Fast processing

**Features:**
* Uncluttered layout
* Minimal styling
* Content-focused

```bash
exc-to-pdf convert data.xlsx output.pdf --template minimal
```

## 📐 Layout Optimization

### Portrait vs Landscape

**Portrait (Default):**
* Standard documents
* Reports with text
* Most business documents

**Landscape:**
* Wide tables
* Financial statements
* Charts and graphics

### Margin Guidelines

| Document Type | Top/Bottom | Left/Right |
|---------------|------------|------------|
| Standard Reports | 72pt (1") | 72pt (1") |
| Formal Documents | 90pt (1.25") | 72pt (1") |
| Data-Heavy | 50pt (0.7") | 40pt (0.55") |
| Presentations | 60pt (0.83") | 60pt (0.83") |

```bash
# Example: Formal document margins
exc-to-pdf convert document.pdf output.pdf \
  --margin-top 90 \
  --margin-bottom 90 \
  --margin-left 72 \
  --margin-right 72
```

## 🔧 Configuration Management

### Configuration File Usage

Create reusable configurations:

```toml
# company-config.toml
[page]
template = "modern"
orientation = "portrait"
margin_top = 80
margin_bottom = 80
margin_left = 72
margin_right = 72

[features]
include_bookmarks = true
include_metadata = true
optimize_for_ai = true

[output]
compression = true
quality = "high"
```

```bash
# Use configuration file
exc-to-pdf convert input.xlsx output.pdf --config company-config.toml
```

### Environment-Specific Configurations

**Development:**
```toml
# dev-config.toml
[page]
template = "minimal"
[features]
include_bookmarks = false
```

**Production:**
```toml
# prod-config.toml
[page]
template = "modern"
[features]
include_bookmarks = true
include_metadata = true
```

## 📊 Performance Considerations

### File Size Management

| Excel Size | Expected PDF Size | Processing Time |
|------------|-------------------|-----------------|
| < 1 MB | 0.3-0.7 MB | < 5 seconds |
| 1-10 MB | 0.5-2 MB | 5-30 seconds |
| 10-50 MB | 1-5 MB | 30-120 seconds |
| > 50 MB | 2-10 MB | 2+ minutes |

### Memory Usage

* **Base Memory**: ~100MB
* **Per MB Excel**: ~2-4MB additional RAM
* **Peak Memory**: Base + (Excel size × 3)

```bash
# Monitor memory usage on large files
exc-to-pdf convert large.xlsx output.pdf --verbose

# Use minimal template for lower memory usage
exc-to-pdf convert large.xlsx output.pdf --template minimal --no-bookmarks
```

## 🔍 Troubleshooting Quick Reference

### Common Issues

**Issue**: File not found
```bash
# Solution: Use absolute paths
exc-to-pdf convert /full/path/to/file.xlsx output.pdf
```

**Issue**: Permission denied
```bash
# Solution: Check permissions or use different output directory
exc-to-pdf convert input.xlsx ~/output.pdf
```

**Issue**: Out of memory
```bash
# Solution: Use minimal template
exc-to-pdf convert large.xlsx output.pdf --template minimal --no-bookmarks
```

**Issue**: Poor formatting
```bash
# Solution: Try different templates or orientation
exc-to-pdf convert data.xlsx output.pdf --template modern --orientation landscape
```

## 🎯 Best Practices

### Before Conversion

1. **Check Excel file integrity**
2. **Verify sufficient disk space** (2x Excel file size)
3. **Close other applications** for large files
4. **Test with sample data** first

### During Conversion

1. **Use verbose mode** for large files
2. **Monitor system resources**
3. **Save intermediate results** for batch processing

### After Conversion

1. **Verify PDF integrity**
2. **Check file size** (unusually small = error)
3. **Test in Google NotebookLM** if that's your target
4. **Backup important conversions**

## 🔗 Integration Examples

### Shell Script Integration

```bash
#!/bin/bash
# daily-reports.sh - Convert daily Excel reports

REPORTS_DIR="/path/to/daily_reports"
OUTPUT_DIR="/path/to/pdf_output"
CONFIG_FILE="/path/to/config.toml"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y-%m-%d")
TODAY_OUTPUT="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$TODAY_OUTPUT"

echo "📊 Processing daily reports for $TIMESTAMP"

# Process all Excel files
for excel_file in "$REPORTS_DIR"/*.xlsx; do
    if [ -f "$excel_file" ]; then
        filename=$(basename "$excel_file" .xlsx)
        pdf_file="$TODAY_OUTPUT/${filename}.pdf"

        echo "🔄 Converting: $filename"

        exc-to-pdf convert "$excel_file" "$pdf_file" \
            --config "$CONFIG_FILE" \
            --verbose

        if [ $? -eq 0 ]; then
            echo "✅ Success: $filename"
        else
            echo "❌ Failed: $filename"
        fi
    fi
done

echo "🎯 Daily reports processing completed"
```

### Python Integration

```python
# excel_to_pdf_service.py - Service for Excel to PDF conversion

import os
import logging
from pathlib import Path
from typing import List, Optional
from exc_to_pdf import PDFGenerator

class ExcelToPDFService:
    """Service class for Excel to PDF conversions"""

    def __init__(self, config_path: Optional[str] = None):
        self.generator = PDFGenerator()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def convert_file(self, input_path: str, output_path: str, **kwargs) -> bool:
        """Convert single Excel file to PDF"""
        try:
            self.logger.info(f"Converting: {input_path}")
            self.generator.convert_excel_to_pdf(input_path, output_path, **kwargs)
            self.logger.info(f"Success: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert {input_path}: {e}")
            return False

    def convert_directory(self, input_dir: str, output_dir: str,
                         pattern: str = "*.xlsx", **kwargs) -> List[str]:
        """Convert all Excel files in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        converted_files = []
        failed_files = []

        for excel_file in input_path.glob(pattern):
            pdf_file = output_path / f"{excel_file.stem}.pdf"

            if self.convert_file(str(excel_file), str(pdf_file), **kwargs):
                converted_files.append(str(pdf_file))
            else:
                failed_files.append(str(excel_file))

        self.logger.info(f"Converted: {len(converted_files)}, Failed: {len(failed_files)}")
        return converted_files

# Usage example
if __name__ == "__main__":
    service = ExcelToPDFService()

    # Convert single file
    service.convert_file("data.xlsx", "output.pdf", template="modern")

    # Convert directory
    converted = service.convert_directory("./excel_files/", "./pdf_output/")
    print(f"Converted {len(converted)} files")
```

## 📚 Additional Resources

* [CLI Reference](cli-reference.md) - Complete command documentation
* [Configuration Guide](configuration.md) - Advanced configuration options
* [Performance Optimization](performance.md) - Performance tuning
* [Troubleshooting](troubleshooting.md) - Common issues and solutions

---

!!! info "Need Help?"
    * Check the [troubleshooting guide](troubleshooting.md) for common issues
    * Review the [CLI reference](cli-reference.md) for detailed command options
    * [Open an issue](https://github.com/exc-to-pdf/exc-to-pdf/issues) for bugs or feature requests