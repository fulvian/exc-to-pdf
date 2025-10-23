---
title: CLI Reference
description: Complete command-line interface reference
---

# CLI Reference

This page provides comprehensive documentation for the exc-to-pdf command-line interface, including all commands, options, and usage examples.

## 📋 Commands Overview

exc-to-pdf provides a hierarchical command structure:

```
exc-to-pdf [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Available Commands

* **`convert`** - Convert Excel files to PDF
* **`config`** - Configuration management
  * **`validate`** - Validate configuration files
  * **`template`** - Generate configuration templates

## 🌐 Global Options

These options apply to all commands:

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--verbose` | `-v` | Enable detailed output | `--verbose` |
| `--quiet` | `-q` | Suppress output except errors | `--quiet` |
| `--version` | | Show version information | `--version` |
| `--help` | `-h` | Show help message | `--help` |

### Usage Examples

```bash
# Show version
exc-to-pdf --version

# Enable verbose output for all commands
exc-to-pdf --verbose convert data.xlsx output.pdf

# Suppress output
exc-to-pdf --quiet convert data.xlsx output.pdf

# Get help
exc-to-pdf --help
exc-to-pdf convert --help
```

## 🔄 Convert Command

The `convert` command is the primary command for converting Excel files to PDF.

### Syntax

```bash
exc-to-pdf convert [OPTIONS] INPUT_FILE [OUTPUT_FILE]
```

### Arguments

| Argument | Description | Required | Example |
|----------|-------------|----------|---------|
| `INPUT_FILE` | Path to Excel file to convert | Yes | `data.xlsx` |
| `OUTPUT_FILE` | Path for output PDF file | No | `output.pdf` |

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | None | Path to configuration file |
| `--template` | `-t` | CHOICE | `modern` | PDF template style |
| `--orientation` | `-o` | CHOICE | `portrait` | Page orientation |
| `--sheet` | `-s` | STRING | None | Specific worksheet name |
| `--no-bookmarks` | | FLAG | False | Disable bookmark generation |
| `--no-metadata` | | FLAG | False | Disable AI metadata |
| `--margin-top` | | FLOAT | 72 | Top margin in points |
| `--margin-bottom` | | FLOAT | 72 | Bottom margin in points |
| `--margin-left` | | FLOAT | 72 | Left margin in points |
| `--margin-right` | | FLOAT | 72 | Right margin in points |

### Template Choices

| Value | Description |
|-------|-------------|
| `modern` | Clean, contemporary design (default) |
| `classic` | Traditional, formal appearance |
| `minimal` | Simple, uncluttered layout |

### Orientation Choices

| Value | Description |
|-------|-------------|
| `portrait` | Standard vertical orientation (default) |
| `landscape` | Horizontal orientation for wide content |

### Examples

#### Basic Conversion

```bash
# Simple conversion
exc-to-pdf convert data.xlsx output.pdf

# Convert with automatic output naming
exc-to-pdf convert report.xlsx
# Creates report.pdf
```

#### Template Selection

```bash
# Use modern template
exc-to-pdf convert data.xlsx output.pdf --template modern

# Use classic template
exc-to-pdf convert data.xlsx output.pdf --template classic

# Use minimal template
exc-to-pdf convert data.xlsx output.pdf --template minimal
```

#### Page Configuration

```bash
# Landscape orientation
exc-to-pdf convert wide-data.xlsx output.pdf --orientation landscape

# Custom margins (in points)
exc-to-pdf convert data.xlsx output.pdf \
  --margin-top 50 \
  --margin-bottom 50 \
  --margin-left 40 \
  --margin-right 40
```

#### Worksheet Selection

```bash
# Convert specific worksheet
exc-to-pdf convert workbook.xlsx output.pdf --sheet "Sales Data"

# Convert worksheet with spaces
exc-to-pdf convert workbook.xlsx output.pdf --sheet "Q4 Results"
```

#### Feature Control

```bash
# Disable bookmarks
exc-to-pdf convert data.xlsx output.pdf --no-bookmarks

# Disable metadata
exc-to-pdf convert data.xlsx output.pdf --no-metadata

# Disable both features
exc-to-pdf convert data.xlsx output.pdf --no-bookmarks --no-metadata
```

#### Configuration File

```bash
# Use configuration file
exc-to-pdf convert data.xlsx output.pdf --config my-config.toml

# Configuration file example (my-config.toml)
[page]
template = "modern"
orientation = "landscape"
margin_top = 80
margin_bottom = 80

[features]
include_bookmarks = true
include_metadata = true
```

#### Verbosity Control

```bash
# Verbose output (shows progress)
exc-to-pdf convert large-file.xlsx output.pdf --verbose

# Quiet output (silent except errors)
exc-to-pdf convert data.xlsx output.pdf --quiet

# Both verbose and quiet (quiet takes precedence)
exc-to-pdf convert data.xlsx output.pdf --verbose --quiet
```

#### Complex Example

```bash
# Full example with all options
exc-to-pdf convert \
  financial-report.xlsx \
  financial-report.pdf \
  --config company-config.toml \
  --template modern \
  --orientation landscape \
  --sheet "Q4 Results" \
  --margin-top 60 \
  --margin-bottom 60 \
  --margin-left 50 \
  --margin-right 50 \
  --verbose
```

## ⚙️ Config Command

The `config` command group provides configuration management utilities.

### Config Validate

Validate configuration files for syntax and correctness.

#### Syntax

```bash
exc-to-pdf config validate [OPTIONS] --config CONFIG_FILE
```

#### Options

| Option | Short | Type | Required | Description |
|--------|-------|------|----------|-------------|
| `--config` | `-c` | PATH | Yes | Configuration file to validate |

#### Examples

```bash
# Validate configuration file
exc-to-pdf config validate --config my-config.toml

# Validate with verbose output
exc-to-pdf --verbose config validate --config my-config.toml
```

#### Validation Checks

The validation checks for:

* **Syntax errors** - Invalid TOML syntax
* **Required fields** - Missing required configuration
* **Value validation** - Invalid option values
* **File references** - Referenced files exist and are readable

### Config Template

Generate configuration file templates for customization.

#### Syntax

```bash
exc-to-pdf config template [OPTIONS]
```

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output` | `-o` | PATH | `exc-to-pdf-config.toml` | Output configuration file |

#### Examples

```bash
# Generate default template
exc-to-pdf config template

# Generate with custom filename
exc-to-pdf config template --output my-config.toml

# Generate in specific directory
exc-to-pdf config template --output ./config/production.toml
```

#### Generated Template

The generated template includes:

```toml
# exc-to-pdf Configuration Template
# Generated on: 2025-10-22

[page]
# Template style: modern, classic, minimal
template = "modern"

# Page orientation: portrait, landscape
orientation = "portrait"

# Margins in points (72 points = 1 inch)
margin_top = 72
margin_bottom = 72
margin_left = 72
margin_right = 72

[features]
# Include automatic bookmarks for navigation
include_bookmarks = true

# Include AI-optimized metadata
include_metadata = true

# Optimize for Google NotebookLM analysis
optimize_for_notebooklm = true

[processing]
# Memory usage limit in MB (0 = no limit)
memory_limit = 0

# Cache processed data for faster repeated conversions
enable_cache = true

# Maximum cache size in MB
cache_size = 100

[output]
# PDF compression level: none, low, medium, high
compression = "medium"

# Image quality for embedded images (1-100)
image_quality = 85

# Embed fonts in PDF
embed_fonts = true
```

## 📊 Exit Codes

exc-to-pdf uses standard exit codes to indicate success or failure:

| Exit Code | Description |
|-----------|-------------|
| `0` | Success |
| `1` | General error |
| `2` | File not found or inaccessible |
| `3` | Configuration error |
| `4` | Permission denied |
| `5` | Memory or resource error |
| `130` | Interrupted (Ctrl+C) |

### Using Exit Codes

```bash
# Check exit code in shell script
exc-to-pdf convert data.xlsx output.pdf
if [ $? -eq 0 ]; then
    echo "✅ Conversion successful"
else
    echo "❌ Conversion failed with exit code $?"
fi

# Use in conditional
exc-to-pdf convert data.xlsx output.pdf && echo "Success" || echo "Failed"
```

## 🔧 Environment Variables

exc-to-pdf can be configured using environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `EXC_TO_PDF_CONFIG` | Default configuration file path | None | `/path/to/config.toml` |
| `EXC_TO_PDF_TEMPLATE` | Default template style | `modern` | `classic` |
| `EXC_TO_PDF_VERBOSE` | Enable verbose output | `false` | `true` |
| `EXC_TO_PDF_QUIET` | Enable quiet mode | `false` | `true` |
| `EXC_TO_PDF_CACHE_DIR` | Cache directory | `~/.exc-to-pdf/cache` | `/tmp/cache` |

### Usage Examples

```bash
# Set default template
export EXC_TO_PDF_TEMPLATE=classic
exc-to-pdf convert data.xlsx output.pdf  # Uses classic template

# Set default configuration
export EXC_TO_PDF_CONFIG=~/.exc-to-pdf/config.toml
exc-to-pdf convert data.xlsx output.pdf  # Uses config file

# Enable verbose output
export EXC_TO_PDF_VERBOSE=true
exc-to-pdf convert data.xlsx output.pdf  # Verbose output

# Temporary override
EXC_TO_PDF_TEMPLATE=minimal exc-to-pdf convert data.xlsx output.pdf
```

## 📝 Output Formats

### Standard Output

Normal operation produces status messages:

```bash
$ exc-to-pdf convert data.xlsx output.pdf
📄 Converting: data.xlsx
📄 Output:    output.pdf
🎨 Template:  modern
📐 Orientation: portrait
🔄 Processing Excel file...
✅ Conversion completed successfully!
📊 Output size: 2.34 MB
```

### Verbose Output

Verbose mode provides detailed progress information:

```bash
$ exc-to-pdf --verbose convert data.xlsx output.pdf
🔧 Initializing PDF generator...
📄 Converting: data.xlsx
📄 Output:    output.pdf
🎨 Template:  modern
📐 Orientation: portrait
📋 Found 3 worksheets:
  1. Sheet1
  2. Sales Data
  3. Summary
🔄 Processing worksheet: Sheet1
📊 Detected 1 table (15 rows × 8 columns)
📄 Rendering table to PDF...
🔄 Processing worksheet: Sales Data
📊 Detected 1 table (245 rows × 12 columns)
📄 Rendering table to PDF...
🔄 Processing worksheet: Summary
📊 Detected 2 tables (12 rows × 6 columns, 8 rows × 4 columns)
📄 Rendering tables to PDF...
📑 Creating bookmarks...
🏷️  Adding metadata for AI optimization...
✅ Conversion completed successfully!
📊 Output size: 4.67 MB
📈 Processing time: 3.2 seconds
```

### Error Output

Error messages provide helpful context:

```bash
$ exc-to-pdf convert missing.xlsx output.pdf
❌ Error: File not found: missing.xlsx

💡 Suggestions:
   • Check if the file exists and is readable
   • Ensure the file is a valid Excel file (.xlsx, .xls)
   • Try using the absolute file path
```

## 🔄 Batch Processing

### Shell Script Example

```bash
#!/bin/bash
# batch-convert.sh - Convert multiple Excel files

INPUT_DIR="./excel_files"
OUTPUT_DIR="./pdf_output"
CONFIG_FILE="./config.toml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all Excel files
for excel_file in "$INPUT_DIR"/*.xlsx; do
    if [ -f "$excel_file" ]; then
        filename=$(basename "$excel_file" .xlsx)
        pdf_file="$OUTPUT_DIR/${filename}.pdf"

        echo "🔄 Converting: $filename"

        exc-to-pdf convert "$excel_file" "$pdf_file" \
            --config "$CONFIG_FILE" \
            --verbose

        # Check exit code
        if [ $? -eq 0 ]; then
            echo "✅ Success: $filename"
        else
            echo "❌ Failed: $filename"
        fi
    fi
done

echo "🎯 Batch processing completed"
```

### Find Command Example

```bash
# Convert all Excel files recursively
find . -name "*.xlsx" -type f -exec exc-to-pdf convert {} {}.pdf \;

# Convert with parallel processing (requires GNU parallel)
find . -name "*.xlsx" | parallel -j 4 exc-to-pdf convert {} {.}.pdf
```

## 🔍 Troubleshooting CLI Issues

### Common Command Errors

#### File Not Found

```bash
$ exc-to-pdf convert non-existent.xlsx output.pdf
❌ Error: File not found: non-existent.xlsx

# Solutions:
# 1. Check file exists
ls -la non-existent.xlsx

# 2. Use absolute path
exc-to-pdf convert /full/path/to/file.xlsx output.pdf

# 3. Check current directory
pwd
ls -la *.xlsx
```

#### Permission Denied

```bash
$ exc-to-pdf convert /protected/file.xlsx output.pdf
❌ Error: Permission denied: /protected/file.xlsx

# Solutions:
# 1. Check permissions
ls -la /protected/file.xlsx

# 2. Use different output directory
exc-to-pdf convert input.xlsx ~/output.pdf

# 3. Fix permissions (if possible)
chmod 644 /protected/file.xlsx
```

#### Invalid Configuration

```bash
$ exc-to-pdf convert data.xlsx output.pdf --config invalid.toml
❌ Error: Configuration validation failed: Invalid TOML syntax

# Solutions:
# 1. Validate configuration
exc-to-pdf config validate --config invalid.toml

# 2. Generate new template
exc-to-pdf config template --output new-config.toml
```

### Debug Mode

For detailed troubleshooting, use verbose mode:

```bash
# Enable verbose output
exc-to-pdf --verbose convert problematic.xlsx output.pdf

# Check configuration
exc-to-pdf --verbose config validate --config config.toml
```

---

!!! info "Need More Help?"
    * Check the [User Guide](index.md) for general usage information
    * Review the [Troubleshooting](troubleshooting.md) guide for common issues
    * [Open an issue](https://github.com/exc-to-pdf/exc-to-pdf/issues) for bugs or feature requests