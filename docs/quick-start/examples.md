---
title: Examples
description: Practical examples and use cases for exc-to-pdf
---

# Examples

This page provides practical, real-world examples of using exc-to-pdf for various scenarios. Each example includes the command-line usage and, where applicable, the Python API equivalent.

## 📊 Business Reports

### Example 1: Financial Statements

Convert quarterly financial reports with professional formatting.

```bash
# Convert financial report with landscape orientation for wide tables
exc-to-pdf convert Q3-Financials.xlsx Q3-Financials.pdf \
  --template modern \
  --orientation landscape \
  --margin-top 60 \
  --margin-bottom 60 \
  --margin-left 40 \
  --margin-right 40 \
  --verbose
```

**Python API:**
```python
from exc_to_pdf import PDFGenerator
from exc_to_pdf.config import PDFConfig

# Configure for financial reports
config = PDFConfig()
config.table_style = "modern"
config.orientation = "landscape"
config.margin_top = 60
config.margin_bottom = 60
config.margin_left = 40
config.margin_right = 40

generator = PDFGenerator(config)
generator.convert_excel_to_pdf("Q3-Financials.xlsx", "Q3-Financials.pdf")
```

### Example 2: Sales Dashboard

Convert sales dashboards with multiple visualizations.

```bash
# Convert sales dashboard with all worksheets
exc-to-pdf convert Sales-Dashboard-2024.xlsx Sales-Report.pdf \
  --template classic \
  --orientation portrait \
  --include-bookmarks \
  --verbose
```

### Example 3: Executive Summary

Create a concise executive summary from a detailed workbook.

```bash
# Convert only the summary worksheet
exc-to-pdf convert Annual-Report.xlsx Executive-Summary.pdf \
  --sheet "Executive Summary" \
  --template modern \
  --margin-top 72 \
  --margin-bottom 72
```

## 🎓 Academic & Research

### Example 4: Research Data

Convert experimental research data for publication.

```bash
# Convert research data with classic academic styling
exc-to-pdf convert Research-Data.xlsx Research-Paper.pdf \
  --template classic \
  --orientation portrait \
  --sheet "Experimental Results" \
  --no-bookmarks \
  --verbose
```

**Python API for research data:**
```python
from exc_to_pdf import PDFGenerator
import os

def convert_research_data(data_dir, output_dir):
    """Convert all research data files"""
    generator = PDFGenerator()

    research_files = [
        "experimental-data.xlsx",
        "statistical-analysis.xlsx",
        "survey-results.xlsx"
    ]

    for filename in research_files:
        input_path = os.path.join(data_dir, filename)
        output_name = filename.replace('.xlsx', '_paper.pdf')
        output_path = os.path.join(output_dir, output_name)

        try:
            generator.convert_excel_to_pdf(
                input_file=input_path,
                output_file=output_path,
                template="classic"
            )
            print(f"✅ Converted: {filename}")
        except Exception as e:
            print(f"❌ Failed: {filename} - {e}")

# Usage
convert_research_data("./research_data/", "./papers/")
```

### Example 5: Survey Results

Convert survey data with clean, readable formatting.

```bash
# Convert survey results with minimal styling
exc-to-pdf convert Survey-Results-2024.xlsx Survey-Report.pdf \
  --template minimal \
  --orientation portrait \
  --margin-top 80 \
  --margin-bottom 80
```

## 🏭 Data Analysis

### Example 6: Large Dataset Processing

Process large datasets efficiently.

```bash
# Convert large dataset with progress monitoring
exc-to-pdf convert Large-Dataset.xlsx Dataset-Analysis.pdf \
  --template modern \
  --orientation landscape \
  --verbose
```

**Python API for batch processing:**
```python
import os
import time
from exc_to_pdf import PDFGenerator
from pathlib import Path

def process_large_dataset(input_dir, output_dir):
    """Process multiple large datasets with monitoring"""
    generator = PDFGenerator()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    excel_files = list(Path(input_dir).glob("*.xlsx"))

    for i, excel_file in enumerate(excel_files, 1):
        print(f"\n📊 Processing {i}/{len(excel_files)}: {excel_file.name}")

        output_file = Path(output_dir) / f"{excel_file.stem}.pdf"

        start_time = time.time()
        try:
            generator.convert_excel_to_pdf(
                input_file=str(excel_file),
                output_file=str(output_file)
            )

            elapsed = time.time() - start_time
            file_size = output_file.stat().st_size / (1024 * 1024)

            print(f"✅ Completed in {elapsed:.1f}s - Size: {file_size:.2f}MB")

        except Exception as e:
            print(f"❌ Failed: {e}")

# Usage
process_large_dataset("./large_datasets/", "./processed_pdfs/")
```

### Example 7: Data Validation Report

Create a data validation report from multiple sources.

```bash
# Convert validation report with bookmarks for navigation
exc-to-pdf convert Data-Validation.xlsx Validation-Report.pdf \
  --template modern \
  --include-bookmarks \
  --verbose
```

## 📈 Business Intelligence

### Example 8: Monthly KPI Report

Generate monthly KPI reports with consistent styling.

```bash
# Convert monthly KPI report
exc-to-pdf convert KPI-Report-October-2024.xlsx KPI-Report-Oct.pdf \
  --template modern \
  --orientation portrait \
  --margin-top 50 \
  --margin-bottom 50
```

**Python API for automated monthly reports:**
```python
from datetime import datetime
from exc_to_pdf import PDFGenerator
import os

def generate_monthly_reports(reports_dir, output_dir):
    """Generate monthly KPI reports automatically"""
    generator = PDFGenerator()

    # Get current month
    current_month = datetime.now().strftime("%B-%Y")

    # Find monthly report files
    report_files = [f for f in os.listdir(reports_dir)
                   if f.startswith("KPI-Report-") and f.endswith(".xlsx")]

    for report_file in report_files:
        input_path = os.path.join(reports_dir, report_file)

        # Create output filename
        month_name = report_file.replace("KPI-Report-", "").replace(".xlsx", "")
        output_file = f"KPI-Report-{month_name}.pdf"
        output_path = os.path.join(output_dir, output_file)

        try:
            generator.convert_excel_to_pdf(
                input_file=input_path,
                output_file=output_path,
                template="modern"
            )
            print(f"✅ Generated: {output_file}")
        except Exception as e:
            print(f"❌ Failed: {report_file} - {e}")

# Usage (would be run monthly via cron/scheduled task)
generate_monthly_reports("./monthly_reports/", "./kpi_pdfs/")
```

### Example 9: Dashboard Export

Export dashboard data for sharing.

```bash
# Convert dashboard with landscape orientation
exc-to-pdf convert Dashboard-Export.xlsx Dashboard-Share.pdf \
  --template modern \
  --orientation landscape \
  --no-metadata \
  --quiet
```

## 🔄 Batch Processing Examples

### Example 10: Department Reports

Process reports for multiple departments.

```bash
#!/bin/bash
# Process all department reports

DEPARTMENTS=("Sales" "Marketing" "Finance" "HR" "Operations")
TEMPLATE="modern"
OUTPUT_DIR="./department_reports/"

mkdir -p "$OUTPUT_DIR"

for dept in "${DEPARTMENTS[@]}"; do
    echo "📊 Processing $dept Department..."

    input_file="./source_reports/${dept}-Report.xlsx"
    output_file="${OUTPUT_DIR}${dept}-Report.pdf"

    if [ -f "$input_file" ]; then
        exc-to-pdf convert "$input_file" "$output_file" \
            --template "$TEMPLATE" \
            --orientation portrait \
            --verbose
    else
        echo "⚠️  File not found: $input_file"
    fi
done

echo "✅ Department reports processing completed!"
```

### Example 11: Time-Based Processing

Process files based on creation/modification time.

```python
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from exc_to_pdf import PDFGenerator

def process_recent_files(input_dir, output_dir, days=7):
    """Process Excel files modified in the last N days"""
    generator = PDFGenerator()

    cutoff_date = datetime.now() - timedelta(days=days)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for excel_file in input_path.glob("*.xlsx"):
        # Check file modification time
        mod_time = datetime.fromtimestamp(excel_file.stat().st_mtime)

        if mod_time > cutoff_date:
            print(f"📄 Processing: {excel_file.name}")

            output_file = output_path / f"{excel_file.stem}.pdf"

            try:
                generator.convert_excel_to_pdf(
                    input_file=str(excel_file),
                    output_file=str(output_file)
                )
                processed_count += 1
                print(f"✅ Converted: {excel_file.name}")

            except Exception as e:
                print(f"❌ Failed: {excel_file.name} - {e}")

    print(f"\n🎯 Processed {processed_count} recent files")

# Usage - process files from last 7 days
process_recent_files("./incoming_reports/", "./recent_pdfs/", days=7)
```

## 🔧 Advanced Configuration Examples

### Example 12: Custom Template Workflow

Create a workflow with different templates for different content types.

```python
from exc_to_pdf import PDFGenerator
from pathlib import Path
import os

def intelligent_conversion(input_dir, output_dir):
    """Convert files with appropriate templates based on content"""

    # Template selection rules
    template_rules = {
        "financial": "classic",
        "dashboard": "modern",
        "data": "minimal",
        "report": "modern"
    }

    generator = PDFGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for excel_file in Path(input_dir).glob("*.xlsx"):
        filename = excel_file.name.lower()

        # Determine template based on filename
        selected_template = "modern"  # default
        for keyword, template in template_rules.items():
            if keyword in filename:
                selected_template = template
                break

        print(f"📄 {excel_file.name} → Template: {selected_template}")

        output_file = output_path / f"{excel_file.stem}.pdf"

        try:
            generator.convert_excel_to_pdf(
                input_file=str(excel_file),
                output_file=str(output_file),
                template=selected_template
            )
            print(f"✅ Converted with {selected_template} template")

        except Exception as e:
            print(f"❌ Failed: {e}")

# Usage
intelligent_conversion("./mixed_reports/", "./smart_output/")
```

### Example 13: Performance Monitoring

Monitor conversion performance and statistics.

```python
import time
import json
from datetime import datetime
from pathlib import Path
from exc_to_pdf import PDFGenerator

class ConversionMonitor:
    def __init__(self):
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_size_mb": 0,
            "total_time": 0,
            "files": []
        }

    def convert_and_monitor(self, input_file, output_file, **kwargs):
        """Convert file while monitoring performance"""
        generator = PDFGenerator()

        input_path = Path(input_file)
        output_path = Path(output_file)

        # Record start
        start_time = time.time()
        input_size = input_path.stat().st_size / (1024 * 1024)

        self.stats["total_files"] += 1
        self.stats["total_size_mb"] += input_size

        file_stats = {
            "filename": input_path.name,
            "input_size_mb": input_size,
            "start_time": datetime.now().isoformat()
        }

        try:
            generator.convert_excel_to_pdf(input_file, output_file, **kwargs)

            # Record success
            elapsed = time.time() - start_time
            output_size = output_path.stat().st_size / (1024 * 1024)

            file_stats.update({
                "status": "success",
                "elapsed_time": elapsed,
                "output_size_mb": output_size,
                "compression_ratio": output_size / input_size if input_size > 0 else 0
            })

            self.stats["successful"] += 1
            self.stats["total_time"] += elapsed

            print(f"✅ {input_path.name}: {elapsed:.1f}s, {output_size:.2f}MB")

        except Exception as e:
            # Record failure
            file_stats.update({
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            })

            self.stats["failed"] += 1
            print(f"❌ {input_path.name}: {e}")

        self.stats["files"].append(file_stats)

    def get_summary(self):
        """Get performance summary"""
        if self.stats["total_files"] == 0:
            return "No files processed"

        avg_time = self.stats["total_time"] / self.stats["successful"] if self.stats["successful"] > 0 else 0
        success_rate = (self.stats["successful"] / self.stats["total_files"]) * 100

        return {
            "total_files": self.stats["total_files"],
            "success_rate": f"{success_rate:.1f}%",
            "total_input_size_mb": f"{self.stats['total_size_mb']:.2f}",
            "average_time_per_file": f"{avg_time:.1f}s",
            "total_time": f"{self.stats['total_time']:.1f}s"
        }

    def save_report(self, report_file):
        """Save detailed report to JSON file"""
        report = {
            "summary": self.get_summary(),
            "details": self.stats,
            "generated_at": datetime.now().isoformat()
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Report saved: {report_file}")

# Usage
monitor = ConversionMonitor()

# Process files with monitoring
files_to_process = [
    ("report1.xlsx", "report1.pdf"),
    ("report2.xlsx", "report2.pdf"),
    ("report3.xlsx", "report3.pdf")
]

for input_file, output_file in files_to_process:
    monitor.convert_and_monitor(input_file, output_file, template="modern")

# Get summary and save report
print("\n📊 Performance Summary:")
summary = monitor.get_summary()
for key, value in summary.items():
    print(f"  {key}: {value}")

monitor.save_report("conversion_report.json")
```

## 🎯 Best Practices

### General Tips

1. **Choose the right template**:
   * `modern` for business reports
   * `classic` for academic/formal documents
   * `minimal` for data-focused content

2. **Use appropriate orientation**:
   * `portrait` for standard reports
   * `landscape` for wide tables/charts

3. **Monitor large conversions** with `--verbose`

4. **Process in batches** for multiple files

5. **Test with sample files** before processing important data

### Performance Optimization

```bash
# For faster processing of simple files
exc-to-pdf convert simple.xlsx output.pdf \
  --template minimal \
  --no-bookmarks \
  --no-metadata \
  --quiet
```

### Error Handling

```bash
# Use verbose mode to debug issues
exc-to-pdf convert problem.xlsx output.pdf --verbose

# Check file permissions first
ls -la input.xlsx
ls -la output_directory/
```

---

!!! success "Try these examples!"
    Copy and modify these examples for your specific use cases. Each example can be adapted to your data and requirements.

    Need more help? Check the [User Guide](../user-guide/index.md) or [open an issue](https://github.com/exc-to-pdf/exc-to-pdf/issues).