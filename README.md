# exc-to-pdf

Excel to PDF converter optimized for Google NotebookLM analysis.

## 🎯 Overview

**exc-to-pdf** is a Python tool that converts Excel files (.xlsx) into PDF documents specifically optimized for AI analysis with Google NotebookLM. The tool preserves all data, maintains structure, and creates navigation-friendly PDFs that AI systems can effectively analyze.

### Key Features

- 📊 **Multi-sheet Support**: Processes all worksheets in Excel files
- 🔍 **Table Detection**: Automatically identifies and preserves table structures
- 📑 **PDF Navigation**: Creates bookmarks and structured PDF for easy AI navigation
- 🎯 **NotebookLM Optimized**: Text-based PDF output perfect for AI analysis
- ⚡ **High Quality**: 100% data preservation with structured formatting
- 🐍 **Python Powered**: Built with openpyxl, pandas, and reportlab

## 🚀 Quick Start

### Installation

#### From PyPI (Recommended)

```bash
pip install exc-to-pdf
```

#### From Source

```bash
# Clone the repository
git clone https://github.com/fulvian/exc-to-pdf.git
cd exc-to-pdf

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Convert Excel to PDF
exc-to-pdf input.xlsx output.pdf

# With options
exc-to-pdf input.xlsx output.pdf --bookmarks --preserve-formatting

# Python module alternative
python -m exc_to_pdf input.xlsx output.pdf
```

### Python API

```python
from exc_to_pdf.excel_processor import ExcelProcessor
from exc_to_pdf.pdf_generator import PDFGenerator

# Process Excel file
processor = ExcelProcessor("input.xlsx")
sheets_data = processor.extract_all_sheets()

# Generate PDF
generator = PDFGenerator()
generator.create_pdf(sheets_data, "output.pdf")
```

## 📋 Requirements

- Python 3.9+
- Dependencies automatically installed with `pip install exc-to-pdf`

### Core Dependencies

- **openpyxl** (>=3.1.0) - Excel file parsing
- **pandas** (>=2.0.0) - Data processing
- **reportlab** (>=4.0.0) - PDF generation
- **Pillow** (>=10.0.0) - Image handling
- **matplotlib** (>=3.7.0) - Chart recreation
- **babel** (>=2.12.0) - Internationalization
- **numpy** (>=1.24.0) - Numerical computing

## 🏗️ Project Structure

```
exc-to-pdf/
├── src/                    # Source code
│   ├── excel_processor.py  # Excel reading logic
│   ├── pdf_generator.py    # PDF generation
│   ├── table_detector.py   # Table identification
│   └── main.py            # CLI interface
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
│   ├── idee_fondanti/     # Foundational documents
│   └── api/               # API documentation
├── scripts/               # Utility scripts
└── requirements.txt       # Dependencies
```

## 🔄 Development Workflow

This project follows the **DevStream 7-Step Workflow**:

1. **DISCUSS** - Requirements analysis and planning
2. **ANALYZE** - Technical analysis and research
3. **RESEARCH** - Context7 and best practices research
4. **PLAN** - Implementation planning
5. **APPROVE** - Architecture validation
6. **IMPLEMENT** - Code development
7. **VERIFY** - Testing and validation

### Current Development Phase

**Phase**: P1 - Project Foundation ✅
**Next**: P2 - Excel Processing Engine

See [docs/idee_fondanti/piano_fondante_exc-to-pdf.md](docs/idee_fondanti/piano_fondante_exc-to-pdf.md) for complete development plan.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_excel_processor.py
```

## 📊 Architecture

### Data Flow

```
Excel File → openpyxl parsing → pandas processing → reportlab rendering → PDF Output
```

### Key Components

1. **ExcelProcessor**: Reads and parses Excel files
2. **TableDetector**: Identifies table structures
3. **PDFGenerator**: Creates structured PDF output
4. **BookmarkManager**: Adds navigation elements

## 🎯 Google NotebookLM Optimization

The PDF output is specifically designed for AI analysis:

- **Text-based tables** (not images)
- **Structured navigation** with bookmarks
- **Accessibility tags** for better AI understanding
- **Semantic structure** preservation
- **Metadata inclusion** for context

## 📝 Development Status

- [x] Project Foundation (P1)
- [ ] Excel Processing Engine (P2)
- [ ] PDF Generation Engine (P3)
- [ ] Integration & Pipeline (P4)
- [ ] Quality Assurance (P5)
- [ ] Optimization (P6)
- [ ] Documentation & Release (P7)

## 🤝 Contributing

1. Follow DevStream workflow
2. Maintain 95%+ test coverage
3. Use type hints and docstrings
4. Pass code review validation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Google NotebookLM](https://notebooklm.google.com/) - AI-powered notebook
- [openpyxl](https://openpyxl.readthedocs.io/) - Excel file library
- [reportlab](https://www.reportlab.com/) - PDF generation library
- [pandas](https://pandas.pydata.org/) - Data analysis library

---

**Built with ❤️ using DevStream framework**