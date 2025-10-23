#!/usr/bin/env python3
"""
Test script to validate code examples in documentation.
This script validates Python API examples from the documentation.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def test_imports():
    """Test that all documented imports work."""
    print("🧪 Testing imports...")

    try:
        from exc_to_pdf import PDFGenerator

        print("✅ PDFGenerator import successful")
    except ImportError as e:
        print(f"❌ PDFGenerator import failed: {e}")
        return False

    try:
        from exc_to_pdf.config import PDFConfig

        print("✅ PDFConfig import successful")
    except ImportError as e:
        print(f"❌ PDFConfig import failed: {e}")
        return False

    try:
        from exc_to_pdf.exceptions import (
            InvalidFileException,
            PDFGenerationException,
            ExcelReaderError,
        )

        print("✅ Exception imports successful")
    except ImportError as e:
        print(f"❌ Exception imports failed: {e}")
        return False

    return True


def test_basic_pdfgenerator_creation():
    """Test basic PDFGenerator creation."""
    print("\n🧪 Testing PDFGenerator creation...")

    try:
        from exc_to_pdf import PDFGenerator

        # Test default constructor
        generator = PDFGenerator()
        print("✅ PDFGenerator() with default config successful")

        # Test with custom config
        from exc_to_pdf.config import PDFConfig

        config = PDFConfig()
        config.table_style = "modern"
        config.orientation = "portrait"

        generator_with_config = PDFGenerator(config)
        print("✅ PDFGenerator(config) successful")

        return True

    except Exception as e:
        print(f"❌ PDFGenerator creation failed: {e}")
        return False


def test_configuration_options():
    """Test configuration options."""
    print("\n🧪 Testing configuration...")

    try:
        from exc_to_pdf.config import PDFConfig

        config = PDFConfig()

        # Test basic configuration
        config.table_style = "modern"
        config.orientation = "landscape"
        config.margin_top = 50
        config.margin_bottom = 50
        config.margin_left = 40
        config.margin_right = 40
        config.include_bookmarks = True
        config.include_metadata = True

        print("✅ Configuration options set successfully")

        # Verify configuration values
        assert config.table_style == "modern"
        assert config.orientation == "landscape"
        assert config.margin_top == 50
        assert config.include_bookmarks == True

        print("✅ Configuration values verified")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling patterns."""
    print("\n🧪 Testing error handling...")

    try:
        from exc_to_pdf.exceptions import (
            InvalidFileException,
            ExcelReaderError,
            PDFGenerationException,
        )

        # Test exception creation
        error = InvalidFileException(
            message="Test file not found",
            file_path="/path/to/test.xlsx",
            context={"operation": "test"},
        )

        # Test exception attributes
        assert error.message == "Test file not found"
        assert error.file_path == "/path/to/test.xlsx"
        assert error.context["operation"] == "test"

        # Test string representation
        error_str = str(error)
        assert "Test file not found" in error_str
        assert "/path/to/test.xlsx" in error_str

        print("✅ Exception handling test successful")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_example_code_structure():
    """Test that example code structure is valid."""
    print("\n🧪 Testing example code structure...")

    # Test basic service class pattern
    try:
        from exc_to_pdf import PDFGenerator

        class ExcelToPDFService:
            """Service class from documentation example"""

            def __init__(self):
                self.generator = PDFGenerator()

            def convert_file(self, input_path: str, output_path: str) -> bool:
                """Convert single Excel file to PDF"""
                try:
                    # Don't actually convert - just test the structure
                    self.generator  # Verify generator exists
                    return True
                except Exception as e:
                    print(f"Conversion failed: {e}")
                    return False

        # Test service creation
        service = ExcelToPDFService()
        print("✅ Service class pattern successful")

        # Test method structure (without actual file operations)
        result = service.convert_file("test.xlsx", "test.pdf")
        print(f"✅ Service method structure: {result}")

        return True

    except Exception as e:
        print(f"❌ Example code structure test failed: {e}")
        return False


def test_batch_processing_pattern():
    """Test batch processing pattern from documentation."""
    print("\n🧪 Testing batch processing pattern...")

    try:
        from exc_to_pdf import PDFGenerator
        import os

        def convert_directory_mock(
            input_dir: str, output_dir: str, template: str = "modern"
        ):
            """Mock version of batch processing from documentation"""
            generator = PDFGenerator()

            # Mock directory processing (don't actually process files)
            if not os.path.exists(input_dir):
                os.makedirs(input_dir, exist_ok=True)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Simulate processing logic structure
            print(f"Would process files in {input_dir}")
            print(f"Would output to {output_dir}")
            print(f"Using template: {template}")

            return True

        # Test with temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")

            result = convert_directory_mock(input_dir, output_dir, "modern")
            assert result == True

        print("✅ Batch processing pattern successful")

        return True

    except Exception as e:
        print(f"❌ Batch processing pattern test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing exc-to-pdf Documentation Examples")
    print("=" * 50)

    tests = [
        test_imports,
        test_basic_pdfgenerator_creation,
        test_configuration_options,
        test_error_handling,
        test_example_code_structure,
        test_batch_processing_pattern,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test {test.__name__} failed!")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Documentation examples are structurally valid.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Review documentation examples.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
