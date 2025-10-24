"""
Excel file recovery and repair utilities.

This module provides comprehensive Excel file corruption detection,
repair mechanisms, and fallback reading strategies for robust
Excel processing when standard methods fail.
"""

import logging
import os
import re
import shutil
import tempfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ExcelErrorType(Enum):
    """Excel error types for categorization."""
    CORRUPTION = "file_corruption"
    FORMAT_UNSUPPORTED = "format_unsupported"
    PERMISSION_DENIED = "permission_denied"
    MEMORY_ERROR = "memory_error"
    MISSING_SHEETS = "missing_sheets"
    INVALID_SPECIFICATION = "invalid_specification"


@dataclass
class ExcelProcessingResult:
    """Result of Excel processing with detailed error information."""
    success: bool
    data: Optional[Any] = None
    error_type: Optional[ExcelErrorType] = None
    error_message: Optional[str] = None
    fallback_used: Optional[str] = None
    repair_attempts: List[str] = None
    sheets_detected: Optional[int] = None

    def __post_init__(self):
        if self.repair_attempts is None:
            self.repair_attempts = []


class ExcelFileRepairer:
    """
    Excel file repair and recovery utility.

    Provides comprehensive repair mechanisms for corrupted Excel files,
    with focus on workbook.xml sanitization and ZIP structure repair.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_corruption_indicators(self, file_path: Union[str, Path]) -> List[str]:
        """
        Detect various forms of Excel file corruption.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of corruption indicator descriptions
        """
        file_path = Path(file_path)
        corruption_indicators = []

        try:
            # Check file format support
            if file_path.suffix.lower() not in ['.xlsx', '.xlsm', '.xltx', '.xltm']:
                corruption_indicators.append("Unsupported file format")
                return corruption_indicators

            # Check if file is empty
            if file_path.stat().st_size == 0:
                corruption_indicators.append("Empty file")
                return corruption_indicators

            # Test ZIP archive integrity
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    required_files = ['xl/workbook.xml', '[Content_Types].xml']
                    for req_file in required_files:
                        if req_file not in zf.namelist():
                            corruption_indicators.append(f"Missing essential file: {req_file}")

                    # Check for worksheets
                    worksheet_files = [f for f in zf.namelist() if f.startswith('xl/worksheets/sheet')]
                    if len(worksheet_files) == 0:
                        corruption_indicators.append("No worksheet files found")

            except zipfile.BadZipFile:
                corruption_indicators.append("Corrupt ZIP archive")
                return corruption_indicators

            # Test workbook.xml for invalid specifications
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    with zf.open('xl/workbook.xml') as wb_file:
                        wb_content = wb_file.read().decode('utf-8', errors='ignore')

                        # Check for absolute paths (common corruption cause)
                        if re.search(r'[A-Z]:\\[^<>:"/\\|?*\x00-\x1f]*', wb_content):
                            corruption_indicators.append("Absolute file paths found in workbook.xml")

                        # Check for file URLs that cause issues
                        if re.search(r'file:///[^<>\s]*', wb_content):
                            corruption_indicators.append("File URLs found in workbook.xml")

                        # Check for mc: prefix issues that cause XML parsing problems
                        if 'mc:' in wb_content and 'xmlns:mc=' not in wb_content:
                            corruption_indicators.append("Unbound XML namespace prefix found")

                        # Check for malformed XML
                        try:
                            ET.fromstring(wb_content)
                        except ET.ParseError as e:
                            corruption_indicators.append(f"Malformed workbook.xml: {str(e)}")

            except Exception as e:
                corruption_indicators.append(f"Error reading workbook.xml: {str(e)}")

            # Test if openpyxl can read sheets (this is the actual problem)
            try:
                import openpyxl
                wb = openpyxl.load_workbook(file_path, read_only=True)
                sheet_count = len(wb.sheetnames)
                wb.close()

                if sheet_count == 0:
                    corruption_indicators.append("Zero sheets detected by openpyxl - invalid specification")

            except Exception as e:
                # Check if it's the specific "invalid specification" error
                if "invalid specification for" in str(e).lower():
                    corruption_indicators.append("Invalid workbook specification detected")
                else:
                    corruption_indicators.append(f"openpyxl reading error: {str(e)}")

        except Exception as e:
            corruption_indicators.append(f"General corruption detection error: {str(e)}")

        self.logger.info(f"Corruption indicators detected: {len(corruption_indicators)}")
        for indicator in corruption_indicators:
            self.logger.debug(f"  - {indicator}")

        return corruption_indicators

    def sanitize_workbook_xml(self, xml_content: str) -> str:
        """
        Sanitize workbook.xml to remove problematic content.

        Args:
            xml_content: Raw XML content from workbook.xml

        Returns:
            Sanitized XML content

        Raises:
            ValueError: If XML sanitization fails
        """
        try:
            self.logger.debug("Starting workbook.xml sanitization")

            # Remove absolute file paths that cause corruption
            # Pattern matches Windows absolute paths like C:\Users\...
            absolute_path_pattern = r'[A-Z]:\\[^<>:"/\\|?*\x00-\x1f]*'
            sanitized = re.sub(absolute_path_pattern, '', xml_content)

            # Remove file URLs that might cause issues
            file_url_pattern = r'file:///[^\s<>"]*'
            sanitized = re.sub(file_url_pattern, '', sanitized)

            # Parse and re-serialize to ensure valid XML structure
            try:
                root = ET.fromstring(sanitized)

                # Ensure worksheet references are valid
                for sheet in root.findall('.//{http://purl.oclc.org/ooxml/spreadsheetml/main}sheet'):
                    name = sheet.get('name', '')
                    sheet_id = sheet.get('sheetId', '')

                    # Sanitize sheet names
                    if not name:
                        sheet.set('name', f"Sheet_{sheet_id}")
                    elif '/' in name or '\\' in name or ':' in name:
                        # Replace problematic characters
                        clean_name = re.sub(r'[\\/:\*\?"<>|]', '_', name)
                        sheet.set('name', clean_name)

                # Remove problematic elements
                for elem in root.findall('.//*[@mc:Ignorable]'):
                    if elem.get('mc:Ignorable'):
                        # Remove Microsoft compatibility attributes that might cause issues
                        elem.attrib.pop('mc:Ignorable', None)

                # Re-serialize to clean XML
                namespaces = {
                    'main': 'http://purl.oclc.org/ooxml/spreadsheetml/main',
                    'r': 'http://purl.oclc.org/ooxml/officeDocument/relationships'
                }

                for prefix, uri in namespaces.items():
                    ET.register_namespace(prefix, uri)

                sanitized_xml = ET.tostring(root, encoding='unicode', method='xml')

                self.logger.debug("Workbook.xml sanitization completed successfully")
                return sanitized_xml

            except ET.ParseError as e:
                # If XML parsing fails, try basic cleanup
                self.logger.warning(f"XML parsing failed during sanitization: {e}")
                # Remove problematic elements with regex
                sanitized = re.sub(r'<[^>]*mc:[^>]*>', '', sanitized)
                sanitized = re.sub(r'[^<]*mc:[^=]*="[^"]*"[^>]*>', '', sanitized)
                return sanitized

        except Exception as e:
            error_msg = f"XML sanitization failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e

    def repair_excel_zip_structure(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Attempt to repair corrupted ZIP structure of Excel files.

        Args:
            file_path: Path to the Excel file

        Returns:
            Tuple of (success: bool, message: str)
        """
        file_path = Path(file_path)

        try:
            self.logger.info(f"Attempting ZIP structure repair for: {file_path}")

            # Create backup
            backup_path = file_path.with_suffix('.backup.xlsx')
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract what we can from the corrupted file
                try:
                    with zipfile.ZipFile(file_path, 'r') as zin:
                        zin.extractall(temp_path)
                        self.logger.debug(f"Extracted {len(zin.namelist())} files to temporary directory")

                except Exception as e:
                    error_msg = f"Failed to extract ZIP content: {str(e)}"
                    self.logger.error(error_msg)
                    return False, error_msg

                # Sanitize workbook.xml if present
                workbook_path = temp_path / 'xl' / 'workbook.xml'
                if workbook_path.exists():
                    try:
                        with open(workbook_path, 'r', encoding='utf-8') as f:
                            wb_content = f.read()

                        sanitized_content = self.sanitize_workbook_xml(wb_content)

                        with open(workbook_path, 'w', encoding='utf-8') as f:
                            f.write(sanitized_content)

                        self.logger.debug("Workbook.xml sanitized successfully")

                    except Exception as e:
                        error_msg = f"Failed to sanitize workbook.xml: {str(e)}"
                        self.logger.warning(error_msg)
                        # Continue with the repair even if sanitization fails

                # Recreate ZIP with proper compression and structure
                try:
                    with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
                        # Add all files from temp directory
                        for root, dirs, files in os.walk(temp_path):
                            for file in files:
                                file_path_full = os.path.join(root, file)
                                # Calculate archive path relative to temp directory
                                arcname = os.path.relpath(file_path_full, temp_dir)

                                # Use forward slashes for ZIP compatibility
                                arcname = arcname.replace('\\', '/')

                                zout.write(file_path_full, arcname)

                    self.logger.info("ZIP structure repair completed successfully")
                    return True, "ZIP structure repaired successfully"

                except Exception as e:
                    # Restore from backup if repair failed
                    shutil.copy2(backup_path, file_path)
                    error_msg = f"Failed to recreate ZIP file: {str(e)}"
                    self.logger.error(error_msg)
                    return False, error_msg

        except Exception as e:
            error_msg = f"ZIP repair process failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
        finally:
            # Clean up backup file if repair was successful
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                pass

    def extract_raw_worksheet_content(self, file_path: Union[str, Path]) -> Dict[str, List[List[str]]]:
        """
        Extract raw worksheet content when standard reading fails.

        Args:
            file_path: Path to the Excel file

        Returns:
            Dictionary mapping sheet names to raw data
        """
        file_path = Path(file_path)
        worksheets_data = {}

        try:
            self.logger.info("Attempting raw worksheet content extraction")

            with zipfile.ZipFile(file_path, 'r') as zf:
                # First, get sheet names from workbook.xml
                sheet_names = []
                try:
                    with zf.open('xl/workbook.xml') as wb_file:
                        wb_content = wb_file.read().decode('utf-8', errors='ignore')

                        # Extract sheet names using regex
                        sheet_pattern = r'<sheet[^>]*name="([^"]*)"[^>]*sheetId="(\d+)"'
                        matches = re.findall(sheet_pattern, wb_content)

                        for name, sheet_id in matches:
                            if name:
                                sheet_names.append((name, sheet_id))

                except Exception as e:
                    self.logger.warning(f"Failed to extract sheet names: {e}")
                    # Fallback: look for sheet files directly
                    sheet_files = sorted([f for f in zf.namelist() if f.startswith('xl/worksheets/sheet')])
                    for i, sheet_file in enumerate(sheet_files, 1):
                        sheet_names.append((f"Sheet_{i}", str(i)))

                # Extract data from each worksheet
                for sheet_name, sheet_id in sheet_names:
                    try:
                        worksheet_file = f'xl/worksheets/sheet{sheet_id}.xml'
                        if worksheet_file not in zf.namelist():
                            continue

                        with zf.open(worksheet_file) as ws_file:
                            ws_content = ws_file.read().decode('utf-8', errors='ignore')

                            # Extract cell values using regex
                            cell_pattern = r'<c[^>]*>(?:<v>([^<]*)</v>)?(?:<is><t>([^<]*)</t></is>)?</c>'
                            matches = re.findall(cell_pattern, ws_content)

                            # Extract shared strings if available
                            shared_strings = []
                            try:
                                with zf.open('xl/sharedStrings.xml') as ss_file:
                                    ss_content = ss_file.read().decode('utf-8', errors='ignore')
                                    shared_string_pattern = r'<t[^>]*>([^<]*)</t>'
                                    shared_strings = re.findall(shared_string_pattern, ss_content)
                            except Exception:
                                pass

                            # Process cell data
                            row_data = []
                            current_row = []

                            for value_num, text_value in matches:
                                cell_value = value_num if value_num else text_value

                                # Handle shared string references
                                if cell_value.isdigit() and shared_strings:
                                    try:
                                        idx = int(cell_value)
                                        if 0 <= idx < len(shared_strings):
                                            cell_value = shared_strings[idx]
                                    except (ValueError, IndexError):
                                        pass

                                current_row.append(cell_value)

                            # Simple row grouping (this is a basic approach)
                            if current_row:
                                worksheets_data[sheet_name] = [current_row]

                            self.logger.debug(f"Extracted raw data for sheet: {sheet_name}")

                    except Exception as e:
                        self.logger.warning(f"Failed to extract data from sheet {sheet_name}: {e}")
                        continue

                self.logger.info(f"Raw extraction completed for {len(worksheets_data)} sheets")
                return worksheets_data

        except Exception as e:
            error_msg = f"Raw content extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {}

    def create_repair_summary(self, repair_attempts: List[str]) -> Dict[str, Any]:
        """
        Create a summary of repair attempts for logging purposes.

        Args:
            repair_attempts: List of repair attempt descriptions

        Returns:
            Dictionary with repair summary
        """
        return {
            "total_attempts": len(repair_attempts),
            "attempts": repair_attempts,
            "success_rate": 0.0,  # Will be calculated by caller
            "repair_categories": {
                "xml_sanitization": any("XML" in attempt for attempt in repair_attempts),
                "zip_repair": any("ZIP" in attempt for attempt in repair_attempts),
                "raw_extraction": any("raw" in attempt.lower() for attempt in repair_attempts),
            }
        }


def detect_excel_corruption(file_path: Union[str, Path]) -> List[str]:
    """
    Convenience function to detect Excel file corruption.

    Args:
        file_path: Path to the Excel file

    Returns:
        List of corruption indicator descriptions
    """
    repairer = ExcelFileRepairer()
    return repairer.detect_corruption_indicators(file_path)


def sanitize_workbook_xml(xml_content: str) -> str:
    """
    Convenience function to sanitize workbook XML content.

    Args:
        xml_content: Raw XML content

    Returns:
        Sanitized XML content
    """
    repairer = ExcelFileRepairer()
    return repairer.sanitize_workbook_xml(xml_content)


def repair_excel_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Convenience function to repair Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        Tuple of (success: bool, message: str)
    """
    repairer = ExcelFileRepairer()
    return repairer.repair_excel_zip_structure(file_path)