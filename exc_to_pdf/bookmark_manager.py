"""
PDF bookmark management module for navigation structure.

This module provides bookmark management functionality for creating
hierarchical navigation structures in PDF documents, with support for
sheet-level and table-level bookmarks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from .exceptions import PDFGenerationException

logger = structlog.get_logger()


@dataclass
class BookmarkInfo:
    """Information about a PDF bookmark."""

    title: str
    page_number: int
    level: int
    parent: Optional[str] = None


class BookmarkManager:
    """Manages PDF bookmarks and navigation structure."""

    def __init__(self) -> None:
        """Initialize bookmark manager."""
        self.bookmarks: List[BookmarkInfo] = []
        self.page_counter: int = 0

    def add_sheet_bookmark(self, sheet_name: str, page_number: int) -> BookmarkInfo:
        """Add bookmark for worksheet.

        Args:
            sheet_name: Name of the worksheet
            page_number: Page number where sheet starts

        Returns:
            Created bookmark information

        Raises:
            PDFGenerationException: If bookmark creation fails
        """
        try:
            if not sheet_name or not sheet_name.strip():
                raise ValueError("Sheet name cannot be empty")

            if page_number < 1:
                raise ValueError(f"Page number must be positive, got {page_number}")

            bookmark = BookmarkInfo(
                title=sheet_name.strip(),
                page_number=page_number,
                level=0,  # Sheet bookmarks are top-level
                parent=None,
            )

            self.bookmarks.append(bookmark)
            self.page_counter = max(self.page_counter, page_number)

            logger.debug(
                "Sheet bookmark added",
                extra={
                    "sheet_name": sheet_name,
                    "page_number": page_number,
                    "total_bookmarks": len(self.bookmarks),
                },
            )

            return bookmark

        except ValueError as e:
            logger.error(
                "Failed to add sheet bookmark",
                extra={
                    "sheet_name": sheet_name,
                    "page_number": page_number,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to add sheet bookmark: {e}") from e
        except Exception as e:
            logger.error(
                "Unexpected error adding sheet bookmark",
                extra={
                    "sheet_name": sheet_name,
                    "page_number": page_number,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(
                "Unexpected error adding sheet bookmark"
            ) from e

    def add_table_bookmark(
        self, table_name: str, page_number: int, parent_sheet: str, level: int = 1
    ) -> BookmarkInfo:
        """Add bookmark for table within a sheet.

        Args:
            table_name: Name of the table
            page_number: Page number where table appears
            parent_sheet: Parent sheet name
            level: Bookmark hierarchy level

        Returns:
            Created bookmark information

        Raises:
            PDFGenerationException: If bookmark creation fails
        """
        try:
            if not table_name or not table_name.strip():
                raise ValueError("Table name cannot be empty")

            if not parent_sheet or not parent_sheet.strip():
                raise ValueError("Parent sheet name cannot be empty")

            if page_number < 1:
                raise ValueError(f"Page number must be positive, got {page_number}")

            if level < 1:
                raise ValueError(f"Bookmark level must be positive, got {level}")

            # Verify parent sheet exists
            parent_exists = any(
                bookmark.title == parent_sheet.strip() and bookmark.level == 0
                for bookmark in self.bookmarks
            )

            if not parent_exists:
                logger.warning(
                    "Adding table bookmark for non-existent parent sheet",
                    extra={
                        "table_name": table_name,
                        "parent_sheet": parent_sheet,
                        "page_number": page_number,
                    },
                )

            bookmark = BookmarkInfo(
                title=table_name.strip(),
                page_number=page_number,
                level=level,
                parent=parent_sheet.strip(),
            )

            self.bookmarks.append(bookmark)
            self.page_counter = max(self.page_counter, page_number)

            logger.debug(
                "Table bookmark added",
                extra={
                    "table_name": table_name,
                    "page_number": page_number,
                    "parent_sheet": parent_sheet,
                    "level": level,
                    "total_bookmarks": len(self.bookmarks),
                },
            )

            return bookmark

        except ValueError as e:
            logger.error(
                "Failed to add table bookmark",
                extra={
                    "table_name": table_name,
                    "page_number": page_number,
                    "parent_sheet": parent_sheet,
                    "level": level,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(f"Failed to add table bookmark: {e}") from e
        except Exception as e:
            logger.error(
                "Unexpected error adding table bookmark",
                extra={
                    "table_name": table_name,
                    "page_number": page_number,
                    "parent_sheet": parent_sheet,
                    "level": level,
                    "error": str(e),
                },
            )
            raise PDFGenerationException(
                "Unexpected error adding table bookmark"
            ) from e

    def generate_bookmark_outline(self) -> Dict[str, Any]:
        """Generate bookmark outline structure for PDF.

        Returns:
            Dictionary containing bookmark hierarchy

        Raises:
            PDFGenerationException: If outline generation fails
        """
        try:
            if not self.bookmarks:
                logger.debug("No bookmarks to generate outline")
                return {
                    "outline": [],
                    "metadata": {
                        "total_bookmarks": 0,
                        "max_level": 0,
                        "total_pages": self.page_counter,
                    },
                }

            # Sort bookmarks by page number, then by level, then by title
            sorted_bookmarks = sorted(
                self.bookmarks, key=lambda b: (b.page_number, b.level, b.title)
            )

            # Build hierarchical structure
            outline = []
            sheet_bookmarks = {}

            # First, group table bookmarks under their parent sheets
            for bookmark in sorted_bookmarks:
                if bookmark.level == 0:
                    # Sheet bookmark
                    sheet_entry: Dict[str, Any] = {
                        "title": bookmark.title,
                        "page": bookmark.page_number,
                        "level": bookmark.level,
                        "children": [],
                    }
                    sheet_bookmarks[bookmark.title] = sheet_entry
                else:
                    # Table bookmark
                    if bookmark.parent and bookmark.parent in sheet_bookmarks:
                        children_list = sheet_bookmarks[bookmark.parent]["children"]
                        if isinstance(children_list, list):
                            children_list.append(
                                {
                                    "title": bookmark.title,
                                    "page": bookmark.page_number,
                                    "level": bookmark.level,
                                    "children": [],  # Support for deeper nesting if needed
                                }
                            )
                    else:
                        # Orphan table bookmark - add as top-level
                        logger.warning(
                            "Orphan table bookmark found",
                            extra={
                                "title": bookmark.title,
                                "parent": bookmark.parent,
                                "page": bookmark.page_number,
                            },
                        )
                        outline.append(
                            {
                                "title": bookmark.title,
                                "page": bookmark.page_number,
                                "level": bookmark.level,
                                "children": [],
                            }
                        )

            # Add sheet bookmarks to outline in order
            for bookmark in sorted_bookmarks:
                if bookmark.level == 0 and bookmark.title in sheet_bookmarks:
                    outline.append(sheet_bookmarks[bookmark.title])

            # Calculate metadata
            max_level = (
                max(bookmark.level for bookmark in self.bookmarks)
                if self.bookmarks
                else 0
            )

            result = {
                "outline": outline,
                "metadata": {
                    "total_bookmarks": len(self.bookmarks),
                    "max_level": max_level,
                    "total_pages": self.page_counter,
                    "sheet_count": len([b for b in self.bookmarks if b.level == 0]),
                    "table_count": len([b for b in self.bookmarks if b.level > 0]),
                },
            }

            logger.info(
                "Bookmark outline generated",
                extra={
                    "total_bookmarks": len(self.bookmarks),
                    "outline_items": len(outline),
                    "max_level": max_level,
                    "total_pages": self.page_counter,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to generate bookmark outline",
                extra={"bookmark_count": len(self.bookmarks), "error": str(e)},
            )
            raise PDFGenerationException("Failed to generate bookmark outline") from e

    def get_bookmarks_by_sheet(self, sheet_name: str) -> List[BookmarkInfo]:
        """Get all bookmarks for a specific sheet.

        Args:
            sheet_name: Name of the sheet

        Returns:
            List of bookmarks belonging to the sheet
        """
        return [
            bookmark
            for bookmark in self.bookmarks
            if bookmark.parent == sheet_name or bookmark.title == sheet_name
        ]

    def get_bookmarks_by_level(self, level: int) -> List[BookmarkInfo]:
        """Get all bookmarks at a specific hierarchy level.

        Args:
            level: Hierarchy level (0 for sheets, 1+ for tables)

        Returns:
            List of bookmarks at the specified level
        """
        return [bookmark for bookmark in self.bookmarks if bookmark.level == level]

    def get_sheet_bookmarks(self) -> List[BookmarkInfo]:
        """Get all top-level sheet bookmarks.

        Returns:
            List of sheet bookmarks (level 0)
        """
        return self.get_bookmarks_by_level(0)

    def get_table_bookmarks(self) -> List[BookmarkInfo]:
        """Get all table bookmarks.

        Returns:
            List of table bookmarks (level 1+)
        """
        return [bookmark for bookmark in self.bookmarks if bookmark.level > 0]

    def clear_bookmarks(self) -> None:
        """Clear all bookmarks."""
        bookmark_count = len(self.bookmarks)
        self.bookmarks.clear()
        self.page_counter = 0

        logger.debug("Bookmarks cleared", extra={"cleared_count": bookmark_count})

    def validate_bookmark_structure(self) -> List[str]:
        """Validate bookmark structure and return list of issues.

        Returns:
            List of validation issue descriptions (empty if valid)
        """
        issues: List[str] = []

        if not self.bookmarks:
            return issues  # Empty structure is valid

        # Check for duplicate bookmarks at same level
        seen_bookmarks = set()
        for bookmark in self.bookmarks:
            key = (bookmark.title.lower(), bookmark.level)
            if key in seen_bookmarks:
                issues.append(
                    f"Duplicate bookmark found: '{bookmark.title}' at level {bookmark.level}"
                )
            seen_bookmarks.add(key)

        # Check for orphan table bookmarks
        sheet_names = {
            bookmark.title for bookmark in self.bookmarks if bookmark.level == 0
        }
        for bookmark in self.bookmarks:
            if (
                bookmark.level > 0
                and bookmark.parent
                and bookmark.parent not in sheet_names
            ):
                issues.append(
                    f"Orphan table bookmark: '{bookmark.title}' with non-existent parent '{bookmark.parent}'"
                )

        # Check page number consistency
        if self.bookmarks:
            min_page = min(bookmark.page_number for bookmark in self.bookmarks)
            max_page = max(bookmark.page_number for bookmark in self.bookmarks)

            if min_page < 1:
                issues.append(f"Invalid minimum page number: {min_page}")

            if max_page < min_page:
                issues.append(f"Invalid page range: {min_page} to {max_page}")

        # Check for level gaps
        levels = sorted(set(bookmark.level for bookmark in self.bookmarks))
        if levels and levels[0] != 0:
            issues.append(
                f"Bookmark levels should start at 0, but start at {levels[0]}"
            )

        for i in range(1, len(levels)):
            if levels[i] != levels[i - 1] + 1:
                issues.append(
                    f"Gap in bookmark levels: missing level {levels[i-1] + 1}"
                )

        if issues:
            logger.warning(
                "Bookmark structure validation issues found",
                extra={
                    "issue_count": len(issues),
                    "bookmark_count": len(self.bookmarks),
                    "issues": issues,
                },
            )
        else:
            logger.debug(
                "Bookmark structure validation passed",
                extra={"bookmark_count": len(self.bookmarks)},
            )

        return issues

    def get_statistics(self) -> Dict[str, Any]:
        """Get bookmark manager statistics.

        Returns:
            Dictionary containing statistics about bookmarks
        """
        if not self.bookmarks:
            return {
                "total_bookmarks": 0,
                "sheet_count": 0,
                "table_count": 0,
                "max_level": 0,
                "total_pages": 0,
                "average_tables_per_sheet": 0,
                "sheets_with_tables": 0,
            }

        sheet_bookmarks = self.get_sheet_bookmarks()
        table_bookmarks = self.get_table_bookmarks()

        # Calculate sheets with tables
        sheets_with_tables = len(
            set(bookmark.parent for bookmark in table_bookmarks if bookmark.parent)
        )

        return {
            "total_bookmarks": len(self.bookmarks),
            "sheet_count": len(sheet_bookmarks),
            "table_count": len(table_bookmarks),
            "max_level": max(bookmark.level for bookmark in self.bookmarks),
            "total_pages": self.page_counter,
            "average_tables_per_sheet": (
                len(table_bookmarks) / len(sheet_bookmarks) if sheet_bookmarks else 0
            ),
            "sheets_with_tables": sheets_with_tables,
        }
