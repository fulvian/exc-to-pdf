"""
Unit tests for PDF bookmark manager.

Tests the BookmarkManager class including bookmark creation, hierarchy
management, outline generation, and validation functionality.
"""

import pytest

from exc_to_pdf.bookmark_manager import BookmarkManager, BookmarkInfo
from exc_to_pdf.exceptions import PDFGenerationException


class TestBookmarkManager:
    """Test cases for BookmarkManager class."""

    def test_initialization(self) -> None:
        """Test BookmarkManager initialization."""
        manager = BookmarkManager()

        assert manager.bookmarks == []
        assert manager.page_counter == 0

    def test_add_sheet_bookmark(self) -> None:
        """Test adding a sheet bookmark."""
        manager = BookmarkManager()

        bookmark = manager.add_sheet_bookmark("Sheet1", 1)

        assert isinstance(bookmark, BookmarkInfo)
        assert bookmark.title == "Sheet1"
        assert bookmark.page_number == 1
        assert bookmark.level == 0
        assert bookmark.parent is None
        assert len(manager.bookmarks) == 1
        assert manager.page_counter == 1

    def test_add_sheet_bookmark_trims_whitespace(self) -> None:
        """Test that sheet bookmark names are trimmed."""
        manager = BookmarkManager()

        bookmark = manager.add_sheet_bookmark("  Sheet1  ", 1)

        assert bookmark.title == "Sheet1"

    def test_add_sheet_bookmark_empty_name_raises_exception(self) -> None:
        """Test that empty sheet name raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add sheet bookmark"
        ):
            manager.add_sheet_bookmark("", 1)

        with pytest.raises(
            PDFGenerationException, match="Failed to add sheet bookmark"
        ):
            manager.add_sheet_bookmark("   ", 1)

    def test_add_sheet_bookmark_invalid_page_number_raises_exception(self) -> None:
        """Test that invalid page number raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add sheet bookmark"
        ):
            manager.add_sheet_bookmark("Sheet1", 0)

        with pytest.raises(
            PDFGenerationException, match="Failed to add sheet bookmark"
        ):
            manager.add_sheet_bookmark("Sheet1", -5)

    def test_add_multiple_sheet_bookmarks(self) -> None:
        """Test adding multiple sheet bookmarks."""
        manager = BookmarkManager()

        bookmark1 = manager.add_sheet_bookmark("Sheet1", 1)
        bookmark2 = manager.add_sheet_bookmark("Sheet2", 5)

        assert len(manager.bookmarks) == 2
        assert manager.bookmarks[0] == bookmark1
        assert manager.bookmarks[1] == bookmark2
        assert manager.page_counter == 5

    def test_add_table_bookmark(self) -> None:
        """Test adding a table bookmark."""
        manager = BookmarkManager()

        # First add a sheet bookmark
        manager.add_sheet_bookmark("Sheet1", 1)

        # Then add a table bookmark
        bookmark = manager.add_table_bookmark("Table1", 2, "Sheet1", 1)

        assert isinstance(bookmark, BookmarkInfo)
        assert bookmark.title == "Table1"
        assert bookmark.page_number == 2
        assert bookmark.level == 1
        assert bookmark.parent == "Sheet1"
        assert len(manager.bookmarks) == 2

    def test_add_table_bookmark_trims_whitespace(self) -> None:
        """Test that table bookmark names are trimmed."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        bookmark = manager.add_table_bookmark("  Table1  ", 2, "  Sheet1  ", 1)

        assert bookmark.title == "Table1"
        assert bookmark.parent == "Sheet1"

    def test_add_table_bookmark_empty_name_raises_exception(self) -> None:
        """Test that empty table name raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add table bookmark"
        ):
            manager.add_table_bookmark("", 1, "Sheet1", 1)

    def test_add_table_bookmark_empty_parent_raises_exception(self) -> None:
        """Test that empty parent name raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add table bookmark"
        ):
            manager.add_table_bookmark("Table1", 1, "", 1)

    def test_add_table_bookmark_invalid_page_number_raises_exception(self) -> None:
        """Test that invalid page number raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add table bookmark"
        ):
            manager.add_table_bookmark("Table1", 0, "Sheet1", 1)

    def test_add_table_bookmark_invalid_level_raises_exception(self) -> None:
        """Test that invalid level raises exception."""
        manager = BookmarkManager()

        with pytest.raises(
            PDFGenerationException, match="Failed to add table bookmark"
        ):
            manager.add_table_bookmark("Table1", 1, "Sheet1", 0)

        with pytest.raises(
            PDFGenerationException, match="Failed to add table bookmark"
        ):
            manager.add_table_bookmark("Table1", 1, "Sheet1", -1)

    def test_add_table_bookmark_without_parent_sheet(self) -> None:
        """Test adding table bookmark without existing parent sheet."""
        manager = BookmarkManager()

        # Should not raise exception but log warning
        bookmark = manager.add_table_bookmark("Table1", 2, "NonExistentSheet", 1)

        assert bookmark.title == "Table1"
        assert bookmark.parent == "NonExistentSheet"
        assert len(manager.bookmarks) == 1

    def test_generate_bookmark_outline_empty(self) -> None:
        """Test generating outline with no bookmarks."""
        manager = BookmarkManager()

        outline = manager.generate_bookmark_outline()

        assert outline["outline"] == []
        assert outline["metadata"]["total_bookmarks"] == 0
        assert outline["metadata"]["max_level"] == 0
        assert outline["metadata"]["total_pages"] == 0

    def test_generate_bookmark_outline_sheets_only(self) -> None:
        """Test generating outline with only sheet bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)

        outline = manager.generate_bookmark_outline()

        assert len(outline["outline"]) == 2
        assert outline["outline"][0]["title"] == "Sheet1"
        assert outline["outline"][0]["page"] == 1
        assert outline["outline"][0]["level"] == 0
        assert outline["outline"][0]["children"] == []

        assert outline["metadata"]["total_bookmarks"] == 2
        assert outline["metadata"]["max_level"] == 0
        assert outline["metadata"]["sheet_count"] == 2
        assert outline["metadata"]["table_count"] == 0

    def test_generate_bookmark_outline_with_tables(self) -> None:
        """Test generating outline with sheets and tables."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)
        manager.add_table_bookmark("Table2", 3, "Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_table_bookmark("Table3", 6, "Sheet2", 1)

        outline = manager.generate_bookmark_outline()

        assert len(outline["outline"]) == 2  # 2 sheets

        # Check first sheet
        sheet1 = outline["outline"][0]
        assert sheet1["title"] == "Sheet1"
        assert sheet1["page"] == 1
        assert len(sheet1["children"]) == 2  # 2 tables

        # Check tables in first sheet
        tables1 = sheet1["children"]
        assert tables1[0]["title"] == "Table1"
        assert tables1[0]["page"] == 2
        assert tables1[1]["title"] == "Table2"
        assert tables1[1]["page"] == 3

        # Check metadata
        assert outline["metadata"]["total_bookmarks"] == 5
        assert outline["metadata"]["max_level"] == 1
        assert outline["metadata"]["sheet_count"] == 2
        assert outline["metadata"]["table_count"] == 3

    def test_generate_bookmark_outline_orphan_tables(self) -> None:
        """Test generating outline with orphan table bookmarks."""
        manager = BookmarkManager()

        # Add table bookmark without parent sheet
        manager.add_table_bookmark("OrphanTable", 1, "NonExistentSheet", 1)

        outline = manager.generate_bookmark_outline()

        # Orphan should appear as top-level item
        assert len(outline["outline"]) == 1
        assert outline["outline"][0]["title"] == "OrphanTable"
        assert outline["outline"][0]["level"] == 1

    def test_get_bookmarks_by_sheet(self) -> None:
        """Test getting bookmarks by sheet name."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)
        manager.add_table_bookmark("Table2", 3, "Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_table_bookmark("Table3", 6, "Sheet2", 1)

        # Get bookmarks for Sheet1
        sheet1_bookmarks = manager.get_bookmarks_by_sheet("Sheet1")
        assert len(sheet1_bookmarks) == 3  # 1 sheet + 2 tables
        assert all(
            b.title == "Sheet1" or b.parent == "Sheet1" for b in sheet1_bookmarks
        )

        # Get bookmarks for Sheet2
        sheet2_bookmarks = manager.get_bookmarks_by_sheet("Sheet2")
        assert len(sheet2_bookmarks) == 2  # 1 sheet + 1 table

    def test_get_bookmarks_by_level(self) -> None:
        """Test getting bookmarks by hierarchy level."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)
        manager.add_table_bookmark("Table2", 3, "Sheet1", 2)  # Level 2

        level0 = manager.get_bookmarks_by_level(0)
        level1 = manager.get_bookmarks_by_level(1)
        level2 = manager.get_bookmarks_by_level(2)

        assert len(level0) == 2  # 2 sheets
        assert len(level1) == 1  # 1 table at level 1
        assert len(level2) == 1  # 1 table at level 2

    def test_get_sheet_bookmarks(self) -> None:
        """Test getting all sheet bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)

        sheet_bookmarks = manager.get_sheet_bookmarks()

        assert len(sheet_bookmarks) == 2
        assert all(bookmark.level == 0 for bookmark in sheet_bookmarks)
        assert sheet_bookmarks[0].title == "Sheet1"
        assert sheet_bookmarks[1].title == "Sheet2"

    def test_get_table_bookmarks(self) -> None:
        """Test getting all table bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)
        manager.add_table_bookmark("Table2", 3, "Sheet1", 2)

        table_bookmarks = manager.get_table_bookmarks()

        assert len(table_bookmarks) == 2
        assert all(bookmark.level > 0 for bookmark in table_bookmarks)

    def test_clear_bookmarks(self) -> None:
        """Test clearing all bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)

        assert len(manager.bookmarks) == 2
        assert manager.page_counter > 0

        manager.clear_bookmarks()

        assert len(manager.bookmarks) == 0
        assert manager.page_counter == 0

    def test_validate_bookmark_structure_empty(self) -> None:
        """Test validating empty bookmark structure."""
        manager = BookmarkManager()

        issues = manager.validate_bookmark_structure()

        assert issues == []

    def test_validate_bookmark_structure_valid(self) -> None:
        """Test validating valid bookmark structure."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)

        issues = manager.validate_bookmark_structure()

        assert issues == []

    def test_validate_bookmark_structure_duplicates(self) -> None:
        """Test validating structure with duplicate bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet1", 5)  # Duplicate

        issues = manager.validate_bookmark_structure()

        assert len(issues) == 1
        assert "Duplicate bookmark found" in issues[0]

    def test_validate_bookmark_structure_orphans(self) -> None:
        """Test validating structure with orphan table bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("OrphanTable", 2, "NonExistentSheet", 1)

        issues = manager.validate_bookmark_structure()

        assert len(issues) == 1
        assert "Orphan table bookmark" in issues[0]

    def test_validate_bookmark_structure_level_gaps(self) -> None:
        """Test validating structure with level gaps."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_table_bookmark("Table1", 2, "Sheet1", 2)  # Level 2 without level 1

        issues = manager.validate_bookmark_structure()

        assert len(issues) == 1
        assert "Gap in bookmark levels" in issues[0]

    def test_get_statistics_empty(self) -> None:
        """Test getting statistics for empty manager."""
        manager = BookmarkManager()

        stats = manager.get_statistics()

        assert stats["total_bookmarks"] == 0
        assert stats["sheet_count"] == 0
        assert stats["table_count"] == 0
        assert stats["max_level"] == 0
        assert stats["total_pages"] == 0
        assert stats["average_tables_per_sheet"] == 0
        assert stats["sheets_with_tables"] == 0

    def test_get_statistics_with_data(self) -> None:
        """Test getting statistics with bookmarks."""
        manager = BookmarkManager()

        manager.add_sheet_bookmark("Sheet1", 1)
        manager.add_sheet_bookmark("Sheet2", 5)
        manager.add_sheet_bookmark("Sheet3", 10)  # Sheet without tables
        manager.add_table_bookmark("Table1", 2, "Sheet1", 1)
        manager.add_table_bookmark("Table2", 3, "Sheet1", 2)
        manager.add_table_bookmark("Table3", 6, "Sheet2", 1)

        stats = manager.get_statistics()

        assert stats["total_bookmarks"] == 6
        assert stats["sheet_count"] == 3
        assert stats["table_count"] == 3
        assert stats["max_level"] == 2
        assert stats["total_pages"] == 10
        assert stats["average_tables_per_sheet"] == 1.0  # 3 tables / 3 sheets
        assert stats["sheets_with_tables"] == 2  # Sheet1 and Sheet2 have tables

    def test_page_counter_updates(self) -> None:
        """Test that page counter updates correctly."""
        manager = BookmarkManager()

        assert manager.page_counter == 0

        manager.add_sheet_bookmark("Sheet1", 5)
        assert manager.page_counter == 5

        manager.add_table_bookmark("Table1", 3, "Sheet1", 1)
        assert manager.page_counter == 5  # Should keep max

        manager.add_sheet_bookmark("Sheet2", 10)
        assert manager.page_counter == 10

    def test_bookmark_info_dataclass(self) -> None:
        """Test BookmarkInfo dataclass."""
        bookmark = BookmarkInfo(
            title="Test Bookmark", page_number=5, level=1, parent="Parent Sheet"
        )

        assert bookmark.title == "Test Bookmark"
        assert bookmark.page_number == 5
        assert bookmark.level == 1
        assert bookmark.parent == "Parent Sheet"

        # Test with default parent
        bookmark_no_parent = BookmarkInfo(title="Test", page_number=1, level=0)
        assert bookmark_no_parent.parent is None
