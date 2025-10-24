#!/usr/bin/env python3
"""
DevStream SQLite-Vec Helper - Context7 Compliant

Utility for loading sqlite-vec extension properly in DevStream hooks.
Implements Context7 best practices from sqlite-vec official documentation.

Context7 Pattern:
    import sqlite3
    import sqlite_vec

    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

Source: https://github.com/asg017/sqlite-vec (Python Integration)
"""

import sqlite3
from typing import Optional
from pathlib import Path
import sys

# Import connection manager
sys.path.append(str(Path(__file__).parent))
from connection_manager import get_connection_manager


def get_db_connection_with_vec(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Get SQLite connection with sqlite-vec extension loaded via ConnectionManager.

    Uses centralized ConnectionManager for automatic WAL mode enforcement,
    then loads sqlite-vec extension for vector operations.

    Context7 Pattern: Enable extensions, load sqlite-vec, disable extensions.
    WAL mode and safety pragmas are automatically set by ConnectionManager.

    Args:
        db_path: Path to database file (default: data/devstream.db)

    Returns:
        sqlite3.Connection with sqlite-vec extension loaded and WAL mode enabled

    Raises:
        ImportError: If sqlite-vec not installed
        sqlite3.Error: If connection or extension loading fails

    Example:
        >>> conn = get_db_connection_with_vec()
        >>> cursor = conn.cursor()
        >>> version = cursor.execute("SELECT vec_version()").fetchone()[0]
        >>> print(f"vec_version={version}")
    """
    try:
        # Import sqlite_vec (Context7 pattern)
        import sqlite_vec
    except ImportError as e:
        raise ImportError(
            "sqlite-vec not installed. Install with: pip install sqlite-vec"
        ) from e

    # Get connection via ConnectionManager (automatic WAL mode + safety pragmas)
    manager = get_connection_manager(db_path)
    conn = manager._get_thread_connection()

    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    return conn


def verify_vec_extension(conn: sqlite3.Connection) -> bool:
    """
    Verify that sqlite-vec extension is loaded and working.

    Args:
        conn: SQLite connection to test

    Returns:
        True if vec0 extension is available, False otherwise
    """
    try:
        cursor = conn.cursor()
        version = cursor.execute("SELECT vec_version()").fetchone()[0]
        return version is not None
    except Exception:
        return False


# Convenience function for quick one-off connections
def get_devstream_db() -> sqlite3.Connection:
    """
    Get DevStream database connection with vec0 extension loaded.

    Convenience wrapper for get_db_connection_with_vec() with default path.

    Returns:
        sqlite3.Connection to data/devstream.db with sqlite-vec loaded
    """
    return get_db_connection_with_vec()


if __name__ == "__main__":
    # Test script
    print("DevStream SQLite-Vec Helper Test")
    print("=" * 50)

    try:
        # Test connection
        print("\n1. Testing connection with vec0 extension...")
        conn = get_devstream_db()
        print("   ✅ Connection established")

        # Verify extension
        print("\n2. Verifying vec0 extension...")
        if verify_vec_extension(conn):
            cursor = conn.cursor()
            version = cursor.execute("SELECT vec_version()").fetchone()[0]
            print(f"   ✅ vec_version={version}")
        else:
            print("   ❌ vec0 extension not available")

        # Test vec_semantic_memory access
        print("\n3. Testing vec_semantic_memory access...")
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM vec_semantic_memory").fetchone()[0]
        print(f"   ✅ vec_semantic_memory records: {count}")

        conn.close()
        print("\n" + "=" * 50)
        print("Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()