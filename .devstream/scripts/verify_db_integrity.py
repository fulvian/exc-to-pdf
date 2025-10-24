#!/usr/bin/env python3
"""
DevStream Database Integrity Verification Script
Checks alignment between semantic_memory and vec_semantic_memory
"""

import sqlite3
import sqlite_vec
import sys
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "devstream.db"

def main():
    """Verify database integrity and report statistics."""

    print("🔍 DevStream Database Integrity Verification\n")
    print(f"Database: {DB_PATH}\n")

    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load sqlite-vec extension
    try:
        sqlite_vec.load(conn)
        print("✅ sqlite-vec extension loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load sqlite-vec: {e}\n")
        sys.exit(1)

    cursor = conn.cursor()

    # 1. Count semantic_memory records
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_json_embedding,
            COUNT(CASE WHEN embedding IS NULL THEN 1 END) as null_embedding,
            COUNT(CASE WHEN embedding_model IS NOT NULL THEN 1 END) as has_model_info
        FROM semantic_memory
    """)

    sem_stats = cursor.fetchone()

    print("=" * 70)
    print("📊 SEMANTIC_MEMORY TABLE")
    print("=" * 70)
    print(f"Total records: {sem_stats['total']:,}")
    print(f"Records with JSON embedding: {sem_stats['with_json_embedding']:,}")
    print(f"Records with NULL embedding: {sem_stats['null_embedding']:,}")
    print(f"Records with model info: {sem_stats['has_model_info']:,}")
    print()

    # 2. Count vec_semantic_memory records (virtual table)
    try:
        cursor.execute("SELECT COUNT(*) as total FROM vec_semantic_memory")
        vec_stats = cursor.fetchone()

        print("=" * 70)
        print("🎯 VEC_SEMANTIC_MEMORY TABLE (Vector Store)")
        print("=" * 70)
        print(f"Total vector records: {vec_stats['total']:,}")
        print()
    except Exception as e:
        print(f"❌ Error querying vec_semantic_memory: {e}\n")

    # 3. Find misalignment (records in semantic_memory but NOT in vec_semantic_memory)
    try:
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM semantic_memory sm
            LEFT JOIN vec_semantic_memory vsm ON sm.id = vsm.memory_id
            WHERE vsm.memory_id IS NULL
            AND sm.embedding_model IS NOT NULL
        """)

        missing_stats = cursor.fetchone()

        print("=" * 70)
        print("⚠️  ALIGNMENT CHECK")
        print("=" * 70)
        print(f"Records with model info but NO vector: {missing_stats['missing']:,}")

        if missing_stats['missing'] > 0:
            print("\n❌ GAP DETECTED: Some records need backfill!")

            # Show sample of missing records
            cursor.execute("""
                SELECT
                    sm.id,
                    sm.content_type,
                    sm.embedding_model,
                    sm.embedding_dimension,
                    substr(sm.content, 1, 100) as content_preview,
                    sm.created_at
                FROM semantic_memory sm
                LEFT JOIN vec_semantic_memory vsm ON sm.id = vsm.memory_id
                WHERE vsm.memory_id IS NULL
                AND sm.embedding_model IS NOT NULL
                ORDER BY sm.created_at DESC
                LIMIT 5
            """)

            print("\nSample of records needing backfill (most recent 5):")
            print("-" * 70)

            for row in cursor.fetchall():
                print(f"\nID: {row['id']}")
                print(f"Type: {row['content_type']}")
                print(f"Model: {row['embedding_model']} ({row['embedding_dimension']}D)")
                print(f"Created: {row['created_at']}")
                print(f"Content: {row['content_preview']}...")
        else:
            print("✅ All records are properly aligned!")

        print()

    except Exception as e:
        print(f"❌ Error checking alignment: {e}\n")

    # 4. Check for orphaned vector records
    try:
        cursor.execute("""
            SELECT COUNT(*) as orphaned
            FROM vec_semantic_memory vsm
            LEFT JOIN semantic_memory sm ON vsm.memory_id = sm.id
            WHERE sm.id IS NULL
        """)

        orphaned_stats = cursor.fetchone()

        print("=" * 70)
        print("🔍 ORPHANED VECTORS CHECK")
        print("=" * 70)
        print(f"Orphaned vectors (no parent record): {orphaned_stats['orphaned']:,}")

        if orphaned_stats['orphaned'] > 0:
            print("⚠️  WARNING: Some vectors have no parent record (cleanup recommended)")
        else:
            print("✅ No orphaned vectors found")

        print()

    except Exception as e:
        print(f"❌ Error checking orphaned vectors: {e}\n")

    # 5. Content type distribution
    cursor.execute("""
        SELECT
            content_type,
            COUNT(*) as count
        FROM semantic_memory
        GROUP BY content_type
        ORDER BY count DESC
    """)

    print("=" * 70)
    print("📈 CONTENT TYPE DISTRIBUTION")
    print("=" * 70)

    for row in cursor.fetchall():
        print(f"{row['content_type']:15s}: {row['count']:>8,}")

    print()

    # 6. Recently created records (last 24 hours)
    cursor.execute("""
        SELECT
            COUNT(*) as recent,
            COUNT(CASE WHEN embedding_model IS NOT NULL THEN 1 END) as recent_with_model
        FROM semantic_memory
        WHERE created_at >= datetime('now', '-1 day')
    """)

    recent_stats = cursor.fetchone()

    print("=" * 70)
    print("🕐 RECENT ACTIVITY (Last 24 hours)")
    print("=" * 70)
    print(f"New records: {recent_stats['recent']:,}")
    print(f"New records with embeddings: {recent_stats['recent_with_model']:,}")
    print()

    conn.close()

    print("=" * 70)
    print("✅ Integrity verification complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
