#!/usr/bin/env python3
"""
Context Retention Policy Script

Removes context records older than specified days from semantic_memory table.
Implements Opzione B: 7-day retention for content_type='context'.

Features:
- Configurable retention period (default: 7 days)
- Dry-run mode for safety
- VACUUM after cleanup to reclaim disk space
- Detailed logging and metrics
- Transaction safety

Usage:
    .devstream/bin/python scripts/cleanup_old_context.py [--days N] [--dry-run]

Environment:
    DEVSTREAM_CONTEXT_RETENTION_DAYS: Retention period in days (default: 7)
"""

import sqlite3
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Logging setup
log_dir = Path.home() / ".claude" / "logs" / "devstream"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "context_cleanup.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_database_size(db_path: str) -> int:
    """Get database file size in bytes."""
    return Path(db_path).stat().st_size


def analyze_context_records(conn: sqlite3.Connection, retention_days: int) -> Dict[str, Any]:
    """Analyze context records to be deleted."""
    cursor = conn.cursor()

    # Cutoff date
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_iso = cutoff_date.isoformat()

    # Total context records
    cursor.execute(
        "SELECT COUNT(*) FROM semantic_memory WHERE content_type = 'context'"
    )
    total_context = cursor.fetchone()[0]

    # Old context records (to be deleted)
    cursor.execute(
        "SELECT COUNT(*) FROM semantic_memory WHERE content_type = 'context' AND created_at < ?",
        (cutoff_iso,)
    )
    old_context = cursor.fetchone()[0]

    # Total records
    cursor.execute("SELECT COUNT(*) FROM semantic_memory")
    total_records = cursor.fetchone()[0]

    # Breakdown by date
    cursor.execute("""
        SELECT date(created_at) as day, COUNT(*) as count
        FROM semantic_memory
        WHERE content_type = 'context' AND created_at < ?
        GROUP BY day
        ORDER BY day DESC
    """, (cutoff_iso,))
    breakdown = cursor.fetchall()

    return {
        "total_records": total_records,
        "total_context": total_context,
        "old_context": old_context,
        "retention_context": total_context - old_context,
        "cutoff_date": cutoff_iso,
        "breakdown": breakdown
    }


def cleanup_old_context(
    db_path: str = "data/devstream.db",
    retention_days: int = 7,
    dry_run: bool = False,
    vacuum: bool = True
) -> Dict[str, Any]:
    """
    Execute context retention policy cleanup.

    Args:
        db_path: Path to DevStream database
        retention_days: Number of days to retain context records
        dry_run: If True, analyze only without deleting
        vacuum: If True, run VACUUM after cleanup to reclaim space

    Returns:
        Dictionary with cleanup metrics
    """
    logger.info("=" * 70)
    logger.info("🧹 CONTEXT RETENTION POLICY - START")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Retention: {retention_days} days")
    logger.info(f"Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}")
    logger.info(f"Vacuum: {'enabled' if vacuum else 'disabled'}")
    logger.info("=" * 70)

    # Initial size
    initial_size = get_database_size(db_path)
    logger.info(f"\n📊 Initial DB size: {initial_size / (1024**2):.1f} MB")

    # Connect
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Analyze
        logger.info("\n🔍 Analyzing records...")
        analysis = analyze_context_records(conn, retention_days)

        logger.info(f"\n📊 Analysis Results:")
        logger.info(f"  Total records: {analysis['total_records']:,}")
        logger.info(f"  Total context: {analysis['total_context']:,}")
        logger.info(f"  Old context (to delete): {analysis['old_context']:,}")
        logger.info(f"  Retention context (kept): {analysis['retention_context']:,}")
        logger.info(f"  Cutoff date: {analysis['cutoff_date']}")

        if analysis['breakdown']:
            logger.info(f"\n📅 Breakdown by date (to be deleted):")
            for day, count in analysis['breakdown'][:10]:
                logger.info(f"    {day}: {count:,} records")

        if analysis['old_context'] == 0:
            logger.info("\n✅ No old context records to delete!")
            conn.close()
            return {
                "deleted": 0,
                "initial_size_mb": initial_size / (1024**2),
                "final_size_mb": initial_size / (1024**2),
                "space_saved_mb": 0,
                "dry_run": dry_run
            }

        if dry_run:
            logger.info("\n⚠️ DRY-RUN MODE: No changes will be made")
            conn.close()
            return {
                "deleted": 0,
                "would_delete": analysis['old_context'],
                "initial_size_mb": initial_size / (1024**2),
                "dry_run": True
            }

        # Execute cleanup
        logger.info(f"\n🗑️ Deleting {analysis['old_context']:,} old context records...")

        cursor = conn.cursor()
        cutoff_iso = analysis['cutoff_date']

        cursor.execute(
            "DELETE FROM semantic_memory WHERE content_type = 'context' AND created_at < ?",
            (cutoff_iso,)
        )
        deleted_count = cursor.rowcount
        conn.commit()

        logger.info(f"✅ Deleted {deleted_count:,} records")

        # Analyze tables (optimize query planner)
        logger.info("\n📈 Running ANALYZE...")
        cursor.execute("ANALYZE")
        conn.commit()
        logger.info("✅ ANALYZE complete")

        # Vacuum (reclaim disk space)
        if vacuum:
            logger.info("\n🗜️ Running VACUUM (may take a few minutes)...")
            # VACUUM cannot be run in transaction, so we close and reopen
            conn.close()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            conn.commit()
            logger.info("✅ VACUUM complete")

        # WAL Checkpoint (integrate WAL and truncate)
        logger.info("\n📦 Running WAL checkpoint...")
        if not conn or conn.in_transaction:
            conn.close()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        logger.info("✅ WAL checkpoint complete")

        # Final size
        final_size = get_database_size(db_path)
        space_saved = initial_size - final_size

        logger.info("\n" + "=" * 70)
        logger.info("📋 CLEANUP COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Deleted records: {deleted_count:,}")
        logger.info(f"Initial DB size: {initial_size / (1024**2):.1f} MB")
        logger.info(f"Final DB size: {final_size / (1024**2):.1f} MB")
        logger.info(f"Space saved: {space_saved / (1024**2):.1f} MB ({space_saved / initial_size * 100:.1f}%)")
        logger.info("=" * 70)

        return {
            "deleted": deleted_count,
            "initial_size_mb": initial_size / (1024**2),
            "final_size_mb": final_size / (1024**2),
            "space_saved_mb": space_saved / (1024**2),
            "space_saved_percent": space_saved / initial_size * 100,
            "dry_run": False
        }

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DevStream context retention policy cleanup"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Retention period in days (default: 7)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, don't delete"
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip VACUUM operation"
    )
    parser.add_argument(
        "--db-path",
        default="data/devstream.db",
        help="Path to database (default: data/devstream.db)"
    )

    args = parser.parse_args()

    try:
        result = cleanup_old_context(
            db_path=args.db_path,
            retention_days=args.days,
            dry_run=args.dry_run,
            vacuum=not args.no_vacuum
        )

        if result.get("dry_run"):
            logger.info("\n💡 Run without --dry-run to execute cleanup")
            sys.exit(0)

        if result.get("deleted", 0) > 0:
            logger.info("\n✅ Cleanup successful!")
            sys.exit(0)
        else:
            logger.info("\n✅ No cleanup needed")
            sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Cleanup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
