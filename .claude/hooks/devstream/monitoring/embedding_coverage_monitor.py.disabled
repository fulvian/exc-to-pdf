#!/usr/bin/env python3
"""
Embedding Coverage Monitor - Detects missing embeddings in semantic_memory.

This script monitors the DevStream database to ensure every record in semantic_memory
has a corresponding embedding. Alerts when coverage drops below threshold to prevent
future backfill operations.

Usage:
    .devstream/bin/python .claude/hooks/devstream/monitoring/embedding_coverage_monitor.py --threshold 95
"""

import asyncio
import sys
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog

# Setup structured logging
logger = structlog.get_logger(__name__)


class EmbeddingCoverageMonitor:
    """Monitor for detecting missing embeddings in semantic_memory."""

    def __init__(self, coverage_threshold: float = 95.0):
        """
        Initialize embedding coverage monitor.

        Args:
            coverage_threshold: Minimum acceptable coverage percentage (default: 95.0)
        """
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.db_path = self.project_root / "data" / "devstream.db"
        self.log_file = self.project_root / ".claude" / "logs" / "devstream" / "embedding_coverage_monitor.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.coverage_threshold = coverage_threshold
        self.alert_sent = False  # Prevent duplicate alerts
        self.last_missing_ids: List[str] = []  # Track missing record IDs

    def _get_db_with_vec(self) -> sqlite3.Connection:
        """
        Get database connection with sqlite-vec extension loaded.

        Returns:
            SQLite connection with vec0 extension loaded
        """
        from pathlib import Path

        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)

        # Find vec0 extension (.dylib for macOS, .so for Linux)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        vec_path = project_root / ".devstream/lib/python3.11/site-packages/sqlite_vec/vec0.dylib"
        if not vec_path.exists():
            vec_path = project_root / ".devstream/lib/python3.11/site-packages/sqlite_vec/vec0.so"

        if vec_path.exists():
            conn.load_extension(str(vec_path.parent / "vec0"))
        else:
            raise Exception(f"sqlite-vec extension not found at {vec_path}")

        return conn

    def _get_coverage_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get embedding coverage statistics from database.

        Directly queries vec_semantic_memory (virtual table) with sqlite-vec loaded.
        This gives REAL-TIME coverage, not outdated history snapshots.

        Returns:
            Dictionary with coverage stats or None if query fails
        """
        try:
            conn = self._get_db_with_vec()
            cursor = conn.cursor()

            # Count total records in semantic_memory
            cursor.execute("SELECT COUNT(*) FROM semantic_memory")
            total_records = cursor.fetchone()[0]

            # Count records WITH embeddings in vec_semantic_memory (virtual table)
            cursor.execute("SELECT COUNT(*) FROM vec_semantic_memory")
            records_with_embeddings = cursor.fetchone()[0]

            records_without_embeddings = total_records - records_with_embeddings
            coverage_percentage = (records_with_embeddings / total_records * 100) if total_records > 0 else 0.0

            # Get IDs of records WITH embeddings (to identify missing ones)
            cursor.execute("""
                SELECT memory_id FROM vec_semantic_memory
                ORDER BY rowid DESC
                LIMIT 5
            """)
            with_emb_ids = {row[0] for row in cursor.fetchall()}

            # Get sample records WITHOUT embeddings (for actionable alerts)
            cursor.execute("""
                SELECT id, content_type, SUBSTR(content, 1, 50) as preview, created_at
                FROM semantic_memory
                WHERE id NOT IN (SELECT memory_id FROM vec_semantic_memory)
                ORDER BY created_at DESC
                LIMIT 10
            """)
            missing_records = cursor.fetchall()

            conn.close()

            return {
                "total_records": total_records,
                "records_with_embeddings": records_with_embeddings,
                "records_without_embeddings": records_without_embeddings,
                "coverage_percentage": round(coverage_percentage, 2),
                "missing_records": [
                    {
                        "id": row[0],
                        "content_type": row[1],
                        "preview": row[2],
                        "created_at": row[3]
                    }
                    for row in missing_records
                ],
                "data_source": "real_time_vec_query"
            }

        except sqlite3.Error as e:
            logger.error("Database query error", error=str(e))
            return None

        except Exception as e:
            logger.error("Failed to get coverage stats", error=str(e))
            return None

    def _get_recent_inserts_without_embeddings(self, minutes: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Get records inserted in last N minutes without embeddings.

        Args:
            minutes: Time window in minutes (default: 5)

        Returns:
            List of records or None if query fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Calculate cutoff time
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=minutes)
            cutoff_iso = cutoff.isoformat() + "Z"

            # Get recent inserts without embeddings
            cursor.execute("""
                SELECT id, content_type, SUBSTR(content, 1, 80) as preview, created_at
                FROM semantic_memory
                WHERE (embedding IS NULL OR embedding = '' OR LENGTH(embedding) <= 10)
                  AND created_at >= ?
                ORDER BY created_at DESC
            """, (cutoff_iso,))

            recent_missing = cursor.fetchall()
            conn.close()

            return [
                {
                    "id": row[0],
                    "content_type": row[1],
                    "preview": row[2],
                    "created_at": row[3]
                }
                for row in recent_missing
            ]

        except Exception as e:
            logger.error("Failed to get recent inserts", error=str(e))
            return None

    async def check_coverage(self) -> Dict[str, Any]:
        """
        Check embedding coverage and return alert status.

        Returns:
            Dictionary with coverage check results
        """
        timestamp = datetime.utcnow()

        # Get coverage stats
        stats = self._get_coverage_stats()

        if stats is None:
            result = {
                "timestamp": timestamp.isoformat() + "Z",
                "status": "error",
                "message": "Unable to query database",
                "alert": False
            }

            logger.error("Coverage check failed: database query error")
            return result

        # Check if coverage below threshold
        is_below_threshold = stats["coverage_percentage"] < self.coverage_threshold

        # Get recent inserts without embeddings (last 5 minutes)
        recent_missing = self._get_recent_inserts_without_embeddings(minutes=5)

        result = {
            "timestamp": timestamp.isoformat() + "Z",
            "status": "below_threshold" if is_below_threshold else "ok",
            "total_records": stats["total_records"],
            "records_with_embeddings": stats["records_with_embeddings"],
            "records_without_embeddings": stats["records_without_embeddings"],
            "coverage_percentage": stats["coverage_percentage"],
            "threshold": self.coverage_threshold,
            "recent_missing_count": len(recent_missing) if recent_missing else 0,
            "alert": is_below_threshold and not self.alert_sent
        }

        # Add sample missing records (max 5)
        if stats["missing_records"]:
            result["missing_samples"] = stats["missing_records"][:5]

        # Add recent missing records if any
        if recent_missing:
            result["recent_missing"] = recent_missing[:5]

        # Log to file
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Log to structured logger
        if is_below_threshold:
            if not self.alert_sent:
                logger.error(
                    "Embedding coverage ALERT",
                    coverage_percentage=stats["coverage_percentage"],
                    threshold=self.coverage_threshold,
                    missing_count=stats["records_without_embeddings"]
                )
                self.alert_sent = True
            else:
                logger.warning(
                    "Coverage still below threshold",
                    coverage_percentage=stats["coverage_percentage"],
                    missing_count=stats["records_without_embeddings"]
                )
        else:
            logger.info(
                "Embedding coverage check: OK",
                coverage_percentage=stats["coverage_percentage"],
                total_records=stats["total_records"]
            )
            self.alert_sent = False  # Reset alert flag when recovered

        return result

    async def monitor_loop(self, interval_seconds: int = 60):
        """
        Continuous coverage monitoring loop.

        Args:
            interval_seconds: Check interval in seconds (default: 60)
        """
        logger.info(
            "Starting embedding coverage monitor",
            interval=interval_seconds,
            threshold=self.coverage_threshold
        )

        while True:
            try:
                result = await self.check_coverage()

                # Alert on low coverage
                if result.get("alert"):
                    print(
                        f"⚠️  EMBEDDING COVERAGE ALERT: {result['coverage_percentage']}% "
                        f"(threshold: {self.coverage_threshold}%)\n"
                        f"   Missing embeddings: {result['records_without_embeddings']}/{result['total_records']} records\n"
                        f"   Recent inserts without embeddings: {result['recent_missing_count']} in last 5 min\n"
                        f"   Action required: Check Ollama service or run embedding backfill",
                        file=sys.stderr
                    )

                    # Show sample missing records if recent inserts detected
                    if result.get("recent_missing"):
                        print("\n   Recent missing embeddings:", file=sys.stderr)
                        for rec in result["recent_missing"][:3]:
                            print(
                                f"     • {rec['id'][:8]}... ({rec['content_type']}) - {rec['created_at']}",
                                file=sys.stderr
                            )

            except Exception as e:
                logger.error("Coverage check loop error", error=str(e))

            await asyncio.sleep(interval_seconds)


async def main():
    """Main entry point for embedding coverage monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Coverage Monitor")
    parser.add_argument(
        "--threshold",
        type=float,
        default=95.0,
        help="Minimum acceptable coverage percentage (default: 95.0)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single check and exit"
    )

    args = parser.parse_args()

    monitor = EmbeddingCoverageMonitor(coverage_threshold=args.threshold)

    # Single check mode
    if args.once:
        result = await monitor.check_coverage()
        import json
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "ok" else 1)

    # Continuous monitoring mode
    await monitor.monitor_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
