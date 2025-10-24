#!/usr/bin/env .devstream/bin/python
"""
DevStream Embedding Coverage Monitor

Tracks embedding coverage metrics for semantic search system.
Provides real-time monitoring and alerting for embedding completeness.

FASE 4.5: Coverage monitoring with health checks and alerts.
"""

import sqlite3
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import structlog

# Initialize logging
logger = structlog.get_logger()


class EmbeddingCoverageMonitor:
    """
    Monitor embedding coverage in DevStream semantic memory.

    Tracks:
    - Total records vs records with embeddings
    - Coverage percentage over time
    - Missing embeddings by content type
    - Embedding generation failures
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.alert_threshold = 90.0  # Alert if coverage < 90%
        self.critical_threshold = 75.0  # Critical alert if coverage < 75%

    def get_coverage_stats(self) -> Dict[str, any]:
        """
        Get current embedding coverage statistics.

        Returns:
            Dictionary with coverage metrics
        """
        try:
            with sqlite3.connect(self.db_path) as db:
                # Get total records
                total_result = db.execute(
                    "SELECT COUNT(*) as count FROM semantic_memory"
                ).fetchone()
                total_records = total_result[0]

                # Get records with embeddings
                embedding_result = db.execute(
                    "SELECT COUNT(*) as count FROM semantic_memory "
                    "WHERE embedding IS NOT NULL AND embedding != ''"
                ).fetchone()
                records_with_embeddings = embedding_result[0]

                # Calculate coverage
                coverage_percent = (records_with_embeddings / total_records * 100) if total_records > 0 else 0

                # Get coverage by content type
                coverage_by_type = self._get_coverage_by_content_type(db)

                # Get recent embedding activity
                recent_activity = self._get_recent_embedding_activity(db)

                return {
                    "timestamp": datetime.now().isoformat(),
                    "total_records": total_records,
                    "records_with_embeddings": records_with_embeddings,
                    "records_missing_embeddings": total_records - records_with_embeddings,
                    "coverage_percent": round(coverage_percent, 2),
                    "coverage_by_type": coverage_by_type,
                    "recent_activity": recent_activity,
                    "status": self._get_status(coverage_percent),
                }

        except Exception as e:
            logger.error("Failed to get coverage stats", error=str(e))
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }

    def _get_coverage_by_content_type(self, db: sqlite3.Connection) -> Dict[str, Dict[str, any]]:
        """Get coverage statistics broken down by content type."""
        query = """
        SELECT
            content_type,
            COUNT(*) as total,
            COUNT(CASE WHEN embedding IS NOT NULL AND embedding != '' THEN 1 END) as with_embeddings
        FROM semantic_memory
        GROUP BY content_type
        ORDER BY total DESC
        """

        results = db.execute(query).fetchall()
        coverage_by_type = {}

        for content_type, total, with_embeddings in results:
            coverage = (with_embeddings / total * 100) if total > 0 else 0
            coverage_by_type[content_type] = {
                "total": total,
                "with_embeddings": with_embeddings,
                "missing_embeddings": total - with_embeddings,
                "coverage_percent": round(coverage, 2)
            }

        return coverage_by_type

    def _get_recent_embedding_activity(self, db: sqlite3.Connection) -> Dict[str, any]:
        """Get recent embedding generation activity."""
        # Get embeddings generated in last 24 hours
        yesterday = datetime.now() - timedelta(days=1)

        recent_query = """
        SELECT COUNT(*) as count
        FROM semantic_memory
        WHERE embedding IS NOT NULL
        AND embedding != ''
        AND created_at >= ?
        """

        recent_result = db.execute(recent_query, (yesterday.isoformat(),)).fetchone()
        recent_count = recent_result[0]

        # Get embeddings generated in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)

        hourly_query = """
        SELECT COUNT(*) as count
        FROM semantic_memory
        WHERE embedding IS NOT NULL
        AND embedding != ''
        AND created_at >= ?
        """

        hourly_result = db.execute(hourly_query, (one_hour_ago.isoformat(),)).fetchone()
        hourly_count = hourly_result[0]

        return {
            "embeddings_last_24h": recent_count,
            "embeddings_last_hour": hourly_count,
            "last_embedding_time": self._get_last_embedding_time(db)
        }

    def _get_last_embedding_time(self, db: sqlite3.Connection) -> Optional[str]:
        """Get timestamp of most recent embedding generation."""
        query = """
        SELECT created_at
        FROM semantic_memory
        WHERE embedding IS NOT NULL AND embedding != ''
        ORDER BY created_at DESC
        LIMIT 1
        """

        result = db.execute(query).fetchone()
        return result[0] if result else None

    def _get_status(self, coverage_percent: float) -> str:
        """Get health status based on coverage percentage."""
        if coverage_percent >= self.alert_threshold:
            return "healthy"
        elif coverage_percent >= self.critical_threshold:
            return "degraded"
        else:
            return "critical"

    def check_backfill_progress(self) -> Dict[str, any]:
        """
        Check backfill progress for records without embeddings.

        Returns:
            Backfill progress information
        """
        try:
            with sqlite3.connect(self.db_path) as db:
                # Count records without embeddings
                without_embeddings = db.execute(
                    "SELECT COUNT(*) as count FROM semantic_memory "
                    "WHERE embedding IS NULL OR embedding = ''"
                ).fetchone()[0]

                # Estimate time to complete based on recent rate
                recent_rate = self._estimate_embedding_rate(db)

                if without_embeddings > 0 and recent_rate > 0:
                    estimated_hours = without_embeddings / recent_rate / 60  # records per minute to hours
                    eta = datetime.now() + timedelta(hours=estimated_hours)
                else:
                    eta = None

                return {
                    "records_without_embeddings": without_embeddings,
                    "estimated_rate_per_minute": recent_rate,
                    "estimated_completion_time": eta.isoformat() if eta else None,
                    "status": "in_progress" if without_embeddings > 0 else "completed"
                }

        except Exception as e:
            logger.error("Failed to check backfill progress", error=str(e))
            return {"error": str(e), "status": "error"}

    def _estimate_embedding_rate(self, db: sqlite3.Connection) -> float:
        """Estimate embedding generation rate based on recent activity."""
        # Get embeddings from last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)

        query = """
        SELECT COUNT(*) as count
        FROM semantic_memory
        WHERE embedding IS NOT NULL
        AND embedding != ''
        AND created_at >= ?
        """

        result = db.execute(query, (one_hour_ago.isoformat(),)).fetchone()
        return result[0] / 60.0  # records per minute

    def generate_health_report(self) -> str:
        """Generate a comprehensive health report."""
        stats = self.get_coverage_stats()
        backfill = self.check_backfill_progress()

        report = f"""
📊 DevStream Embedding Coverage Report
{'='*50}

📈 Coverage Status: {stats.get('status', 'unknown').upper()}
   Coverage: {stats.get('coverage_percent', 0):.1f}%
   Total Records: {stats.get('total_records', 0):,}
   With Embeddings: {stats.get('records_with_embeddings', 0):,}
   Missing Embeddings: {stats.get('records_missing_embeddings', 0):,}

📂 Coverage by Type:
"""

        for content_type, type_stats in stats.get('coverage_by_type', {}).items():
            report += f"   {content_type}: {type_stats['coverage_percent']:.1f}% ({type_stats['with_embeddings']:,}/{type_stats['total']:,})\n"

        report += f"""
🔄 Backfill Status: {backfill.get('status', 'unknown').upper()}
   Records Remaining: {backfill.get('records_without_embeddings', 0):,}
   Rate: {backfill.get('estimated_rate_per_minute', 0):.1f} records/min
   ETA: {backfill.get('estimated_completion_time', 'Unknown')}

🕒 Recent Activity:
   Last 24h: {stats.get('recent_activity', {}).get('embeddings_last_24h', 0):,} embeddings
   Last Hour: {stats.get('recent_activity', {}).get('embeddings_last_hour', 0):,} embeddings
   Last Embedding: {stats.get('recent_activity', {}).get('last_embedding_time', 'Never')}

📋 Health Check:
"""

        status = stats.get('status', 'unknown')
        if status == 'healthy':
            report += "   ✅ Embedding coverage is healthy (≥90%)\n"
        elif status == 'degraded':
            report += f"   ⚠️  Embedding coverage is degraded ({stats.get('coverage_percent', 0):.1f}% < 90%)\n"
        else:
            report += f"   ❌ Embedding coverage is critical ({stats.get('coverage_percent', 0):.1f}% < 75%)\n"

        report += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return report

    def store_coverage_snapshot(self) -> bool:
        """
        Store current coverage snapshot for historical tracking.

        Returns:
            True if snapshot stored successfully
        """
        try:
            stats = self.get_coverage_stats()

            with sqlite3.connect(self.db_path) as db:
                # Create coverage_history table if it doesn't exist
                db.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_coverage_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_records INTEGER NOT NULL,
                        records_with_embeddings INTEGER NOT NULL,
                        coverage_percent REAL NOT NULL,
                        status TEXT NOT NULL,
                        snapshot_data TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Insert snapshot
                db.execute("""
                    INSERT INTO embedding_coverage_history (
                        timestamp, total_records, records_with_embeddings,
                        coverage_percent, status, snapshot_data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    stats['timestamp'],
                    stats['total_records'],
                    stats['records_with_embeddings'],
                    stats['coverage_percent'],
                    stats['status'],
                    json.dumps(stats)
                ))

            logger.info("Coverage snapshot stored successfully")
            return True

        except Exception as e:
            logger.error("Failed to store coverage snapshot", error=str(e))
            return False

    def get_coverage_trend(self, hours: int = 24) -> List[Dict[str, any]]:
        """
        Get coverage trend over time.

        Args:
            hours: Number of hours to look back

        Returns:
            List of coverage snapshots
        """
        try:
            with sqlite3.connect(self.db_path) as db:
                since = datetime.now() - timedelta(hours=hours)

                query = """
                SELECT timestamp, total_records, records_with_embeddings,
                       coverage_percent, status, created_at
                FROM embedding_coverage_history
                WHERE created_at >= ?
                ORDER BY created_at ASC
                """

                results = db.execute(query, (since.isoformat(),)).fetchall()

                return [
                    {
                        "timestamp": row[0],
                        "total_records": row[1],
                        "records_with_embeddings": row[2],
                        "coverage_percent": row[3],
                        "status": row[4],
                        "created_at": row[5]
                    }
                    for row in results
                ]

        except Exception as e:
            logger.error("Failed to get coverage trend", error=str(e))
            return []


def main():
    """Main function for standalone execution"""
    import sys
    from pathlib import Path

    # Get database path from command line or use default
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/devstream.db"

    monitor = EmbeddingCoverageMonitor(db_path)

    # Generate and print health report
    report = monitor.generate_health_report()
    print(report)

    # Store snapshot
    monitor.store_coverage_snapshot()

    return 0


if __name__ == "__main__":
    sys.exit(main())