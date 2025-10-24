#!/usr/bin/env python3
"""
Database Update Monitor - Detects stale database updates and alerts user.

This script monitors the DevStream database for update staleness and alerts
when no new records have been written for more than the threshold period.

Usage:
    .devstream/bin/python .claude/hooks/devstream/monitoring/database_update_monitor.py --threshold 600
"""

import asyncio
import sys
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

# Setup structured logging
logger = structlog.get_logger(__name__)


class DatabaseUpdateMonitor:
    """Monitor for detecting stale database updates."""

    def __init__(self, threshold_seconds: int = 600):
        """
        Initialize database update monitor.

        Args:
            threshold_seconds: Staleness threshold in seconds (default: 600 = 10 minutes)
        """
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.db_path = self.project_root / "data" / "devstream.db"
        self.log_file = self.project_root / ".claude" / "logs" / "devstream" / "database_update_monitor.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.threshold_seconds = threshold_seconds
        self.alert_sent = False  # Prevent duplicate alerts

    def _get_last_update_time(self) -> Optional[datetime]:
        """
        Get timestamp of most recent database insert.

        Checks semantic_memory table for most recent created_at timestamp.

        Returns:
            Last update datetime or None if no records
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get most recent created_at from semantic_memory
            cursor.execute("""
                SELECT MAX(created_at)
                FROM semantic_memory
            """)

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                # Parse ISO 8601 timestamp: "2025-10-11T23:42:01.451Z"
                timestamp_str = result[0]
                # Handle both with and without fractional seconds
                try:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback: try without timezone
                    return datetime.fromisoformat(timestamp_str.replace('Z', ''))

            return None

        except sqlite3.Error as e:
            logger.error("Database query error", error=str(e))
            return None

        except Exception as e:
            logger.error("Failed to get last update time", error=str(e))
            return None

    def _calculate_staleness(self, last_update: datetime) -> int:
        """
        Calculate staleness in seconds since last update.

        Args:
            last_update: Last update timestamp

        Returns:
            Staleness in seconds
        """
        now = datetime.utcnow()
        # Ensure timezone-naive comparison
        if last_update.tzinfo:
            last_update = last_update.replace(tzinfo=None)

        delta = now - last_update
        return int(delta.total_seconds())

    async def check_staleness(self) -> Dict[str, Any]:
        """
        Check database staleness and return alert status.

        Returns:
            Dictionary with staleness check results
        """
        timestamp = datetime.utcnow()

        # Get last update time
        last_update = self._get_last_update_time()

        if last_update is None:
            result = {
                "timestamp": timestamp.isoformat() + "Z",
                "status": "error",
                "message": "Unable to query database or no records found",
                "staleness_seconds": None,
                "alert": False
            }

            logger.error("Database staleness check failed: no records")
            return result

        # Calculate staleness
        staleness_seconds = self._calculate_staleness(last_update)

        # Check if stale
        is_stale = staleness_seconds > self.threshold_seconds

        result = {
            "timestamp": timestamp.isoformat() + "Z",
            "status": "stale" if is_stale else "ok",
            "last_update": last_update.isoformat() + "Z",
            "staleness_seconds": staleness_seconds,
            "threshold_seconds": self.threshold_seconds,
            "alert": is_stale and not self.alert_sent
        }

        # Log to file
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Log to structured logger
        if is_stale:
            if not self.alert_sent:
                logger.error(
                    "Database staleness ALERT",
                    staleness_seconds=staleness_seconds,
                    threshold_seconds=self.threshold_seconds,
                    last_update=last_update.isoformat()
                )
                self.alert_sent = True
            else:
                logger.warning(
                    "Database still stale",
                    staleness_seconds=staleness_seconds
                )
        else:
            logger.info(
                "Database staleness check: OK",
                staleness_seconds=staleness_seconds
            )
            self.alert_sent = False  # Reset alert flag when recovered

        return result

    async def monitor_loop(self, interval_seconds: int = 60):
        """
        Continuous staleness monitoring loop.

        Args:
            interval_seconds: Check interval in seconds (default: 60)
        """
        logger.info(
            "Starting database staleness monitor",
            interval=interval_seconds,
            threshold=self.threshold_seconds
        )

        while True:
            try:
                result = await self.check_staleness()

                # Alert on stale database
                if result.get("alert"):
                    print(
                        f"⚠️  DATABASE ALERT: No updates for {result['staleness_seconds']}s "
                        f"(threshold: {self.threshold_seconds}s)\n"
                        f"   Last update: {result['last_update']}\n"
                        f"   Possible causes: MCP timeout, zombie processes, hook failure",
                        file=sys.stderr
                    )

            except Exception as e:
                logger.error("Staleness check loop error", error=str(e))

            await asyncio.sleep(interval_seconds)


async def main():
    """Main entry point for database update monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Database Update Monitor")
    parser.add_argument(
        "--threshold",
        type=int,
        default=600,
        help="Staleness threshold in seconds (default: 600 = 10 minutes)"
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

    monitor = DatabaseUpdateMonitor(threshold_seconds=args.threshold)

    # Single check mode
    if args.once:
        result = await monitor.check_staleness()
        import json
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "ok" else 1)

    # Continuous monitoring mode
    await monitor.monitor_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
