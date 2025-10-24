#!/usr/bin/env python3
"""
DevStream Backfill Monitor

Real-time monitoring tool for embedding backfill progress.
Provides live updates and completion detection.
"""

import sqlite3
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class BackfillMonitor:
    """
    Real-time backfill progress monitor.
    """

    def __init__(self, db_path: str = "data/devstream.db"):
        self.db_path = db_path
        self.start_time = datetime.now()
        self.high_priority_target = 5688
        self.last_check = {"embeddings": 0, "time": datetime.now()}

    def get_current_status(self) -> Dict[str, Any]:
        """Get current backfill status."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Overall statistics
            cursor = conn.execute("SELECT COUNT(*) FROM semantic_memory WHERE embedding_blob IS NOT NULL")
            total_embeddings = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM semantic_memory WHERE embedding_blob IS NULL")
            missing_embeddings = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM semantic_memory")
            total_records = cursor.fetchone()[0]

            # High priority progress
            cursor = conn.execute("""
                SELECT COUNT(*) FROM semantic_memory
                WHERE embedding_blob IS NOT NULL
                AND content_type IN ('code', 'documentation', 'decision', 'learning')
                AND embedding_model = 'embeddinggemma:300m'
            """)
            high_priority_processed = cursor.fetchone()[0]

            # Recent activity (last 5 minutes)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM semantic_memory
                WHERE embedding_blob IS NOT NULL
                AND created_at > datetime('now', '-5 minutes')
            """)
            recent_embeddings = cursor.fetchone()[0]

            # Content type breakdown
            cursor = conn.execute("""
                SELECT content_type, COUNT(*)
                FROM semantic_memory
                WHERE embedding_blob IS NOT NULL
                AND content_type IN ('code', 'documentation', 'decision', 'learning')
                GROUP BY content_type
            """)
            by_content_type = dict(cursor.fetchall())

            conn.close()

            return {
                "timestamp": datetime.now(),
                "total_embeddings": total_embeddings,
                "missing_embeddings": missing_embeddings,
                "total_records": total_records,
                "completion_percentage": (total_embeddings / total_records * 100) if total_records > 0 else 0,
                "high_priority_processed": high_priority_processed,
                "high_priority_target": self.high_priority_target,
                "high_priority_percentage": (high_priority_processed / self.high_priority_target * 100),
                "recent_embeddings": recent_embeddings,
                "by_content_type": by_content_type,
                "elapsed_minutes": (datetime.now() - self.start_time).total_seconds() / 60
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now()}

    def calculate_rate(self, current_status: Dict[str, Any]) -> Dict[str, float]:
        """Calculate processing rate based on recent activity."""
        if "error" in current_status:
            return {"rate_per_minute": 0, "rate_per_hour": 0}

        current_embeddings = current_status["total_embeddings"]
        current_time = current_status["timestamp"]

        # Calculate rate since last check
        if self.last_check["embeddings"] > 0:
            time_diff = (current_time - self.last_check["time"]).total_seconds()
            if time_diff > 0:
                embedding_diff = current_embeddings - self.last_check["embeddings"]
                rate_per_minute = (embedding_diff / time_diff) * 60
                rate_per_hour = rate_per_minute * 60
            else:
                rate_per_minute = 0
                rate_per_hour = 0
        else:
            rate_per_minute = 0
            rate_per_hour = 0

        # Update last check
        self.last_check = {"embeddings": current_embeddings, "time": current_time}

        return {
            "rate_per_minute": rate_per_minute,
            "rate_per_hour": rate_per_hour
        }

    def estimate_completion(self, current_status: Dict[str, Any], rate: Dict[str, float]) -> Dict[str, Any]:
        """Estimate completion time based on current rate."""
        if "error" in current_status or rate["rate_per_minute"] <= 0:
            return {"estimated_minutes": None, "estimated_time": None}

        remaining_high_priority = self.high_priority_target - current_status["high_priority_processed"]
        if remaining_high_priority <= 0:
            return {"estimated_minutes": 0, "estimated_time": "COMPLETED"}

        estimated_minutes = remaining_high_priority / rate["rate_per_minute"]
        completion_time = datetime.now() + timedelta(minutes=estimated_minutes)

        return {
            "estimated_minutes": estimated_minutes,
            "estimated_time": completion_time.strftime("%H:%M:%S")
        }

    def format_status_report(self, status: Dict[str, Any], rate: Dict[str, float], completion: Dict[str, Any]) -> str:
        """Format status report for display."""
        if "error" in status:
            return f"❌ Error: {status['error']}"

        report = []
        report.append("📊 DEVSTREAM BACKFILL MONITOR")
        report.append("=" * 45)
        report.append(f"🕐 Timestamp: {status['timestamp'].strftime('%H:%M:%S')}")
        report.append(f"⏱️  Elapsed: {status['elapsed_minutes']:.1f} minutes")
        report.append("")

        report.append("📈 OVERALL PROGRESS")
        report.append(f"   📊 Total embeddings: {status['total_embeddings']:,}")
        report.append(f"   📉 Missing: {status['missing_embeddings']:,}")
        report.append(f"   ✅ Completion: {status['completion_percentage']:.3f}%")
        report.append("")

        report.append("🎯 HIGH PRIORITY TARGET")
        report.append(f"   📊 Processed: {status['high_priority_processed']:,}")
        report.append(f"   🎯 Target: {status['high_priority_target']:,}")
        report.append(f"   📈 Progress: {status['high_priority_percentage']:.1f}%")
        report.append(f"   📉 Remaining: {status['high_priority_target'] - status['high_priority_processed']:,}")
        report.append("")

        report.append("⚡ PERFORMANCE")
        report.append(f"   🚀 Rate: {rate['rate_per_minute']:.1f} embeddings/min")
        report.append(f"   📊 Recent (5min): {status['recent_embeddings']:,}")

        if completion["estimated_time"]:
            if completion["estimated_time"] == "COMPLETED":
                report.append(f"   ✅ Status: HIGH PRIORITY BACKFILL COMPLETED!")
            else:
                report.append(f"   ⏱️  ETA: {completion['estimated_time']} ({completion['estimated_minutes']:.1f} min)")

        report.append("")
        report.append("📋 BY CONTENT TYPE")
        for content_type, count in status['by_content_type'].items():
            report.append(f"   • {content_type}: {count:,}")

        return "\n".join(report)

    def monitor_continuously(self, interval_seconds: int = 30, max_duration_minutes: int = 30) -> None:
        """
        Monitor backfill progress continuously.

        Args:
            interval_seconds: Seconds between status checks
            max_duration_minutes: Maximum monitoring duration
        """
        start_monitor_time = datetime.now()
        end_monitor_time = start_monitor_time + timedelta(minutes=max_duration_minutes)

        print("🚀 STARTING CONTINUOUS MONITORING")
        print(f"⏱️  Check interval: {interval_seconds} seconds")
        print(f"🕐 Max duration: {max_duration_minutes} minutes")
        print()

        try:
            while datetime.now() < end_monitor_time:
                status = self.get_current_status()
                rate = self.calculate_rate(status)
                completion = self.estimate_completion(status, rate)

                # Clear screen and show status
                print("\033[2J\033[H")  # Clear screen
                print(self.format_status_report(status, rate, completion))

                # Check if high priority backfill is completed
                if status.get("high_priority_processed", 0) >= self.high_priority_target:
                    print("\n🎉 HIGH PRIORITY BACKFILL COMPLETED SUCCESSFULLY!")
                    break

                # Check if no recent activity
                if status.get("recent_embeddings", 0) == 0 and status["elapsed_minutes"] > 5:
                    print("\n⚠️  NO RECENT ACTIVITY - Backfill may be paused or completed")

                # Wait for next check
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")

        # Final status
        final_status = self.get_current_status()
        print("\n📊 FINAL MONITORING STATUS")
        print(f"Total embeddings: {final_status.get('total_embeddings', 0):,}")
        print(f"High priority processed: {final_status.get('high_priority_processed', 0):,}")
        print(f"Monitoring duration: {(datetime.now() - start_monitor_time).total_seconds() / 60:.1f} minutes")


def main():
    """Main monitoring function."""
    monitor = BackfillMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Continuous monitoring mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        monitor.monitor_continuously(interval, duration)
    else:
        # Single check mode
        status = monitor.get_current_status()
        rate = monitor.calculate_rate(status)
        completion = monitor.estimate_completion(status, rate)
        print(monitor.format_status_report(status, rate, completion))


if __name__ == "__main__":
    main()