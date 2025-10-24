#!/usr/bin/env python3
"""
DevStream Backfill Monitor - Real-time Progress Tracker

Monitora lo stato del processo di backfill embeddings in esecuzione.
Fornisce statistiche in tempo reale su records processati e rimanenti.

Usage:
    python scripts/monitor_backfill.py [--watch]

    --watch: Modalità continua (aggiorna ogni 2 secondi)
"""

import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))
from sqlite_vec_helper import get_db_connection_with_vec


class BackfillMonitor:
    """Monitor per il processo di backfill embeddings."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent.parent / 'data' / 'devstream.db')
        self.start_time = datetime.now()

    def get_database_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche dal database."""
        try:
            with get_db_connection_with_vec(self.db_path) as conn:
                cursor = conn.cursor()

                # Query per statistiche
                cursor.execute("SELECT COUNT(*) FROM semantic_memory")
                total_records = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(*) FROM semantic_memory
                    WHERE embedding IS NULL OR embedding = ''
                """)
                missing_embeddings = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(*) FROM semantic_memory
                    WHERE embedding IS NOT NULL AND embedding != ''
                """)
                completed_embeddings = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(*) FROM semantic_memory
                    WHERE embedding_model = 'embeddinggemma:300m' AND created_at > ?
                """, (self.start_time.strftime('%Y-%m-%d %H:%M:%S'),))
                recent_updates = cursor.fetchone()[0]

                return {
                    'total_records': total_records,
                    'missing_embeddings': missing_embeddings,
                    'completed_embeddings': completed_embeddings,
                    'recent_updates': recent_updates,
                    'completion_percentage': (completed_embeddings / total_records * 100) if total_records > 0 else 0
                }

        except Exception as e:
            return {'error': str(e)}

    def get_process_status(self) -> Dict[str, Any]:
        """Controlla se il processo backfill è in esecuzione."""
        try:
            import subprocess
            result = subprocess.run(
                ['pgrep', '-f', 'backfill_embeddings.py'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                return {
                    'running': True,
                    'pids': [int(pid) for pid in pids if pid.strip()],
                    'process_count': len([p for p in pids if p.strip()])
                }
            else:
                return {'running': False, 'pids': [], 'process_count': 0}

        except Exception as e:
            return {'running': 'unknown', 'error': str(e)}

    def get_latest_log_entries(self, limit: int = 5) -> list:
        """Ottiene le ultime entry dal log di backfill."""
        try:
            import glob
            log_files = glob.glob('backfill_*.log') + glob.glob('backfill_verbose_*.log')

            if not log_files:
                return []

            # Prendi il file di log più recente
            latest_log = max(log_files, key=lambda x: Path(x).stat().st_mtime)

            with open(latest_log, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-limit:] if line.strip()]

        except Exception:
            return []

    def display_status(self):
        """Mostra lo stato completo."""
        # Clear screen (works on Unix-like systems)
        print('\033[2J\033[H', end='')

        print("🧠 DEVSTREAM BACKFILL MONITOR")
        print("=" * 50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Database: {self.db_path}")
        print()

        # Database stats
        stats = self.get_database_stats()
        if 'error' in stats:
            print(f"❌ Database error: {stats['error']}")
            return

        print("📊 DATABASE STATISTICS")
        print("-" * 30)
        print(f"Total records:        {stats['total_records']:,}")
        print(f"Completed embeddings: {stats['completed_embeddings']:,}")
        print(f"Missing embeddings:  {stats['missing_embeddings']:,}")
        print(f"Completion:           {stats['completion_percentage']:.1f}%")
        print()

        # Process status
        proc_status = self.get_process_status()
        print("🔄 PROCESS STATUS")
        print("-" * 30)
        if proc_status['running'] == True:
            print(f"✅ Backfill process running")
            print(f"📋 Active PIDs: {', '.join(map(str, proc_status['pids']))}")
            print(f"🔢 Process count: {proc_status['process_count']}")
        elif proc_status['running'] == False:
            print("❌ No backfill process found")
            print("💡 Use: python .claude/hooks/devstream/memory/backfill_embeddings.py")
        else:
            print(f"⚠️  Cannot determine process status: {proc_status.get('error', 'unknown')}")
        print()

        # Progress bar
        if stats['total_records'] > 0:
            completed = stats['completed_embeddings']
            total = stats['total_records']
            percentage = completed / total

            bar_length = 40
            filled = int(bar_length * percentage)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"📈 PROGRESS BAR")
            print("-" * 30)
            print(f"[{bar}] {percentage:.1%}")
            print(f"   {completed:,} / {total:,} records")
            print()

        # Recent log entries
        recent_logs = self.get_latest_log_entries(3)
        if recent_logs:
            print("📝 RECENT LOG ENTRIES")
            print("-" * 30)
            for entry in recent_logs:
                if entry:
                    print(f"  • {entry}")
            print()

        # Estimated time remaining
        if stats['missing_embeddings'] > 0 and stats['recent_updates'] > 0:
            # Simple estimation based on recent updates
            estimated_remaining = stats['missing_embeddings'] / max(stats['recent_updates'], 1)
            print(f"⏱️  Estimated time remaining: {estimated_remaining:.0f} minutes (based on recent rate)")

        # Commands
        print("🔧 QUICK COMMANDS")
        print("-" * 30)
        print("  Stop backfill:      pkill -f backfill_embeddings.py")
        print("  Start backfill:      python .claude/hooks/devstream/memory/backfill_embeddings.py")
        print("  Full log:           tail -f backfill_verbose_*.log")
        print()


def main():
    parser = argparse.ArgumentParser(description="Monitor DevStream backfill progress")
    parser.add_argument(
        '--watch',
        action='store_true',
        help="Watch mode - continuously update every 2 seconds"
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=2,
        help="Update interval in seconds (default: 2)"
    )

    args = parser.parse_args()

    monitor = BackfillMonitor()

    if args.watch:
        try:
            while True:
                monitor.display_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
    else:
        monitor.display_status()


if __name__ == "__main__":
    main()