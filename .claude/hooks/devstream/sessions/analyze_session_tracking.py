#!/usr/bin/env python3
"""
Script per analizzare i problemi del sistema di session tracking
"""
import asyncio
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def analyze_sessions():
    """Analizza lo stato attuale delle sessioni nel database"""

    # Find database path
    db_path = os.environ.get('DEVSTREAM_DB_PATH')
    if not db_path:
        # Fallback to default path
        project_root = Path(__file__).parent.parent.parent.parent
        db_path = str(project_root / "data" / "devstream.db")

    print(f"📊 Analizzando sessioni dal database: {db_path}")
    print("=" * 60)

    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        print(f"❌ Database non trovato: {db_path}")
        return

    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. Session Summary Stats
        print("\n🔍 STATISTICHE SESSIONI")
        print("-" * 30)

        cursor.execute("""
            SELECT
                COUNT(*) as total_sessions,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_sessions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
                MAX(created_at) as latest_session
            FROM work_sessions
        """)

        stats = cursor.fetchone()
        print(f"Sessioni totali: {stats[0]}")
        print(f"Sessioni attive: {stats[1]}")
        print(f"Sessioni completate: {stats[2]}")
        print(f"Sessioni fallite: {stats[3]}")
        print(f"Ultima sessione: {stats[4] or 'N/A'}")

        # 2. Active Sessions Detail
        if stats[1] > 0:
            print(f"\n🚨 SESSIONI ATTIVE ({stats[1]})")
            print("-" * 30)

            cursor.execute("""
                SELECT session_id, title, created_at, updated_at, status,
                       duration_minutes, total_estimated_tokens, session_metadata
                FROM work_sessions
                WHERE status = 'active'
                ORDER BY created_at DESC
            """)

            active_sessions = cursor.fetchall()
            for session in active_sessions:
                (session_id, title, created_at, updated_at, status,
                 duration_minutes, tokens, metadata) = session

                print(f"\n📌 Sessione: {session_id}")
                print(f"   Titolo: {title or 'N/A'}")
                print(f"   Creata: {created_at}")
                print(f"   Aggiornata: {updated_at}")
                print(f"   Durata: {duration_minutes or 0} minuti")
                print(f"   Token: {tokens or 0}")
                print(f"   Metadata: {metadata or 'N/A'}")

        # 3. Recent Sessions (last 10)
        print(f"\n📋 ULTIME 10 SESSIONI")
        print("-" * 30)

        cursor.execute("""
            SELECT session_id, title, created_at, status, duration_minutes,
                   total_estimated_tokens, task_count, files_modified
            FROM work_sessions
            ORDER BY created_at DESC
            LIMIT 10
        """)

        recent_sessions = cursor.fetchall()
        for session in recent_sessions:
            (session_id, title, created_at, status, duration,
             tokens, task_count, files) = session

            print(f"\n📝 {session_id}")
            print(f"   Titolo: {title or 'N/A'}")
            print(f"   Stato: {status}")
            print(f"   Creata: {created_at}")
            print(f"   Durata: {duration or 0} min")
            print(f"   Token: {tokens or 0}")
            print(f"   Tasks: {task_count or 0}")
            print(f"   Files: {files or 0}")

        # 4. Check for duplicate sessions
        print(f"\n🔎 DUPLICATE SESSIONS CHECK")
        print("-" * 30)

        cursor.execute("""
            SELECT DATE(created_at) as session_date, COUNT(*) as count
            FROM work_sessions
            GROUP BY DATE(created_at)
            HAVING COUNT(*) > 1
            ORDER BY session_date DESC
        """)

        duplicates = cursor.fetchall()
        if duplicates:
            print("Trovate sessioni multiple nello stesso giorno:")
            for date, count in duplicates:
                print(f"   📅 {date}: {count} sessioni")
        else:
            print("✅ Nessun duplicato rilevato per giorno")

        # 5. Duration Analysis
        print(f"\n⏱️ ANALISI DURATA")
        print("-" * 30)

        cursor.execute("""
            SELECT
                AVG(duration_minutes) as avg_duration,
                MIN(duration_minutes) as min_duration,
                MAX(duration_minutes) as max_duration,
                COUNT(CASE WHEN duration_minutes = 0 OR duration_minutes IS NULL THEN 1 END) as zero_duration
            FROM work_sessions
            WHERE status = 'completed'
        """)

        duration_stats = cursor.fetchone()
        if duration_stats[0]:
            print(f"Durata media: {duration_stats[0]:.1f} minuti")
            print(f"Durata minima: {duration_stats[1]} minuti")
            print(f"Durata massima: {duration_stats[2]} minuti")
            print(f"Sessioni a durata zero: {duration_stats[3]}")
        else:
            print("Nessuna sessione completata per analizzare la durata")

        # 6. Token Analysis
        print(f"\n💰 ANALISI TOKEN")
        print("-" * 30)

        cursor.execute("""
            SELECT
                AVG(total_estimated_tokens) as avg_tokens,
                SUM(total_estimated_tokens) as total_tokens,
                COUNT(CASE WHEN total_estimated_tokens = 0 OR total_estimated_tokens IS NULL THEN 1 END) as zero_tokens
            FROM work_sessions
        """)

        token_stats = cursor.fetchone()
        if token_stats[0]:
            print(f"Token medi per sessione: {token_stats[0]:.0f}")
            print(f"Token totali: {token_stats[1] or 0}")
            print(f"Sessioni con 0 token: {token_stats[2]}")
        else:
            print("Nessun dato token disponibile")

        # 7. Session Metadata Analysis
        print(f"\n📊 METADATA ANALYSIS")
        print("-" * 30)

        cursor.execute("""
            SELECT session_id, session_metadata
            FROM work_sessions
            WHERE session_metadata IS NOT NULL AND session_metadata != ''
            LIMIT 5
        """)

        metadata_samples = cursor.fetchall()
        for session_id, metadata in metadata_samples:
            try:
                parsed = json.loads(metadata)
                print(f"\n📌 Sessione {session_id}:")
                print(f"   Context items: {len(parsed.get('context_items', []))}")
                print(f"   Tasks completed: {parsed.get('tasks_completed', 0)}")
                print(f"   Files modified: {len(parsed.get('files_modified', []))}")
            except:
                print(f"❌ Metadata non valido per sessione {session_id}")

        conn.close()

    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_sessions())