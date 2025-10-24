#!/usr/bin/env .devstream/bin/python
"""
Trigger-Based JSON→BLOB Migration
Leverages existing sync_embedding_update trigger for conversion

Phase 2: Migrate 48,538 existing JSON embeddings through trigger
Expected: -327 MB JSON waste + proper architecture
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = "data/devstream.db"
BACKUP_PATH = "data/devstream.db.backup-phase2"

def get_db_connection_with_vec(db_path: str):
    """Load database with sqlite-vec extension"""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension('.devstream/lib/python3.11/site-packages/sqlite_vec/vec0')
    return conn

def check_prerequisites():
    """Check that prerequisites are met before migration"""
    print("🔍 Checking prerequisites...")

    # Check database exists
    if not Path(DB_PATH).exists():
        print(f"❌ Database not found: {DB_PATH}")
        return False

    # Check trigger exists
    conn = get_db_connection_with_vec(DB_PATH)
    try:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='sync_embedding_update'"
        ).fetchone()
        if not result:
            print("❌ sync_embedding_update trigger not found")
            conn.close()
            return False
        print("✅ sync_embedding_update trigger exists")
    except Exception as e:
        print(f"❌ Error checking trigger: {e}")
        conn.close()
        return False

    # Check vec_f32 function works
    try:
        result = conn.execute("SELECT vec_f32('[0.1, 0.2]')").fetchone()
        if not result:
            print("❌ vec_f32 function not working")
            conn.close()
            return False
        print("✅ vec_f32 function working")
    except Exception as e:
        print(f"❌ Error testing vec_f32: {e}")
        conn.close()
        return False

    conn.close()
    return True

def create_backup():
    """Create backup before Phase 2"""
    print(f"📦 Creating backup: {BACKUP_PATH}")
    try:
        import shutil
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"✅ Backup created: {BACKUP_PATH}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def backfill_via_trigger():
    """
    Trigger-based migration: UPDATE each JSON embedding to activate trigger.
    Trigger handles: JSON→BLOB conversion + auto-cleanup
    """
    print("\n🚀 Starting Phase 2: Backfill via trigger")

    conn = get_db_connection_with_vec(DB_PATH)
    cursor = conn.cursor()

    # Count records needing migration
    cursor.execute("""
        SELECT COUNT(*) FROM semantic_memory
        WHERE embedding IS NOT NULL AND embedding != ''
    """)
    total = cursor.fetchone()[0]
    print(f"📊 Records to migrate: {total:,}")

    if total == 0:
        print("✅ No JSON embeddings to migrate")
        conn.close()
        return True

    # Fetch all IDs with JSON embeddings
    cursor.execute("""
        SELECT id FROM semantic_memory
        WHERE embedding IS NOT NULL AND embedding != ''
        ORDER BY created_at DESC
    """)
    record_ids = [row[0] for row in cursor.fetchall()]

    # Batch UPDATE to trigger sync
    batch_size = 1000
    migrated = 0

    for i in range(0, len(record_ids), batch_size):
        batch = record_ids[i:i+batch_size]

        # UPDATE triggers sync_embedding_update for each record
        placeholders = ','.join('?' * len(batch))
        cursor.execute(f"""
            UPDATE semantic_memory
            SET embedding = embedding  -- Dummy update to trigger
            WHERE id IN ({placeholders})
        """, batch)

        conn.commit()

        migrated = min(i + batch_size, total)
        progress = migrated / total * 100
        print(f"✓ Migrated: {migrated:,}/{total:,} ({progress:.1f}%)")

    conn.close()
    print("✅ Backfill complete via trigger")
    return True

def vacuum_and_checkpoint():
    """Run VACUUM and WAL checkpoint to reclaim space"""
    print("\n🧹 Running VACUUM and WAL checkpoint...")

    conn = get_db_connection_with_vec(DB_PATH)
    cursor = conn.cursor()

    try:
        # VACUUM to reclaim space
        print("Running VACUUM...")
        cursor.execute("VACUUM")

        # WAL checkpoint
        print("Running WAL checkpoint...")
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        conn.close()
        print("✅ Database vacuumed and WAL checkpoint complete")
        return True
    except Exception as e:
        print(f"❌ VACUUM/checkpoint failed: {e}")
        conn.close()
        return False

def verify_results():
    """Verify migration results"""
    print("\n🔍 Verifying migration results...")

    conn = get_db_connection_with_vec(DB_PATH)
    cursor = conn.cursor()

    # Check JSON embeddings remaining
    cursor.execute("""
        SELECT COUNT(*) FROM semantic_memory
        WHERE embedding IS NOT NULL AND embedding != ''
    """)
    json_remaining = cursor.fetchone()[0]
    print(f"📝 JSON embeddings remaining: {json_remaining:,}")

    # Check BLOB embeddings created
    cursor.execute("SELECT COUNT(*) FROM vec_semantic_memory")
    blob_count = cursor.fetchone()[0]
    print(f"📦 BLOB embeddings in vec_semantic_memory: {blob_count:,}")

    # Check database size
    conn.close()
    db_size = Path(DB_PATH).stat().st_size / (1024 * 1024)  # MB
    print(f"💾 Database size: {db_size:.1f} MB")

    # Success criteria
    if json_remaining == 0:
        print("✅ All JSON embeddings migrated")
        success = True
    else:
        print(f"❌ {json_remaining:,} JSON embeddings remaining")
        success = False

    if blob_count >= 45000:  # Expect ~45,385 + our test
        print("✅ Expected number of BLOB embeddings created")
    else:
        print(f"⚠️  Expected ~45,385 BLOB embeddings, got {blob_count:,}")

    return success

def main():
    """Main execution function"""
    print("=== Phase 2: JSON→BLOB Backfill via Trigger ===")

    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met")
        sys.exit(1)

    # Create backup
    if not create_backup():
        print("❌ Backup failed")
        sys.exit(1)

    # Execute backfill
    if not backfill_via_trigger():
        print("❌ Backfill failed")
        sys.exit(1)

    # Vacuum and checkpoint
    if not vacuum_and_checkpoint():
        print("❌ Vacuum/checkpoint failed")
        sys.exit(1)

    # Verify results
    if not verify_results():
        print("❌ Verification failed")
        sys.exit(1)

    print("\n🎉 Phase 2 completed successfully!")
    print("✅ All JSON embeddings migrated to BLOB format")
    print("✅ Database optimized and space reclaimed")

if __name__ == "__main__":
    main()