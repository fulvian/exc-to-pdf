#!/usr/bin/env python3
"""
DevStream Vector Synchronization Setup Script

Applies enhanced triggers for automatic vec_semantic_memory synchronization.
Verifies functionality and performs testing.

Context7 Best Practices + DevStream Integration
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))
from sqlite_vec_helper import get_db_connection_with_vec

def create_trigger(conn):
    """Create the enhanced vector synchronization trigger."""
    print("🔧 Creating enhanced vector synchronization triggers...")

    # Read the SQL script
    script_path = Path(__file__).parent / 'create_vec_sync_trigger.sql'
    with open(script_path, 'r') as f:
        sql_script = f.read()

    # Execute the script
    cursor = conn.cursor()
    cursor.executescript(sql_script)
    conn.commit()
    print("✅ Enhanced triggers created successfully!")

def verify_schema(conn):
    """Verify database schema and trigger existence."""
    print("\n🔍 Verifying schema and triggers...")

    cursor = conn.cursor()

    # Check triggers
    cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    triggers = [row[0] for row in cursor.fetchall()]

    expected_triggers = [
        'sync_insert_memory',
        'sync_update_memory',
        'sync_delete_memory',
        'sync_embedding_update'
    ]

    print(f"📋 Triggers found: {len(triggers)}")
    for trigger in expected_triggers:
        status = "✅" if trigger in triggers else "❌"
        print(f"  {status} {trigger}")

    # Check vec_semantic_memory schema
    cursor.execute("PRAGMA table_info(vec_semantic_memory)")
    columns = [row[1] for row in cursor.fetchall()]
    expected_columns = ['embedding', 'content_type', 'memory_id', 'content_preview']

    print(f"\n📋 vec_semantic_memory columns: {len(columns)}")
    for col in expected_columns:
        status = "✅" if col in columns else "❌"
        print(f"  {status} {col}")

    return all(trigger in triggers for trigger in expected_triggers)

def test_synchronization(conn):
    """Test trigger functionality with a sample record."""
    print("\n🧪 Testing trigger functionality...")

    cursor = conn.cursor()

    # Create a test record with embedding
    test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_content = "Test content for vector synchronization verification"
    test_embedding = json.dumps([0.1, 0.2, 0.3, 0.4, 0.5] * 153 + [0.6])  # 768 dimensions

    # Insert test record
    print("  📝 Inserting test record...")
    cursor.execute('''
        INSERT INTO semantic_memory(
            id, content, content_type, embedding,
            embedding_model, embedding_dimension, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        test_id, test_content, "code", test_embedding,
        "embeddinggemma:300m", 768, datetime.now().isoformat()
    ))
    conn.commit()

    # Check if trigger synchronized to vec_semantic_memory
    cursor.execute('''
        SELECT COUNT(*) FROM vec_semantic_memory
        WHERE memory_id = ?
    ''', (test_id,))

    vec_count = cursor.fetchone()[0]

    # Test update synchronization
    print("  🔄 Testing update synchronization...")
    updated_embedding = json.dumps([0.6, 0.7, 0.8, 0.9, 1.0] * 153 + [0.5])

    cursor.execute('''
        UPDATE semantic_memory
        SET embedding = ?, updated_at = ?
        WHERE id = ?
    ''', (updated_embedding, datetime.now().isoformat(), test_id))
    conn.commit()

    # Verify update was synchronized
    cursor.execute('''
        SELECT COUNT(*) FROM vec_semantic_memory
        WHERE memory_id = ? AND embedding = ?
    ''', (test_id, updated_embedding))

    update_count = cursor.fetchone()[0]

    # Test delete synchronization
    print("  🗑️ Testing delete synchronization...")
    cursor.execute('DELETE FROM semantic_memory WHERE id = ?', (test_id,))
    conn.commit()

    # Verify delete was synchronized
    cursor.execute('''
        SELECT COUNT(*) FROM vec_semantic_memory
        WHERE memory_id = ?
    ''', (test_id,))

    delete_count = cursor.fetchone()[0]

    # Report results
    print(f"\n📊 Test Results:")
    print(f"  ✅ Insert synchronization: {vec_count > 0} (count: {vec_count})")
    print(f"  ✅ Update synchronization: {update_count > 0} (count: {update_count})")
    print(f"  ✅ Delete synchronization: {delete_count == 0} (count: {delete_count})")

    return vec_count > 0 and update_count > 0 and delete_count == 0

def generate_report(conn):
    """Generate comprehensive synchronization report."""
    print("\n📊 Generating comprehensive report...")

    cursor = conn.cursor()

    # Count records in both tables
    cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE embedding IS NOT NULL AND embedding != ""')
    main_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM vec_semantic_memory')
    vec_count = cursor.fetchone()[0]

    # Calculate synchronization percentage
    sync_percentage = (vec_count / main_count * 100) if main_count > 0 else 0

    # Count by content type
    cursor.execute('''
        SELECT sm.content_type, COUNT(*) as count
        FROM semantic_memory sm
        JOIN vec_semantic_memory vm ON sm.id = vm.memory_id
        GROUP BY sm.content_type
        ORDER BY count DESC
    ''')

    type_stats = cursor.fetchall()

    # Print report
    print(f"📈 SYNCHRONIZATION REPORT")
    print("=" * 40)
    print(f"Total records with embeddings: {main_count:,}")
    print(f"Records in vector index: {vec_count:,}")
    print(f"Synchronization rate: {sync_percentage:.1f}%")
    print(f"\n📋 Records by type in vector index:")
    for content_type, count in type_stats:
        percentage = (count / vec_count * 100) if vec_count > 0 else 0
        print(f"  {content_type}: {count:,} ({percentage:.1f}%)")

    # Estimate time to sync remaining records
    if sync_percentage < 100:
        remaining = main_count - vec_count
        print(f"\n⏱️  Remaining to sync: {remaining:,} records")
        print(f"    At current rate: ~{remaining * 0.001:.1f} seconds")

def main():
    """Main execution function."""
    print("🚀 DevStream Vector Synchronization Setup")
    print("=" * 50)

    # Database path
    db_path = Path(__file__).parent.parent / 'data' / 'devstream.db'

    try:
        # Create connection with sqlite-vec
        conn = get_db_connection_with_vec(str(db_path))
        print(f"✅ Connected to database: {db_path}")

        # Create enhanced triggers
        create_trigger(conn)

        # Verify schema
        if verify_schema(conn):
            print("✅ Schema verification passed!")
        else:
            print("❌ Schema verification failed!")
            return 1

        # Test functionality
        if test_synchronization(conn):
            print("✅ Synchronization test passed!")
        else:
            print("❌ Synchronization test failed!")
            return 1

        # Generate report
        generate_report(conn)

        conn.close()

        print("\n🎉 VECTOR SYNCHRONIZATION SETUP COMPLETE!")
        print("\n📋 What happens now:")
        print("  • All new records with embeddings will automatically sync to vec_semantic_memory")
        print("  • All embedding updates will automatically sync")
        print("  • All deletions will automatically clean up vector index")
        print("  • Existing records will sync gradually as they are updated")
        print("\n🔧 To force sync existing records, run the backfill script:")
        print("  .devstream/bin/python .claude/hooks/devstream/memory/backfill_embeddings.py")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())