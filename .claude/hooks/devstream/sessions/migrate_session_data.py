#!/usr/bin/env .devstream/bin/python
"""
Migration script to transfer existing session data to new simplified schema.
Reads from work_sessions_backup and creates proper records in sessions table.
"""

import sqlite3
import json
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from .claude.hooks.devstream.sessions.session_manager import SessionManager
    from .claude.hooks.devstream.sessions.session_tracker import SessionTracker
    from .claude.hooks.devstream.sessions.session_summary import SessionSummary
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure session components are properly installed")
    sys.exit(1)

async def migrate_session_data() -> None:
    """
    Migrate existing session data from work_sessions_backup to new sessions schema.
    """

    # Database path
    db_path = os.environ.get('DEVSTREAM_DB_PATH', 'data/devstream.db')

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    print("🔄 Starting session data migration...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if backup table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='work_sessions_backup'
        """)
        if not cursor.fetchone():
            print("❌ Backup table 'work_sessions_backup' not found")
            return

        # Get all existing sessions
        cursor.execute("SELECT * FROM work_sessions_backup ORDER BY started_at")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if not rows:
            print("ℹ️ No sessions found to migrate")
            return

        print(f"📊 Found {len(rows)} sessions to migrate")

        # Initialize session components
        session_manager = SessionManager.get_instance()
        session_tracker = SessionTracker.get_instance()
        session_summary = SessionSummary.get_instance()

        migrated_count = 0
        error_count = 0

        for row in rows:
            try:
                # Convert row to dict
                old_data = dict(zip(columns, row))
                session_id = old_data['id']

                print(f"📦 Migrating session: {session_id}")

                # Prepare migration data
                migration_data = await prepare_migration_data(old_data)

                # Create session using new manager
                await session_manager.create_session(
                    session_id=session_id,
                    session_name=migration_data['session_name'],
                    metadata=migration_data['metadata']
                )

                # Update status if not active
                if migration_data['status'] != 'active':
                    await session_manager.update_session_status(
                        session_id, migration_data['status']
                    )

                # Add progress tracking if there are completed tasks
                if migration_data.get('completed_tasks'):
                    await session_tracker.add_progress_update(
                        session_id=session_id,
                        update_type="migration",
                        details=f"Migrated with {len(migration_data['completed_tasks'])} completed tasks"
                    )

                # Store context summary if exists
                if migration_data.get('context_summary'):
                    await session_summary.store_context_summary(
                        session_id=session_id,
                        context_summary=migration_data['context_summary'],
                        token_count=migration_data.get('tokens_used', 0)
                    )

                migrated_count += 1
                print(f"✅ Migrated session: {session_id}")

            except Exception as e:
                error_count += 1
                print(f"❌ Error migrating session {session_id}: {e}")
                continue

        print(f"\n📈 Migration Summary:")
        print(f"   ✅ Successfully migrated: {migrated_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   📊 Total processed: {len(rows)}")

        if error_count == 0:
            print("\n🎉 Migration completed successfully!")

            # Optional: Clean up backup table after successful migration
            response = input("\n🗑️  Remove backup table 'work_sessions_backup'? (y/N): ")
            if response.lower() == 'y':
                cursor.execute("DROP TABLE work_sessions_backup")
                conn.commit()
                print("✅ Backup table removed")
            else:
                print("ℹ️ Backup table preserved")
        else:
            print(f"\n⚠️ Migration completed with {error_count} errors")
            print("💡 Check logs above and consider manual fixes")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

async def prepare_migration_data(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old session data format to new schema format.
    """

    # Parse JSON fields
    active_tasks = []
    completed_tasks = []

    try:
        if old_data.get('active_tasks'):
            active_tasks = json.loads(old_data['active_tasks'])
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        if old_data.get('completed_tasks'):
            completed_tasks = json.loads(old_data['completed_tasks'])
    except (json.JSONDecodeError, TypeError):
        pass

    # Prepare metadata for new schema
    metadata = {
        "migration": {
            "migrated_at": datetime.utcnow().isoformat(),
            "original_data": {
                "context_window_size": old_data.get('context_window_size'),
                "active_files": old_data.get('active_files'),
                "plan_id": old_data.get('plan_id'),
                "user_id": old_data.get('user_id')
            }
        },
        "task_counts": {
            "active": len(active_tasks),
            "completed": len(completed_tasks)
        }
    }

    # Determine status
    status = old_data.get('status', 'unknown')
    if status == 'active' and old_data.get('ended_at'):
        status = 'completed'
    elif status == 'active' and not old_data.get('ended_at'):
        status = 'paused'  # Old active sessions are now paused

    return {
        'session_name': old_data.get('session_name', f"Session {old_data['id'][:8]}"),
        'status': status,
        'context_summary': old_data.get('context_summary'),
        'tokens_used': old_data.get('tokens_used', 0),
        'completed_tasks': completed_tasks,
        'metadata': metadata
    }

async def main():
    """Main migration function."""
    try:
        await migrate_session_data()
    except KeyboardInterrupt:
        print("\n⚠️ Migration interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())