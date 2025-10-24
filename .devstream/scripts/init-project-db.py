#!/usr/bin/env python3
"""
DevStream Project Database Initialization Script

Usage:
    python scripts/init-project-db.py [project_path]

This script initializes a DevStream database for a project with all required tables.
It's useful when setting up a new project or when the database schema is incomplete.
"""

import sys
import os
import sqlite3
import argparse
from pathlib import Path

def create_sessions_table(db_path: str) -> bool:
    """Create sessions table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                tokens_used INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                started_at TEXT NOT NULL,
                ended_at TEXT,
                files_modified INTEGER DEFAULT 0,
                tasks_completed INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at)')

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating sessions table: {e}")
        return False

def create_implementation_plans_table(db_path: str) -> bool:
    """Create implementation_plans table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create implementation_plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS implementation_plans (
                id TEXT PRIMARY KEY,
                task_id TEXT,
                model_type TEXT NOT NULL,
                plan_title TEXT NOT NULL,
                plan_content TEXT,
                plan_status TEXT DEFAULT 'draft',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                metadata TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_implementation_plans_task_id ON implementation_plans(task_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_implementation_plans_status ON implementation_plans(plan_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_implementation_plans_model_type ON implementation_plans(model_type)')

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating implementation_plans table: {e}")
        return False

def check_database_schema(db_path: str) -> dict:
    """Check what tables exist in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()
        return {
            'exists': True,
            'tables': sorted(tables),
            'missing_sessions': 'sessions' not in tables,
            'missing_implementation_plans': 'implementation_plans' not in tables
        }

    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'tables': [],
            'missing_sessions': True,
            'missing_implementation_plans': True
        }

def initialize_database(db_path: str) -> bool:
    """Initialize complete database schema for DevStream."""
    print(f"🔧 Initializing DevStream database: {db_path}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Check current schema
    schema_info = check_database_schema(db_path)

    if not schema_info['exists']:
        print(f"❌ Database does not exist: {schema_info.get('error', 'Unknown error')}")
        return False

    print(f"📊 Current tables: {', '.join(schema_info['tables'])}")

    success = True

    # Create missing tables
    if schema_info['missing_sessions']:
        print("🔨 Creating sessions table...")
        if not create_sessions_table(db_path):
            success = False
        else:
            print("✅ Sessions table created")

    if schema_info['missing_implementation_plans']:
        print("🔨 Creating implementation_plans table...")
        if not create_implementation_plans_table(db_path):
            success = False
        else:
            print("✅ Implementation plans table created")

    # Verify final schema
    final_schema = check_database_schema(db_path)
    expected_tables = ['memory', 'semantic_memory', 'tasks', 'sessions', 'implementation_plans']

    missing_critical = [t for t in expected_tables if t not in final_schema['tables']]

    if missing_critical:
        print(f"❌ Missing critical tables: {', '.join(missing_critical)}")
        success = False
    else:
        print("✅ All critical tables present")
        print(f"📋 Final schema: {', '.join(final_schema['tables'])}")

    return success

def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description='Initialize DevStream project database')
    parser.add_argument('project_path', nargs='?',
                       help='Project root path (default: current directory)')
    parser.add_argument('--db-path',
                       help='Custom database path (default: project_path/data/devstream.db)')

    args = parser.parse_args()

    # Determine project path
    if args.project_path:
        project_path = Path(args.project_path).resolve()
    else:
        project_path = Path.cwd()

    print(f"📁 Project path: {project_path}")

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = project_path / 'data' / 'devstream.db'

    print(f"🗄️  Database path: {db_path}")

    # Initialize database
    success = initialize_database(str(db_path))

    if success:
        print("\n🎉 Database initialization completed successfully!")
        print("💡 You can now start using DevStream with this project.")
    else:
        print("\n❌ Database initialization failed!")
        print("💡 Check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()