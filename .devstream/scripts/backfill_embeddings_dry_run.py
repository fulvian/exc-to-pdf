#!/usr/bin/env python3
"""
Backfill Embeddings - Dry Run Script

Tests backfill logic on 5 records without committing to database.

Validates:
- Record selection (NULL embedding)
- Embedding generation (Ollama)
- Database update simulation (dry-run)

Usage:
    .devstream/bin/python scripts/backfill_embeddings_dry_run.py
"""

import asyncio
import sqlite3
import json
import sys
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import aiohttp
except ImportError:
    print("❌ aiohttp not installed. Installing...")
    import subprocess
    subprocess.run([".devstream/bin/python", "-m", "pip", "install", "aiohttp"], check=True)
    import aiohttp


class OllamaClient:
    """Client for Ollama embedding generation."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "embeddinggemma:300m"):
        self.base_url = base_url
        self.model = model

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                data = await response.json()
                return data["embeddings"][0]


async def dry_run_backfill(db_path: str = "data/devstream.db", limit: int = 5):
    """Execute dry-run backfill on N records."""

    print("\n🔍 DRY-RUN BACKFILL TEST")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Test limit: {limit} records")
    print(f"Mode: DRY-RUN (no commit)")
    print("=" * 60)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Step 1: Select records without embeddings
    print("\n📊 Step 1: Selecting records without embeddings...")
    cursor.execute("""
        SELECT id, content, content_type, created_at
        FROM semantic_memory
        WHERE embedding IS NULL OR embedding = ''
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    records = cursor.fetchall()
    print(f"✅ Found {len(records)} records")

    if len(records) == 0:
        print("⚠️ No records without embeddings found!")
        conn.close()
        return

    # Step 2: Generate embeddings
    print("\n🔧 Step 2: Generating embeddings...")
    ollama = OllamaClient()

    results = []
    for i, record in enumerate(records, 1):
        record_id = record['id']
        content = record['content']
        content_type = record['content_type']
        created_at = record['created_at']

        print(f"\n  [{i}/{len(records)}] Record: {record_id}")
        print(f"    Type: {content_type}")
        print(f"    Content: {content[:100]}...")
        print(f"    Created: {created_at}")

        try:
            embedding = await ollama.generate_embedding(content)
            print(f"    ✅ Embedding generated: {len(embedding)} dimensions")

            results.append({
                "id": record_id,
                "content_type": content_type,
                "embedding_dim": len(embedding),
                "status": "success"
            })

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "id": record_id,
                "content_type": content_type,
                "error": str(e),
                "status": "failed"
            })

    # Step 3: Summary (dry-run - no database update)
    print("\n" + "=" * 60)
    print("📋 DRY-RUN SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    print(f"\nTotal records: {len(records)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"Success rate: {(success_count / len(records) * 100):.1f}%")

    print("\n🔍 Detailed Results:")
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"  {status_icon} {r['id'][:8]}... ({r['content_type']})")
        if r["status"] == "success":
            print(f"      Dimensions: {r['embedding_dim']}")
        else:
            print(f"      Error: {r.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("⚠️ DRY-RUN MODE: No database changes committed")
    print("=" * 60)

    # Close connection without committing
    conn.close()

    return success_count == len(records)


if __name__ == "__main__":
    success = asyncio.run(dry_run_backfill())
    sys.exit(0 if success else 1)
