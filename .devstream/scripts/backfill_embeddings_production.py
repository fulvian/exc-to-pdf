#!/usr/bin/env python3
"""
Backfill Embeddings - Production Script

Generates embeddings for all records missing them in semantic_memory table.

Features:
- Batch processing (16 records/batch)
- Retry logic (3 attempts, exponential backoff)
- Progress logging (every 500 records)
- Checkpointing (save progress every 1000 records)
- Error recovery (skip failed records, log to file)
- keep_alive="5m" (Ollama auto-unload optimization)

Usage:
    .devstream/bin/python scripts/backfill_embeddings_production.py [--batch-size N] [--checkpoint-interval N]

Environment:
    DEVSTREAM_BACKFILL_BATCH_SIZE: Batch size (default: 16)
    DEVSTREAM_BACKFILL_CHECKPOINT: Checkpoint interval (default: 1000)
"""

import asyncio
import sqlite3
import json
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import os

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


# Configuration
BATCH_SIZE = int(os.getenv("DEVSTREAM_BACKFILL_BATCH_SIZE", 16))
CHECKPOINT_INTERVAL = int(os.getenv("DEVSTREAM_BACKFILL_CHECKPOINT", 1000))
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
PROGRESS_LOG_INTERVAL = 500  # records
KEEP_ALIVE = "5m"  # Ollama keep_alive setting

# Logging setup
log_dir = Path.home() / ".claude" / "logs" / "devstream"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "backfill_embeddings.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama embedding generation with retry logic."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "embeddinggemma:300m",
        keep_alive: str = KEEP_ALIVE
    ):
        self.base_url = base_url
        self.model = model
        self.keep_alive = keep_alive

    async def generate_embedding(self, text: str, max_retries: int = MAX_RETRIES) -> Optional[List[float]]:
        """Generate embedding with retry logic."""
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model,
                            "input": text,
                            "keep_alive": self.keep_alive
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Ollama API error: {response.status}")
                        data = await response.json()
                        return data["embeddings"][0]

            except Exception as e:
                logger.warning(
                    f"Embedding generation failed (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    return None

        return None


class BackfillProgress:
    """Track and persist backfill progress."""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = None
        self.load()

    def load(self):
        """Load progress from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.processed = data.get('processed', 0)
                    self.successful = data.get('successful', 0)
                    self.failed = data.get('failed', 0)
                    logger.info(f"Loaded checkpoint: {self.processed} processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def save(self):
        """Save progress to checkpoint file."""
        try:
            data = {
                'processed': self.processed,
                'successful': self.successful,
                'failed': self.failed,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': time.time() - self.start_time if self.start_time else 0
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def update(self, successful: bool):
        """Update progress counters."""
        self.processed += 1
        if successful:
            self.successful += 1
        else:
            self.failed += 1


async def backfill_batch(
    conn: sqlite3.Connection,
    records: List[sqlite3.Row],
    ollama: OllamaClient,
    progress: BackfillProgress
) -> int:
    """Process a batch of records."""
    successful_count = 0

    for record in records:
        record_id = record['id']
        content = record['content']

        # Generate embedding with retry
        embedding = await ollama.generate_embedding(content)

        if embedding:
            # Update database
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE semantic_memory SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding), record_id)
                )
                conn.commit()
                successful_count += 1
                progress.update(successful=True)

            except Exception as e:
                logger.error(f"Database update failed for {record_id}: {e}")
                progress.update(successful=False)
        else:
            logger.error(f"Embedding generation failed for {record_id}")
            progress.update(successful=False)

        # Progress logging
        if progress.processed % PROGRESS_LOG_INTERVAL == 0:
            elapsed = time.time() - progress.start_time
            rate = progress.processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {progress.processed} processed "
                f"({progress.successful} success, {progress.failed} failed) "
                f"| Rate: {rate:.1f} records/sec"
            )

        # Checkpoint
        if progress.processed % CHECKPOINT_INTERVAL == 0:
            progress.save()
            logger.info(f"Checkpoint saved at {progress.processed} records")

    return successful_count


async def production_backfill(db_path: str = "data/devstream.db"):
    """Execute production backfill."""

    logger.info("=" * 70)
    logger.info("🚀 PRODUCTION BACKFILL - START")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info(f"Keep alive: {KEEP_ALIVE}")
    logger.info("=" * 70)

    # Initialize
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ollama = OllamaClient()

    checkpoint_file = Path.home() / ".claude" / "state" / "backfill_checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    progress = BackfillProgress(checkpoint_file)
    progress.start_time = time.time()

    # Count total records
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total
        FROM semantic_memory
        WHERE embedding IS NULL OR embedding = ''
    """)
    total_records = cursor.fetchone()['total']

    logger.info(f"\n📊 Total records to process: {total_records}")
    logger.info(f"📊 Already processed (from checkpoint): {progress.processed}")
    logger.info(f"📊 Remaining: {total_records - progress.processed}\n")

    if total_records == 0:
        logger.info("✅ No records to process!")
        conn.close()
        return True

    # Process in batches
    offset = progress.processed
    while offset < total_records:
        # Fetch batch
        cursor.execute("""
            SELECT id, content, content_type, created_at
            FROM semantic_memory
            WHERE embedding IS NULL OR embedding = ''
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (BATCH_SIZE, offset))

        batch = cursor.fetchall()
        if not batch:
            break

        logger.info(f"\n🔧 Processing batch {offset}-{offset + len(batch)} of {total_records}")

        # Process batch
        successful = await backfill_batch(conn, batch, ollama, progress)

        offset += len(batch)

        # Estimate remaining time
        if progress.processed > 0:
            elapsed = time.time() - progress.start_time
            rate = progress.processed / elapsed
            remaining = total_records - progress.processed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            logger.info(f"⏱️ ETA: {eta_minutes:.1f} minutes")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 BACKFILL COMPLETE")
    logger.info("=" * 70)

    elapsed = time.time() - progress.start_time
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Total processed: {progress.processed}")
    logger.info(f"✅ Success: {progress.successful}")
    logger.info(f"❌ Failed: {progress.failed}")
    logger.info(f"Success rate: {(progress.successful / progress.processed * 100):.1f}%")
    logger.info("=" * 70)

    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("✅ Checkpoint file removed")

    conn.close()

    return progress.successful >= (progress.processed * 0.95)  # 95% success threshold


if __name__ == "__main__":
    success = asyncio.run(production_backfill())
    sys.exit(0 if success else 1)
