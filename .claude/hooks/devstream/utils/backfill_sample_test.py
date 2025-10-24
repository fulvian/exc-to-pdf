#!/usr/bin/env python3
"""
Sample backfill test with real embedding generation.
Tests the system with a small, controlled sample.
"""

import asyncio
import time
from datetime import datetime

from embedding_backfill_manager import get_backfill_manager


async def sample_backfill_test():
    """Test backfill with small real sample."""
    print("🧪 Sample Backfill Test (Real Embeddings)")
    print("=" * 45)

    backfill_manager = get_backfill_manager()

    # Get initial progress
    print("\n1. Initial state...")
    initial_progress = await backfill_manager.get_backfill_progress()
    print(f"   📈 With embeddings: {initial_progress['records_with_embeddings']:,}")

    # Run small sample backfill (limit to 10 records)
    print("\n2. Running sample backfill (10 records)...")

    # Temporarily modify batch size for testing
    original_batch_size = backfill_manager.batch_size
    original_max_batches = backfill_manager.max_batches

    backfill_manager.batch_size = 10
    backfill_manager.max_batches = 1  # Only one batch for testing

    start_time = time.time()

    try:
        results = await backfill_manager.run_selective_backfill(
            priority_levels=["high"],
            dry_run=False
        )

        duration = (time.time() - start_time) * 1000

        print(f"   ✅ Sample completed in {duration:.1f}ms")
        print(f"   📊 Processed: {results['processed_count']} records")
        print(f"   🎯 Success: {results['success_count']} records")
        print(f"   ❌ Errors: {results['error_count']} records")
        print(f"   📈 Success rate: {results['success_rate']:.1f}%")

        # Check final progress
        print("\n3. Final state...")
        final_progress = await backfill_manager.get_backfill_progress()
        print(f"   📈 With embeddings: {final_progress['records_with_embeddings']:,}")
        print(f"   📊 New embeddings: {final_progress['records_with_embeddings'] - initial_progress['records_with_embeddings']}")

        # Performance analysis
        if results['processed_count'] > 0:
            avg_time_per_record = duration / results['processed_count']
            print(f"   ⚡ Avg time per record: {avg_time_per_record:.1f}ms")

        return {
            "success": True,
            "processed_count": results['processed_count'],
            "success_count": results['success_count'],
            "success_rate": results['success_rate'],
            "duration_ms": duration,
            "new_embeddings": final_progress['records_with_embeddings'] - initial_progress['records_with_embeddings']
        }

    except Exception as e:
        print(f"   ❌ Sample backfill failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # Restore original settings
        backfill_manager.batch_size = original_batch_size
        backfill_manager.max_batches = original_max_batches


async def main():
    """Main test function."""
    print("🚀 Starting Sample Backfill Test")
    print(f"⏰ Time: {datetime.now().isoformat()}")

    result = await sample_backfill_test()

    print("\n" + "=" * 45)
    if result.get("success", False):
        print("✅ SAMPLE BACKFILL TEST PASSED")
        print(f"📊 Processed {result['processed_count']} records")
        print(f"🎯 Success rate: {result['success_rate']:.1f}%")
        print(f"⚡ Processing time: {result['duration_ms']:.1f}ms")
        print(f"📈 New embeddings: {result['new_embeddings']}")

        # Estimate full processing time
        if result['processed_count'] > 0:
            estimated_full_time = (115560 * result['duration_ms']) / result['processed_count']
            estimated_hours = estimated_full_time / 3600000
            print(f"🕐 Estimated full backfill time: {estimated_hours:.1f} hours")
    else:
        print("❌ SAMPLE BACKFILL TEST FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())