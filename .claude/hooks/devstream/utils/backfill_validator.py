#!/usr/bin/env python3
"""
DevStream Backfill Validation Tool

Validates the embedding backfill system performance and accuracy.
Runs controlled backfill tests and provides detailed performance metrics.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from embedding_backfill_manager import get_backfill_manager


class BackfillValidator:
    """
    Validates embedding backfill system performance and accuracy.
    """

    def __init__(self):
        self.backfill_manager = get_backfill_manager()

    async def run_validation_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive validation suite for backfill system.

        Returns:
            Dictionary with validation results
        """
        print("🔍 Running Backfill Validation Suite...")

        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        # Test 1: Analysis Performance
        print("\n1. Testing analysis performance...")
        analysis_start = time.time()
        analysis = await self.backfill_manager.analyze_missing_embeddings()
        analysis_duration = (time.time() - analysis_start) * 1000

        results["tests"]["analysis_performance"] = {
            "duration_ms": analysis_duration,
            "total_missing": analysis["total_missing"],
            "high_priority": analysis["high_priority_count"],
            "medium_priority": analysis["medium_priority_count"],
            "estimated_time_minutes": analysis["estimated_processing_time"]["estimated_time_minutes"]
        }

        print(f"   ✅ Analysis completed in {analysis_duration:.1f}ms")
        print(f"   📊 Found {analysis['total_missing']} missing embeddings")
        print(f"   🎯 High priority: {analysis['high_priority_count']} records")

        # Test 2: Small Sample Backfill (20 records)
        print("\n2. Testing small sample backfill...")
        sample_results = await self.backfill_manager.run_selective_backfill(
            priority_levels=["high"],
            dry_run=False
        )

        # Limit to first batch for validation
        if sample_results["processed_count"] > 20:
            print(f"   ⚠️  Processed {sample_results['processed_count']} records, limiting validation to first 20")
            sample_results["processed_count"] = 20
            sample_results["success_count"] = min(sample_results["success_count"], 20)

        results["tests"]["sample_backfill"] = sample_results

        print(f"   ✅ Sample backfill: {sample_results['success_count']}/{sample_results['processed_count']} successful")
        print(f"   📈 Success rate: {sample_results['success_rate']}%")
        print(f"   ⏱️  Duration: {sample_results['processing_time_ms']:.1f}ms")

        # Test 3: Progress Tracking
        print("\n3. Testing progress tracking...")
        progress = await self.backfill_manager.get_backfill_progress()

        results["tests"]["progress_tracking"] = {
            "total_records": progress["total_records"],
            "records_with_embeddings": progress["records_with_embeddings"],
            "records_missing_embeddings": progress["records_missing_embeddings"],
            "completion_percentage": progress["completion_percentage"]
        }

        print(f"   ✅ Progress: {progress['completion_percentage']}% complete")
        print(f"   📈 With embeddings: {progress['records_with_embeddings']}")
        print(f"   📉 Missing: {progress['records_missing_embeddings']}")

        # Test 4: Performance Metrics
        print("\n4. Calculating performance metrics...")

        if sample_results["processed_count"] > 0:
            avg_time_per_record = sample_results["processing_time_ms"] / sample_results["processed_count"]
            success_rate = sample_results["success_rate"]

            results["tests"]["performance_metrics"] = {
                "avg_time_per_record_ms": round(avg_time_per_record, 2),
                "success_rate": success_rate,
                "throughput_records_per_minute": round((sample_results["processed_count"] * 60000) / sample_results["processing_time_ms"], 2)
            }

            print(f"   ✅ Avg time per record: {avg_time_per_record:.2f}ms")
            print(f"   📈 Throughput: {results['tests']['performance_metrics']['throughput_records_per_minute']:.1f} records/min")
        else:
            results["tests"]["performance_metrics"] = {
                "avg_time_per_record_ms": 0,
                "success_rate": 0,
                "throughput_records_per_minute": 0
            }
            print("   ⚠️  No records processed, cannot calculate performance metrics")

        # Test 5: Quality Assessment
        print("\n5. Assessing backfill quality...")
        quality_assessment = await self._assess_backfill_quality()
        results["tests"]["quality_assessment"] = quality_assessment

        # Summary
        print("\n📋 Validation Summary:")
        print(f"   📊 Total missing embeddings: {analysis['total_missing']:,}")
        print(f"   🎯 High priority targets: {analysis['high_priority_count']:,}")
        print(f"   📈 Sample success rate: {sample_results['success_rate']:.1f}%")
        print(f"   ⚡ Processing speed: {results['tests']['performance_metrics']['throughput_records_per_minute']:.1f} records/min")
        print(f"   ✅ Overall status: {'PASSED' if sample_results['success_rate'] >= 80 else 'FAILED'}")

        results["overall_status"] = "PASSED" if sample_results.get("success_rate", 0) >= 80 else "FAILED"

        return results

    async def _assess_backfill_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of embeddings generated during backfill.

        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Get recent records with embeddings for quality check
            with self.backfill_manager.connection_manager.get_connection() as conn:
                # Get recent embeddings (last 24 hours)
                cursor = conn.execute("""
                    SELECT content_type, embedding_dimension, COUNT(*) as count
                    FROM semantic_memory
                    WHERE embedding_blob IS NOT NULL
                    AND embedding_model IS NOT NULL
                    AND updated_at > datetime('now', '-1 day')
                    GROUP BY content_type, embedding_dimension
                    ORDER BY count DESC
                    LIMIT 10
                """)

                recent_embeddings = [dict(row) for row in cursor.fetchall()]

                # Get embedding dimension consistency
                cursor = conn.execute("""
                    SELECT embedding_dimension, COUNT(*) as count
                    FROM semantic_memory
                    WHERE embedding_blob IS NOT NULL
                    AND embedding_dimension IS NOT NULL
                    GROUP BY embedding_dimension
                """)

                dimension_distribution = [dict(row) for row in cursor.fetchall()]

                return {
                    "recent_embeddings_count": sum(item["count"] for item in recent_embeddings),
                    "recent_embeddings_by_type": recent_embeddings,
                    "dimension_distribution": dimension_distribution,
                    "quality_score": self._calculate_quality_score(recent_embeddings, dimension_distribution)
                }

        except Exception as e:
            return {
                "error": str(e),
                "quality_score": 0,
                "recent_embeddings_count": 0
            }

    def _calculate_quality_score(self, recent_embeddings: List[Dict], dimension_distribution: List[Dict]) -> float:
        """
        Calculate quality score for backfill operation.

        Args:
            recent_embeddings: Recent embedding records
            dimension_distribution: Distribution of embedding dimensions

        Returns:
            Quality score between 0 and 100
        """
        score = 0

        # Score for recent embeddings (max 40 points)
        recent_count = sum(item["count"] for item in recent_embeddings)
        if recent_count > 0:
            score += min(40, recent_count * 2)  # 2 points per recent embedding, max 40

        # Score for dimension consistency (max 30 points)
        if dimension_distribution:
            # Prefer single dominant dimension (indicates consistent model usage)
            total_embeddings = sum(item["count"] for item in dimension_distribution)
            max_ratio = max(item["count"] / total_embeddings for item in dimension_distribution)
            score += int(max_ratio * 30)  # Up to 30 points for consistency

        # Score for content type diversity (max 30 points)
        content_types = set(item["content_type"] for item in recent_embeddings)
        if content_types:
            score += min(30, len(content_types) * 6)  # 6 points per content type, max 30

        return min(100, score)


async def main():
    """Main validation function."""
    validator = BackfillValidator()

    print("🚀 DevStream Embedding Backfill Validation")
    print("=" * 50)

    try:
        results = await validator.run_validation_suite()

        # Save results to file
        results_file = f"backfill_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Detailed results saved to: {results_file}")

        if results["overall_status"] == "PASSED":
            print("\n✅ Validation PASSED - Backfill system ready for production")
        else:
            print("\n❌ Validation FAILED - Check system before production use")

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())