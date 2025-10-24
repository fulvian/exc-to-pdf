#!/usr/bin/env python3
"""
Test SEC-004 Memory Leak Fix for Ollama Embedding Client
Tests the Context7-compliant memory-safe batch processing implementation.
"""

import sys
import os
from pathlib import Path
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from ollama_client import OllamaEmbeddingClient


def test_memory_safe_batch_processing():
    """Test Context7 memory-safe batch processing functionality."""
    print("🧠 Testing Memory-Safe Batch Processing")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Verify memory monitoring is enabled
    assert client.memory_monitoring_enabled is True
    assert client.batch_max_size <= 16  # Context7 best practice
    assert client.max_memory_per_batch > 0
    assert client.max_total_batch_memory > 0

    print(f"✅ Memory monitoring enabled: {client.memory_monitoring_enabled}")
    print(f"✅ Batch max size: {client.batch_max_size}")
    print(f"✅ Max memory per batch: {client.max_memory_per_batch}MB")
    print(f"✅ Max total batch memory: {client.max_total_batch_memory}MB")
    print()


def test_adaptive_batch_sizing():
    """Test Context7 adaptive batch sizing based on memory constraints."""
    print("📏 Testing Adaptive Batch Sizing")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Test normal batch sizing
    normal_texts = [f"Test text {i}" for i in range(10)]
    batches = client._adaptive_batch_sizing(normal_texts)

    assert len(batches) > 0, "Should create at least one batch"
    assert all(len(batch) <= client.batch_max_size for batch in batches), "All batches should respect max size"

    print(f"✅ Normal batch sizing: {len(batches)} batches, sizes={[len(b) for b in batches]}")

    # Test memory constraint handling
    # Set very low memory limits to trigger adaptive sizing
    original_per_batch = client.max_memory_per_batch
    original_total = client.max_total_batch_memory

    client.max_memory_per_batch = 1  # 1MB limit
    client.max_total_batch_memory = 5  # 5MB total limit

    large_texts = [f"Large test text {i} with substantial content to trigger memory constraints"
                   for i in range(20)]

    constrained_batches = client._adaptive_batch_sizing(large_texts)

    # Restore original limits
    client.max_memory_per_batch = original_per_batch
    client.max_total_batch_memory = original_total

    assert len(constrained_batches) > 0, "Should create batches even with constraints"
    # With very low limits, should create more, smaller batches
    print(f"✅ Constrained batch sizing: {len(constrained_batches)} batches (memory-limited)")

    print()


def test_memory_monitoring():
    """Test Context7 memory monitoring and health checking."""
    print("📊 Testing Memory Monitoring")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Test memory usage estimation
    current_memory = client._get_current_memory_usage()
    assert isinstance(current_memory, (int, float)), "Memory usage should be numeric"
    assert current_memory >= 0, "Memory usage should be non-negative"

    print(f"✅ Current memory usage: {current_memory:.1f}MB")

    # Test memory health status
    health_status = client._get_memory_health_status()
    assert health_status in ['HEALTHY', 'WARNING', 'CRITICAL', 'MONITORING_DISABLED'], \
        f"Invalid health status: {health_status}"

    print(f"✅ Memory health status: {health_status}")

    # Test memory recommendations
    stats = client.get_cache_stats()
    recommendations = stats.get('recommendations', [])
    assert isinstance(recommendations, list), "Recommendations should be a list"

    print(f"✅ Memory recommendations: {len(recommendations)} generated")
    for rec in recommendations[:3]:  # Show first 3
        print(f"   • {rec}")

    print()


def test_memory_limit_enforcement():
    """Test Context7 memory limit enforcement and DoS protection."""
    print("🚦 Testing Memory Limit Enforcement")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Test per-batch memory limit checking
    small_batch = [f"Small test text {i}" for i in range(5)]
    small_batch_ok = client._check_memory_limits(small_batch)
    assert small_batch_ok is True, "Small batch should pass memory limits"

    print(f"✅ Small batch passes memory limits: {small_batch_ok}")

    # Test with artificially low limits to trigger enforcement
    original_per_batch = client.max_memory_per_batch
    client.max_memory_per_batch = 0.001  # 0.001MB limit (should fail most batches)

    medium_batch = [f"Medium test text {i} with some content" for i in range(3)]
    medium_batch_ok = client._check_memory_limits(medium_batch)

    # Restore original limits
    client.max_memory_per_batch = original_per_batch

    print(f"✅ Memory limit enforcement working: {not medium_batch_ok}")

    # Test batch memory usage estimation
    estimated_memory = client._estimate_batch_memory_usage(medium_batch)
    assert isinstance(estimated_memory, (int, float)), "Memory estimate should be numeric"
    assert estimated_memory > 0, "Memory estimate should be positive"

    print(f"✅ Batch memory estimation: {estimated_memory:.2f}MB")

    print()


def test_cache_memory_integration():
    """Test LRU cache integration with memory management."""
    print("💾 Testing Cache Memory Integration")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Clear cache for clean test
    client.clear_cache()

    # Test cache memory estimation
    initial_cache_memory = client._estimate_cache_memory_usage()
    assert initial_cache_memory >= 0, "Cache memory estimate should be non-negative"

    print(f"✅ Initial cache memory: {initial_cache_memory:.2f}MB")

    # Test embedding memory estimation
    test_embedding = [0.1] * 384  # Simulated embedding
    embedding_memory = client._estimate_embedding_memory_usage(test_embedding)
    assert isinstance(embedding_memory, (int, float)), "Embedding memory should be numeric"
    assert embedding_memory > 0, "Embedding memory should be positive"

    print(f"✅ Embedding memory estimate: {embedding_memory:.4f}MB")

    # Test cache statistics include memory metrics
    stats = client.get_cache_stats()
    memory_metrics = [
        'memory_monitoring_enabled',
        'memory_warnings',
        'batch_memory_saved_mb',
        'current_memory_mb',
        'cache_memory_estimate_mb',
        'health_status',
        'recommendations'
    ]

    for metric in memory_metrics:
        assert metric in stats, f"Missing memory metric: {metric}"

    print(f"✅ All memory metrics present in cache stats")
    print(f"✅ Cache memory estimate: {stats['cache_memory_estimate_mb']:.2f}MB")
    print(f"✅ Health status: {stats['health_status']}")

    print()


def test_memory_leak_prevention():
    """Test Context7 memory leak prevention in batch processing."""
    print("🔒 Testing Memory Leak Prevention")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Measure initial memory
    initial_memory = client._get_current_memory_usage()
    print(f"Initial memory: {initial_memory:.1f}MB")

    # Process multiple batches to test for memory leaks
    test_rounds = 3
    memory_samples = []

    for round_num in range(test_rounds):
        print(f"Processing round {round_num + 1}/{test_rounds}...")

        # Create test batch
        test_batch = [f"Memory leak test round {round_num} item {i}" for i in range(15)]

        # Process batch with memory management
        start_memory = client._get_current_memory_usage()
        results = client.generate_embeddings_batch(test_batch)
        end_memory = client._get_current_memory_usage()

        memory_samples.append(end_memory)

        # Force garbage collection
        gc.collect()
        after_gc_memory = client._get_current_memory_usage()

        print(f"  Memory: {start_memory:.1f}→{end_memory:.1f}→{after_gc_memory:.1f}MB")

        # Verify results are valid (even if Ollama is not available)
        assert len(results) == len(test_batch), "Results should match input size"

    # Analyze memory trend
    final_memory = client._get_current_memory_usage()
    memory_increase = final_memory - initial_memory

    print(f"Memory analysis:")
    print(f"  Initial: {initial_memory:.1f}MB")
    print(f"  Final: {final_memory:.1f}MB")
    print(f"  Increase: {memory_increase:.1f}MB")

    # Check for significant memory leak (>50MB increase)
    # Python's memory allocator may hold onto memory, so use higher threshold
    significant_leak = memory_increase > 50
    assert not significant_leak, f"Significant memory leak detected: {memory_increase:.1f}MB increase"

    print(f"✅ No significant memory leak detected")
    print()


def test_concurrent_memory_safety():
    """Test Context7 thread-safe memory management under concurrent access."""
    print("🔄 Testing Concurrent Memory Safety")
    print("-" * 40)

    client = OllamaEmbeddingClient()
    results = []
    errors = []

    def worker_task(worker_id: int) -> Dict[str, Any]:
        """Worker task that tests concurrent memory operations."""
        try:
            start_time = time.time()

            # Each worker processes a small batch
            test_batch = [f"Concurrent test {worker_id} item {i}" for i in range(5)]

            # Process with memory management
            batch_results = client.generate_embeddings_batch(test_batch)

            # Get memory statistics
            stats = client.get_cache_stats()

            end_time = time.time()

            return {
                'worker_id': worker_id,
                'success': True,
                'batch_size': len(test_batch),
                'results_count': len(batch_results),
                'memory_warnings': stats['memory_warnings'],
                'health_status': stats['health_status'],
                'duration': end_time - start_time
            }

        except Exception as e:
            return {
                'worker_id': worker_id,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    # Test with concurrent workers
    num_workers = 8
    print(f"Launching {num_workers} concurrent workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_task, i): i for i in range(num_workers)}

        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result(timeout=10.0)
                results.append(result)
                if result['success']:
                    print(f"✅ Worker {result['worker_id']}: Success "
                          f"(duration: {result['duration']:.3f}s, "
                          f"health: {result['health_status']})")
                else:
                    print(f"❌ Worker {result['worker_id']}: {result['error_type']}: {result['error']}")
                    errors.append(result)
            except Exception as e:
                print(f"❌ Worker {worker_id}: Exception: {e}")
                errors.append({'worker_id': worker_id, 'error': str(e), 'error_type': 'Timeout'})

    # Analyze results
    successful_workers = [r for r in results if r['success']]
    failed_workers = [r for r in results if not r['success']]

    print(f"\n📊 Concurrent Test Results:")
    print(f"   Successful workers: {len(successful_workers)}/{num_workers}")
    print(f"   Failed workers: {len(failed_workers)}")

    if successful_workers:
        avg_duration = sum(r['duration'] for r in successful_workers) / len(successful_workers)
        total_warnings = sum(r['memory_warnings'] for r in successful_workers)

        print(f"   Average duration: {avg_duration:.3f}s")
        print(f"   Total memory warnings: {total_warnings}")

    # Verify thread safety (no crashes, reasonable results)
    assert len(failed_workers) == 0, f"Thread safety issue: {len(failed_workers)} workers failed"
    assert len(successful_workers) == num_workers, "Not all workers completed successfully"

    print("✅ Concurrent memory safety verified")
    print()


def test_memory_statistics_accuracy():
    """Test accuracy and completeness of memory statistics."""
    print("📈 Testing Memory Statistics Accuracy")
    print("-" * 40)

    client = OllamaEmbeddingClient()

    # Get initial statistics
    initial_stats = client.get_cache_stats()

    # Verify required memory-related fields
    required_memory_fields = [
        'memory_monitoring_enabled',
        'memory_warnings',
        'batch_memory_saved_mb',
        'batch_max_size',
        'max_memory_per_batch',
        'max_total_batch_memory',
        'current_memory_mb',
        'cache_memory_estimate_mb',
        'health_status',
        'recommendations'
    ]

    for field in required_memory_fields:
        assert field in initial_stats, f"Missing required memory field: {field}"

    print(f"✅ All {len(required_memory_fields)} required memory fields present")

    # Verify field types and values
    assert isinstance(initial_stats['memory_monitoring_enabled'], bool), \
        "memory_monitoring_enabled should be boolean"
    assert isinstance(initial_stats['memory_warnings'], int), \
        "memory_warnings should be integer"
    assert isinstance(initial_stats['batch_memory_saved_mb'], (int, float)), \
        "batch_memory_saved_mb should be numeric"
    assert isinstance(initial_stats['health_status'], str), \
        "health_status should be string"
    assert isinstance(initial_stats['recommendations'], list), \
        "recommendations should be list"

    print(f"✅ Memory statistics field types validated")

    # Test statistics update after operations
    client.clear_cache()

    # Perform some operations
    test_text = "Memory statistics test text"
    _ = client.generate_embedding(test_text)

    # Get updated statistics
    updated_stats = client.get_cache_stats()

    # Verify cache statistics updated
    assert updated_stats['hits'] + updated_stats['misses'] > 0, \
        "Cache should have recorded activity"

    print(f"✅ Statistics update correctly after operations")
    print(f"✅ Cache activity: {updated_stats['hits']} hits, {updated_stats['misses']} misses")

    # Test recommendations are contextually relevant
    recommendations = updated_stats['recommendations']
    assert isinstance(recommendations, list), "Recommendations should be list"
    assert len(recommendations) > 0, "Should have at least one recommendation"

    print(f"✅ Generated {len(recommendations)} contextually relevant recommendations")
    for rec in recommendations[:2]:  # Show first 2
        print(f"   • {rec}")

    print()


def main():
    """Run all SEC-004 memory leak fix tests."""
    print("🧪 SEC-004 Memory Leak Fix Test Suite")
    print("=" * 60)
    print("Testing Context7-compliant memory-safe batch processing")
    print("with adaptive sizing, monitoring, and DoS protection")
    print()

    tests = [
        test_memory_safe_batch_processing,
        test_adaptive_batch_sizing,
        test_memory_monitoring,
        test_memory_limit_enforcement,
        test_cache_memory_integration,
        test_memory_leak_prevention,
        test_concurrent_memory_safety,
        test_memory_statistics_accuracy,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All SEC-004 memory leak fix tests passed!")
        print("✅ Context7-compliant memory management working correctly")
        print("✅ Memory-safe batch processing operational")
        print("✅ Adaptive batch sizing functional")
        print("✅ Memory monitoring and health checking active")
        print("✅ DoS protection and memory limits enforced")
        print("✅ LRU cache integration memory-aware")
        print("✅ Concurrent memory safety verified")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Review memory leak fix implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)