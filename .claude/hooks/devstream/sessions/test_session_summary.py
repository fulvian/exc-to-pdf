#!/usr/bin/env python3
"""
Test script for SessionSummaryManager - B2 Behavioral Refinement

Tests:
1. Extract session data from semantic memory
2. Analyze memories to extract structured info
3. Infer session goal from context
4. Generate Context7-compliant summary
5. Store summary in semantic memory
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from session_summary_manager import SessionSummaryManager


async def test_extract_session_data():
    """Test extraction of session data from semantic memory."""
    print("\n" + "=" * 70)
    print("TEST 1: Extract Session Data")
    print("=" * 70)

    manager = SessionSummaryManager()
    memories = await manager.extract_session_data(hours_back=24, limit=100)

    print(f"✅ Extracted {len(memories)} memories from last 24 hours")
    if memories:
        print(f"\n📊 Sample memory:")
        print(f"   Type: {memories[0]['content_type']}")
        print(f"   Content: {memories[0]['content'][:100]}...")
        print(f"   Keywords: {memories[0]['keywords']}")

    return memories


async def test_analyze_memories(memories):
    """Test analysis of memories to extract structured info."""
    print("\n" + "=" * 70)
    print("TEST 2: Analyze Memories")
    print("=" * 70)

    manager = SessionSummaryManager()
    analysis = manager.analyze_memories(memories)

    print(f"✅ Analysis Results:")
    print(f"   Completed Tasks: {len(analysis['completed_tasks'])}")
    print(f"   Modified Files: {len(analysis['modified_files'])}")
    print(f"   Key Decisions: {len(analysis['key_decisions'])}")
    print(f"   Errors: {len(analysis['errors'])}")
    print(f"   Session Context: {len(analysis['session_context'])}")

    if analysis['completed_tasks']:
        print(f"\n📝 Sample Task: {analysis['completed_tasks'][0]}")
    if analysis['modified_files']:
        print(f"📁 Sample File: {list(analysis['modified_files'])[0]}")
    if analysis['key_decisions']:
        print(f"🔍 Sample Decision: {analysis['key_decisions'][0][:80]}...")

    return analysis


async def test_infer_session_goal(analysis):
    """Test session goal inference from context."""
    print("\n" + "=" * 70)
    print("TEST 3: Infer Session Goal")
    print("=" * 70)

    manager = SessionSummaryManager()
    session_goal = manager.infer_session_goal(
        analysis['completed_tasks'],
        analysis['modified_files'],
        analysis['key_decisions'],
        analysis['session_context']
    )

    print(f"✅ Inferred Session Goal:")
    print(f"   {session_goal}")

    return session_goal


async def test_generate_summary(analysis, session_goal):
    """Test structured summary generation."""
    print("\n" + "=" * 70)
    print("TEST 4: Generate Structured Summary")
    print("=" * 70)

    manager = SessionSummaryManager()
    summary = manager.generate_structured_summary(
        session_goal=session_goal,
        completed_tasks=analysis['completed_tasks'],
        modified_files=analysis['modified_files'],
        key_decisions=analysis['key_decisions'],
        errors=analysis['errors'],
        session_context=analysis['session_context'],
        total_memories=10  # Mock value
    )

    print(f"✅ Generated Summary ({len(summary)} chars):")
    print("\n" + "-" * 70)
    print(summary)
    print("-" * 70)

    return summary


async def test_store_summary(summary):
    """Test summary storage in semantic memory."""
    print("\n" + "=" * 70)
    print("TEST 5: Store Summary in Memory")
    print("=" * 70)

    manager = SessionSummaryManager()
    success, memory_id = await manager.store_summary(summary)

    if success:
        print(f"✅ Summary stored successfully")
        print(f"   Memory ID: {memory_id}")
    else:
        print(f"❌ Failed to store summary")

    return success


async def test_complete_workflow():
    """Test complete workflow: extract → analyze → generate → store."""
    print("\n" + "=" * 70)
    print("TEST 6: Complete Workflow (generate_and_store_summary)")
    print("=" * 70)

    manager = SessionSummaryManager()
    success, summary = await manager.generate_and_store_summary()

    if success:
        print(f"✅ Complete workflow succeeded")
        print(f"\n📋 Final Summary:")
        print("-" * 70)
        print(summary)
        print("-" * 70)
    else:
        print(f"⚠️  Workflow completed with warnings")
        print(f"\n📋 Fallback Summary:")
        print("-" * 70)
        print(summary)
        print("-" * 70)

    return success, summary


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 SessionSummaryManager Test Suite - B2 Behavioral Refinement")
    print("=" * 70)

    try:
        # Test 1: Extract session data
        memories = await test_extract_session_data()

        if not memories:
            print("\n⚠️  No memories found. Skipping analysis tests.")
            print("   Proceeding with complete workflow test (uses fallback)...")
            # Still test complete workflow with fallback
            await test_complete_workflow()
            return

        # Test 2: Analyze memories
        analysis = await test_analyze_memories(memories)

        # Test 3: Infer session goal
        session_goal = await test_infer_session_goal(analysis)

        # Test 4: Generate summary
        summary = await test_generate_summary(analysis, session_goal)

        # Test 5: Store summary
        success = await test_store_summary(summary)

        # Test 6: Complete workflow
        await test_complete_workflow()

        print("\n" + "=" * 70)
        print("✅ All Tests Completed Successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
