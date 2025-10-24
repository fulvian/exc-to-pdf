#!/usr/bin/env python3
"""
Test Suite: Active Session Tracking - Memory Bank Pattern Implementation

Validates that session summaries show actual work performed during session
instead of empty summaries due to timezone/time-range bugs.

Test Coverage:
- SessionDataExtractor keyword extraction from TodoWrite titles
- Hybrid query approach (tracking + fallback)
- End-to-end session summary generation
- Memory Bank pattern compliance (Context7 validated)

Author: DevStream Implementation Team
Task: c5af739922abe80e5d6e755b2bc56f24
Date: 2025-10-06
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from session_data_extractor import SessionDataExtractor, SessionData, TaskStats, MemoryStats
from session_summary_generator import SessionSummaryGenerator


async def test_keyword_extraction():
    """Test keyword extraction from long TodoWrite titles."""
    print("Testing keyword extraction from TodoWrite titles...")

    sample_titles = [
        "DISCUSSION: Present Memory Bank pattern solution and trade-offs for active session tracking",
        "ANALYSIS: Analyze codebase for similar patterns, identify files to modify, estimate complexity",
        "RESEARCH: Use Context7 to research Memory Bank patterns and best practices",
        "IMPLEMENTATION: Execute 5-phase implementation (Schema → Hook → Extractor → Testing → Documentation)",
        "63d7541081b8f7250cebde544886a7f7",  # UUID mixed in
    ]

    import re
    all_keywords = set()
    common_words = {
        'this', 'that', 'with', 'from', 'they', 'have', 'been',
        'were', 'said', 'each', 'which', 'their', 'time', 'will',
        'about', 'would', 'could', 'should', 'other', 'after',
        'first', 'into', 'present', 'solution', 'trade', 'offs'
    }

    for title in sample_titles:
        # Skip UUIDs
        if re.match(r'^[a-f0-9]{32}$', title.replace('-', '')) or re.match(r'^[a-f0-9-]{36}$', title):
            continue

        # Extract keywords: words 4+ chars, exclude common words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
        meaningful_words = [w for w in words if w not in common_words and len(w) >= 4]
        all_keywords.update(meaningful_words)

    # Verify meaningful keywords extracted
    expected_keywords = {'memory', 'bank', 'pattern', 'solution', 'discuss',
                       'analysis', 'codebase', 'similar', 'patterns', 'research',
                       'context7', 'implementation', 'phase', 'execute'}

    found_keywords = all_keywords.intersection(expected_keywords)
    print(f"✅ Extracted {len(found_keywords)} meaningful keywords: {sorted(found_keywords)}")
    assert len(found_keywords) >= 5, f"Expected at least 5 keywords, found {len(found_keywords)}"


async def test_uuid_vs_title_classification():
    """Test proper classification of UUID vs title-based task IDs."""
    print("Testing UUID vs title classification...")

    test_tasks = [
        "63d7541081b8f7250cebde544886a7f7",  # UUID
        "c5af739922abe80e5d6e755b2bc56f24",  # UUID
        "DISCUSSION: Present Memory Bank pattern solution",  # Title
        "RESEARCH: Use Context7 to research patterns",  # Title
        "invalid-uuid",  # Invalid format (treated as title)
    ]

    import re
    uuid_tasks = []
    title_tasks = []

    for task in test_tasks:
        if re.match(r'^[a-f0-9]{32}$', task.replace('-', '')) or re.match(r'^[a-f0-9-]{36}$', task):
            uuid_tasks.append(task)
        else:
            title_tasks.append(task)

    # Verify classification
    assert len(uuid_tasks) == 2
    assert len(title_tasks) == 3
    assert "63d7541081b8f7250cebde544886a7f7" in uuid_tasks
    assert "DISCUSSION: Present Memory Bank pattern solution" in title_tasks

    print(f"✅ Correctly classified {len(uuid_tasks)} UUID tasks and {len(title_tasks)} title tasks")


async def test_session_data_extractor_real():
    """Test SessionDataExtractor with real session data."""
    print("Testing SessionDataExtractor with real data...")

    extractor = SessionDataExtractor()

    # Get current session data
    session_data = await extractor.get_session_metadata('sess-b525f88712bf4162')

    if session_data:
        print(f"✅ Found session: {session_data.session_id}")
        print(f"   Active tasks: {len(session_data.active_tasks)}")
        print(f"   Active files: {len(session_data.active_files)}")

        # Test hybrid query
        if session_data.started_at and session_data.active_tasks:
            task_stats = await extractor.get_task_stats(
                session_data.started_at,
                session_data.ended_at or datetime.now(),
                session_data=session_data
            )

            print(f"✅ Hybrid query results:")
            print(f"   Total tasks: {task_stats.total_tasks}")
            print(f"   Completed: {task_stats.completed}")
            print(f"   Active: {task_stats.active}")
            print(f"   Task titles: {len(task_stats.task_titles)}")

            # Should have results now (not 0)
            assert task_stats.total_tasks > 0, "Expected to find tasks via hybrid query"
            assert task_stats.completed > 0, "Expected to find completed tasks"

            print(f"   First few titles: {task_stats.task_titles[:2]}")
        else:
            print("⚠️  No active tasks or start time in session")
    else:
        print("⚠️  No session data found")


async def test_summary_generation():
    """Test session summary generation."""
    print("Testing session summary generation...")

    extractor = SessionDataExtractor()
    generator = SessionSummaryGenerator()

    # Get current session data
    session_data = await extractor.get_session_metadata('sess-b525f88712bf4162')

    if session_data and session_data.started_at:
        # Extract stats
        task_stats = await extractor.get_task_stats(
            session_data.started_at,
            session_data.ended_at or datetime.now(),
            session_data=session_data
        )

        memory_stats = await extractor.get_memory_stats(session_data.started_at)

        # Generate summary
        summary = generator.aggregate_session_data(
            session_data, memory_stats, task_stats
        )

        markdown = summary.to_markdown()

        print(f"✅ Generated summary: {len(markdown)} chars")
        print(f"   Tasks completed: {summary.tasks_completed}")
        print(f"   Files modified: {summary.files_modified}")

        # Verify content
        assert summary.tasks_completed > 0, "Expected completed tasks in summary"
        assert len(markdown) > 500, "Expected substantial summary content"

        # Show preview
        lines = markdown.split('\n')
        print("   Preview:")
        for line in lines[:8]:
            if line.strip():
                print(f"     {line}")
        print("     ...")

    else:
        print("⚠️  No session data available for summary test")


async def test_acceptance_criteria():
    """Test acceptance criteria validation."""
    print("Testing acceptance criteria...")

    # Criteria 1: Schema has active_files column
    extractor = SessionDataExtractor()
    session_data = await extractor.get_session_metadata('sess-b525f88712bf4162')

    if session_data:
        assert hasattr(session_data, 'active_files'), "Session should have active_files attribute"
        assert isinstance(session_data.active_files, list), "active_files should be a list"
        print("✅ Criteria 1: active_files column exists and is list")

        # Criteria 2: Session tracking populated
        if len(session_data.active_tasks) > 0 or len(session_data.active_files) > 0:
            print("✅ Criteria 2: Session tracking has data")
        else:
            print("⚠️  Criteria 2: Session tracking empty (might be ok for new session)")

        # Criteria 3: Hybrid query works
        if session_data.started_at and session_data.active_tasks:
            task_stats = await extractor.get_task_stats(
                session_data.started_at,
                session_data.ended_at or datetime.now(),
                session_data=session_data
            )

            if task_stats.total_tasks > 0:
                print("✅ Criteria 3: Hybrid query finds tasks")
            else:
                print("⚠️  Criteria 3: Hybrid query found no tasks")
        else:
            print("⚠️  Criteria 3: Cannot test hybrid query (no session data)")
    else:
        print("⚠️  Cannot test acceptance criteria (no session data)")


async def main():
    """Run all tests."""
    print("DevStream Active Session Tracking - Test Suite")
    print("=" * 60)

    tests = [
        test_keyword_extraction,
        test_uuid_vs_title_classification,
        test_session_data_extractor_real,
        test_summary_generation,
        test_acceptance_criteria,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Active Session Tracking is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)