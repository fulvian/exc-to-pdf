#!/usr/bin/env python3
"""
DevStream Session Data Extractor - Context7 Compliant

Triple-source data extraction for accurate session summaries.
Implements Context7 aiosqlite async patterns with row_factory.

Data Sources:
1. work_sessions → Session metadata, timestamps, tasks
2. semantic_memory → Files modified, decisions, learnings
3. micro_tasks → Task execution history, status changes

Context7 Patterns:
- aiosqlite: async with context managers, Row factory
- Time-range queries for session-scoped data extraction
"""

import sys
import aiosqlite
import sqlite_utils
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import re

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from logger import get_devstream_logger
from sqlite_vec_helper import get_db_connection_with_vec


class DatabaseError(Exception):
    """Database operation error for Context7 sqlite-utils operations."""
    pass


@dataclass
class SessionData:
    """Session metadata from work_sessions table."""
    session_id: str
    session_name: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    tokens_used: int = 0
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    active_files: List[str] = field(default_factory=list)  # FASE 1: Added for tracking
    status: str = "unknown"


@dataclass
class MemoryStats:
    """Aggregated statistics from semantic_memory."""
    files_modified: int = 0
    decisions_made: int = 0
    learnings_captured: int = 0
    total_records: int = 0
    file_list: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)


@dataclass
class TaskStats:
    """Aggregated statistics from micro_tasks."""
    total_tasks: int = 0
    completed: int = 0
    active: int = 0
    failed: int = 0
    task_titles: List[str] = field(default_factory=list)


@dataclass
class RealDataPattern:
    """Real data pattern analysis results."""
    code_changes: List[str] = field(default_factory=list)
    task_completions: List[str] = field(default_factory=list)
    file_modifications: List[str] = field(default_factory=list)
    decision_points: List[str] = field(default_factory=list)
    learning_moments: List[str] = field(default_factory=list)
    error_events: List[str] = field(default_factory=list)
    total_activities: int = 0
    unique_files: int = 0


class SessionDataExtractor:
    """
    Extract session data from multiple sources using Context7 patterns.

    FASE 3 Enhancement:
    - Context7 sqlite-utils time-window queries for precise session filtering
    - Real data pattern recognition for actual semantic_memory structure
    - Session-scoped data extraction with accurate timestamp boundaries

    Implements both async aiosqlite and sync sqlite-utils patterns.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SessionDataExtractor.

        Args:
            db_path: Path to DevStream database (defaults to data/devstream.db)
        """
        self.structured_logger = get_devstream_logger('session_data_extractor')
        self.logger = self.structured_logger.logger

        # Database configuration (updated to use data for Spotlight exclusion)
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.db_path = str(project_root / 'data' / 'devstream.db')
        else:
            self.db_path = db_path

        self.logger.info(f"SessionDataExtractor initialized with DB: {self.db_path}")

    def _get_time_window_bounds(self, session_data: SessionData) -> tuple[datetime, datetime]:
        """
        Get precise time window bounds for session data extraction.

        Context7 Pattern: Accurate timestamp boundary handling with fallbacks.

        Args:
            session_data: Session metadata with timestamps

        Returns:
            Tuple of (start_time, end_time) for precise time-window queries

        Raises:
            ValueError: If session_data is invalid
            DatabaseError: If time window calculation fails

        Note:
            Handles missing timestamps with sensible defaults.
            Uses started_at as primary, last_activity_at as fallback.
        """
        # Context7 Pattern: Input validation
        if not session_data:
            raise ValueError("SessionData cannot be None")

        if not session_data.session_id:
            raise ValueError("SessionData must have a valid session_id")

        try:
            # Primary start time: session started_at
            start_time = session_data.started_at

            # Fallback: use current time - 1 hour if no start time
            if start_time is None:
                self.logger.warning("No started_at found - using fallback (1 hour ago)")
                start_time = datetime.now() - timedelta(hours=1)
            elif not isinstance(start_time, datetime):
                raise ValueError(f"started_at must be datetime, got {type(start_time)}")

            # Primary end time: session ended_at
            end_time = session_data.ended_at

            # Fallback: use last_activity_at or current time
            if end_time is None:
                if hasattr(session_data, 'last_activity_at') and session_data.last_activity_at:
                    end_time = session_data.last_activity_at
                    if not isinstance(end_time, datetime):
                        self.logger.warning(f"last_activity_at is not datetime, using current time")
                        end_time = datetime.now()
                else:
                    end_time = datetime.now()
            elif not isinstance(end_time, datetime):
                raise ValueError(f"ended_at must be datetime, got {type(end_time)}")

            # Ensure time window makes sense (start before end)
            if start_time > end_time:
                self.logger.warning(f"Invalid time window: start {start_time} > end {end_time}")
                start_time, end_time = end_time, start_time

            # Context7 Pattern: Boundary validation
            max_window_days = 7  # Maximum 7-day window to prevent excessive queries
            if (end_time - start_time).days > max_window_days:
                self.logger.warning(f"Time window exceeds {max_window_days} days, truncating")
                start_time = end_time - timedelta(days=max_window_days)

            # Add small buffer (1 minute) to catch edge cases
            start_time = start_time - timedelta(minutes=1)
            end_time = end_time + timedelta(minutes=1)

            self.logger.debug(f"Time window: {start_time} to {end_time}")
            return start_time, end_time

        except Exception as e:
            if isinstance(e, (ValueError, TypeError)):
                self.logger.error(f"Invalid timestamp data: {e}")
                raise ValueError(f"Invalid timestamp data: {e}")
            else:
                self.logger.error(f"Time window calculation failed: {e}")
                raise DatabaseError(f"Failed to calculate time window: {e}")

    def extract_session_data_sqlite_utils(
        self,
        session_data: SessionData,
        content_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract session data using Context7 sqlite-utils time-window pattern.

        FASE 3 Implementation: Precise time-window data extraction.

        Args:
            session_data: Session metadata with timestamps
            content_types: Optional filter by content types

        Returns:
            List of session-scoped memory records

        Raises:
            DatabaseError: If sqlite-utils query fails
        """
        start_time, end_time = self._get_time_window_bounds(session_data)

        try:
            # Context7 Pattern: Use sqlite-utils for precise time-window queries
            # Note: Database object doesn't support context manager, use direct instantiation
            db = sqlite_utils.Database(self.db_path)

            # Build base query with time window
            query_parts = [
                "SELECT id, content_type, content, created_at, keywords",
                "FROM semantic_memory",
                "WHERE created_at BETWEEN ? AND ?"
            ]
            params = [start_time.isoformat(), end_time.isoformat()]

            # Add content type filter if specified
            if content_types:
                placeholders = ','.join('?' * len(content_types))
                query_parts.append(f"AND content_type IN ({placeholders})")
                params.extend(content_types)

            # Order by timestamp (newest first)
            query_parts.append("ORDER BY created_at DESC")

            query = " ".join(query_parts)

            # Execute with Context7 pattern
            results = list(db.query(query, params))

            self.logger.debug(
                f"sqlite-utils time-window query: {len(results)} records "
                f"in {start_time} to {end_time} window"
            )

            return results

        except Exception as e:
            self.logger.error(f"sqlite-utils time-window query failed: {e}")
            raise DatabaseError(f"Failed to extract session data: {e}")

    def analyze_real_patterns(
        self,
        session_records: List[Dict[str, Any]]
    ) -> RealDataPattern:
        """
        Analyze actual patterns in real semantic_memory data.

        FASE 3 Implementation: Pattern recognition for real data analysis.

        Args:
            session_records: Session-scoped memory records from time-window query

        Returns:
            RealDataPattern with analyzed statistics

        Raises:
            ValueError: If session_records is invalid
            DatabaseError: If pattern analysis fails

        Note:
            Analyzes actual data patterns instead of expecting non-existent patterns.
            Implements unique file counting and task completion detection.
        """
        # Context7 Pattern: Input validation
        if not session_records:
            self.logger.debug("No session records provided - returning empty pattern")
            return RealDataPattern()

        if not isinstance(session_records, list):
            raise ValueError(f"session_records must be a list, got {type(session_records)}")

        try:
            pattern = RealDataPattern()
            pattern.total_activities = len(session_records)

            # Track unique files
            unique_files_set = set()
            processed_records = 0
            error_count = 0

            for i, record in enumerate(session_records):
                try:
                    # Context7 Pattern: Record validation
                    if not isinstance(record, dict):
                        self.logger.warning(f"Record {i} is not a dictionary, skipping")
                        error_count += 1
                        continue

                    content_type = record.get('content_type', '')
                    content = record.get('content', '')
                    created_at = record.get('created_at', '')

                    # Validate required fields
                    if not content_type:
                        self.logger.warning(f"Record {i} missing content_type, skipping")
                        continue

                    if not isinstance(content, str):
                        self.logger.warning(f"Record {i} content is not string, skipping")
                        continue

                    # Pattern 1: Code change detection (analyze real content patterns)
                    if content_type == 'code':
                        try:
                            # Extract file paths from actual content patterns
                            file_paths = self._extract_file_paths_from_content(content)
                            for file_path in file_paths:
                                pattern.file_modifications.append(file_path)
                                unique_files_set.add(file_path)

                            # Check for actual code change patterns
                            if self._is_code_change_content(content):
                                # Create safe summary with timestamp
                                summary = self._create_safe_summary(content, created_at, "CODE_CHANGE")
                                pattern.code_changes.append(summary)

                        except Exception as e:
                            self.logger.warning(f"Error processing code record {i}: {e}")
                            error_count += 1

                    # Pattern 2: Task completion detection
                    elif content_type in ['decision', 'learning']:
                        try:
                            # Look for task completion indicators in real content
                            if self._is_task_completion_content(content):
                                summary = self._create_safe_summary(content, created_at, "TASK_COMPLETE")
                                pattern.task_completions.append(summary)

                            # Pattern 3: Decision points (for decision type)
                            if content_type == 'decision':
                                summary = self._create_safe_summary(content, created_at, "DECISION")
                                pattern.decision_points.append(summary)

                            # Pattern 4: Learning moments (for learning type)
                            elif content_type == 'learning':
                                summary = self._create_safe_summary(content, created_at, "LEARNING")
                                pattern.learning_moments.append(summary)

                        except Exception as e:
                            self.logger.warning(f"Error processing decision/learning record {i}: {e}")
                            error_count += 1

                    # Pattern 5: Error events
                    elif content_type == 'error':
                        try:
                            summary = self._create_safe_summary(content, created_at, "ERROR")
                            pattern.error_events.append(summary)
                        except Exception as e:
                            self.logger.warning(f"Error processing error record {i}: {e}")
                            error_count += 1

                    processed_records += 1

                except Exception as e:
                    self.logger.warning(f"Unexpected error processing record {i}: {e}")
                    error_count += 1
                    continue

            # Calculate unique files
            pattern.unique_files = len(unique_files_set)

            # Log processing summary
            self.logger.info(
                f"Pattern analysis completed: {processed_records}/{len(session_records)} records processed, "
                f"{error_count} errors, {pattern.unique_files} unique files, "
                f"{len(pattern.code_changes)} code changes, {len(pattern.task_completions)} task completions"
            )

            # Context7 Pattern: Data quality check
            if error_count > len(session_records) * 0.2:  # >20% error rate
                self.logger.warning(
                    f"High error rate in pattern analysis: {error_count}/{len(session_records)} records failed"
                )

            return pattern

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            raise DatabaseError(f"Failed to analyze patterns: {e}")

    def _create_safe_summary(self, content: str, timestamp: str, pattern_type: str) -> str:
        """
        Create a safe summary string for pattern analysis.

        Context7 Pattern: Safe content summarization with validation.

        Args:
            content: Content to summarize
            timestamp: Timestamp string
            pattern_type: Type of pattern (for logging)

        Returns:
            Safe summary string

        Note:
            Limits content length and handles encoding issues.
        """
        try:
            # Clean and validate content
            if not content:
                content = "(empty content)"

            # Ensure content is string and handle encoding
            if not isinstance(content, str):
                content = str(content)

            # Remove potentially problematic characters
            content = content.replace('\0', '').replace('\r', '').replace('\n', ' ')

            # Limit length for safety
            max_length = 100
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # Clean timestamp
            if not timestamp:
                timestamp = datetime.now().isoformat()

            return f"{timestamp}: {content}"

        except Exception as e:
            self.logger.warning(f"Error creating safe summary for {pattern_type}: {e}")
            return f"{datetime.now().isoformat()}: (error processing content)"

    def _extract_file_paths_from_content(self, content: str) -> List[str]:
        """
        Extract file paths from semantic_memory content using real patterns.

        Args:
            content: Content text to analyze

        Returns:
            List of file paths found in content
        """
        file_paths = []

        # Pattern 1: Look for file modification patterns in real data
        # Based on analysis of actual semantic_memory content
        file_patterns = [
            r'File Modified:\s*([^\s\n]+(?:\.[a-zA-Z0-9]+)?)',
            r'file:\s*([^\s\n]+(?:\.[a-zA-Z0-9]+)?)',
            r'path:\s*([^\s\n]+(?:\.[a-zA-Z0-9]+)?)',
            r'([/\\][\w/\\.-]+\.[a-zA-Z0-9]+)',  # Unix/Windows paths
            r'([\w-]+\.[a-zA-Z0-9]+)',  # Simple filenames
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Clean up and validate file path
                file_path = match.strip().strip('\'"')
                if len(file_path) > 3 and '.' in file_path:  # Basic validation
                    file_paths.append(file_path)

        return list(set(file_paths))  # Remove duplicates

    def _is_code_change_content(self, content: str) -> bool:
        """
        Determine if content represents actual code change.

        Args:
            content: Content text to analyze

        Returns:
            True if content represents code change
        """
        # Real code change indicators found in semantic_memory
        code_indicators = [
            'modified', 'created', 'updated', 'deleted', 'added',
            'function', 'class', 'method', 'import', 'export',
            'def ', 'async def', 'class ', 'import ', 'from import',
            'PostToolUse', 'Edit file', 'Write file', 'Create file'
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in code_indicators)

    def _is_task_completion_content(self, content: str) -> bool:
        """
        Determine if content represents task completion.

        Args:
            content: Content text to analyze

        Returns:
            True if content represents task completion
        """
        # Task completion indicators found in real semantic_memory
        completion_indicators = [
            'completed', 'finished', 'done', 'implemented',
            'task completed', 'phase completed', 'milestone',
            '✅', '✓', '✔️',  # Check marks (unicode)
            'success', 'passed', 'working', 'fixed'
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in completion_indicators)

    async def get_enhanced_memory_stats(
        self,
        session_data: SessionData,
        content_types: Optional[List[str]] = None
    ) -> tuple[MemoryStats, RealDataPattern]:
        """
        Get enhanced memory statistics with real pattern analysis.

        FASE 3 Enhancement: Combines traditional stats with real pattern analysis.

        Args:
            session_data: Session metadata with timestamps
            content_types: Optional filter by content types

        Returns:
            Tuple of (MemoryStats, RealDataPattern)
        """
        # Use Context7 sqlite-utils time-window extraction
        session_records = self.extract_session_data_sqlite_utils(session_data, content_types)

        # Traditional stats aggregation
        stats = MemoryStats()

        for record in session_records:
            content_type = record.get('content_type', '')
            content = record.get('content', '')

            stats.total_records += 1

            if content_type == 'code':
                stats.files_modified += 1
                # Extract file paths for detailed tracking
                file_paths = self._extract_file_paths_from_content(content)
                stats.file_list.extend(file_paths)

            elif content_type == 'decision':
                stats.decisions_made += 1
                stats.decisions.append(content[:200])

            elif content_type == 'learning':
                stats.learnings_captured += 1
                stats.learnings.append(content[:200])

        # Real pattern analysis
        pattern = self.analyze_real_patterns(session_records)

        # Remove duplicates from file list
        stats.file_list = list(set(stats.file_list))

        self.logger.info(
            f"Enhanced memory stats: {stats.total_records} records, "
            f"{pattern.unique_files} unique files, "
            f"{len(pattern.code_changes)} actual code changes detected"
        )

        return stats, pattern

    async def get_session_metadata(self, session_id: str) -> Optional[SessionData]:
        """
        Extract session metadata from work_sessions table.

        Context7 Pattern: async with + row_factory for clean access.

        Args:
            session_id: Session identifier

        Returns:
            SessionData if found, None otherwise
        """
        try:
            # Context7 Pattern: async with aiosqlite.connect()
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row  # Dictionary-like access

                async with db.execute(
                    """
                    SELECT id, session_name, started_at, ended_at, tokens_used,
                           active_tasks, completed_tasks, status
                    FROM work_sessions
                    WHERE id = ?
                    """,
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row is None:
                        self.logger.warning(f"No session found: {session_id}")
                        return None

                    # Parse JSON fields
                    import json
                    active_tasks = json.loads(row['active_tasks']) if row['active_tasks'] else []
                    completed_tasks = json.loads(row['completed_tasks']) if row['completed_tasks'] else []
                    active_files = []  # Column doesn't exist in database, set empty list

                    return SessionData(
                        session_id=row['id'],
                        session_name=row['session_name'],
                        started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                        ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None,
                        tokens_used=row['tokens_used'],
                        active_tasks=active_tasks,
                        completed_tasks=completed_tasks,
                        active_files=active_files,
                        status=row['status']
                    )

        except Exception as e:
            self.logger.error(f"Failed to extract session metadata: {e}")
            return None

    async def get_memory_stats(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        session_data: Optional[SessionData] = None
    ) -> MemoryStats:
        """
        Extract memory statistics for time range with FASE 3 enhancements.

        Context7 Pattern: Enhanced with sqlite-utils time-window queries and
        real data pattern recognition.

        Args:
            start_time: Session start timestamp (for backward compatibility)
            end_time: Session end timestamp (default: now)
            session_data: Session metadata with timestamps (FASE 3 enhancement)

        Returns:
            MemoryStats with aggregated counts and samples

        Note:
            If session_data is provided, uses enhanced time-window extraction.
            Falls back to legacy time-based approach for backward compatibility.
        """
        # FASE 3: Use enhanced approach if session_data available
        if session_data:
            stats, _ = await self.get_enhanced_memory_stats(session_data)
            return stats

        # Legacy approach for backward compatibility
        if end_time is None:
            end_time = datetime.now()

        stats = MemoryStats()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get aggregated counts by content_type
                async with db.execute(
                    """
                    SELECT content_type, COUNT(*) as count
                    FROM semantic_memory
                    WHERE created_at BETWEEN ? AND ?
                    GROUP BY content_type
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        content_type = row['content_type']
                        count = row['count']

                        if content_type == 'code':
                            stats.files_modified = count
                        elif content_type == 'decision':
                            stats.decisions_made = count
                        elif content_type == 'learning':
                            stats.learnings_captured = count

                        stats.total_records += count

                # Get sample file names (top 10)
                async with db.execute(
                    """
                    SELECT DISTINCT content
                    FROM semantic_memory
                    WHERE content_type = 'code'
                      AND created_at BETWEEN ? AND ?
                    ORDER BY created_at DESC
                    LIMIT 10
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        # Extract filename from content (first line usually has it)
                        content = row['content']
                        first_line = content.split('\n')[0] if content else ''
                        if 'File Modified:' in first_line:
                            filename = first_line.split('File Modified:')[1].strip()
                            stats.file_list.append(filename)

                # Get decisions (top 5)
                async with db.execute(
                    """
                    SELECT content
                    FROM semantic_memory
                    WHERE content_type = 'decision'
                      AND created_at BETWEEN ? AND ?
                    ORDER BY created_at DESC
                    LIMIT 5
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        stats.decisions.append(row['content'][:200])  # First 200 chars

                # Get learnings (top 5)
                async with db.execute(
                    """
                    SELECT content
                    FROM semantic_memory
                    WHERE content_type = 'learning'
                      AND created_at BETWEEN ? AND ?
                    ORDER BY created_at DESC
                    LIMIT 5
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        stats.learnings.append(row['content'][:200])

            self.logger.debug(
                f"Memory stats extracted (legacy): {stats.total_records} records, "
                f"{stats.files_modified} files, {stats.decisions_made} decisions"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to extract memory stats (legacy): {e}")
            return stats

    async def _get_task_stats_by_tracking(
        self,
        session_data: SessionData
    ) -> TaskStats:
        """
        Extract task statistics based on SESSION TRACKING (active_tasks).

        Memory Bank Pattern: Query tasks that were ACTIVELY WORKED ON during session,
        not based on time ranges. Fixes timezone bug and empty summary issues.

        Args:
            session_data: Session metadata including active_tasks list

        Returns:
            TaskStats with aggregated counts and task titles

        Note:
            Queries micro_tasks by matching UUID-style task IDs OR title patterns.
            Fallback to empty stats if no active_tasks tracked.
        """
        stats = TaskStats()

        # Early return if no active tasks tracked
        if not session_data.active_tasks:
            self.logger.debug("No active_tasks tracked - returning empty stats")
            return stats

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Separate UUID-style IDs from title strings
                uuid_tasks = []
                title_tasks = []

                for task in session_data.active_tasks:
                    # Check if task looks like a UUID (32 hex chars, possibly with hyphens)
                    import re
                    if re.match(r'^[a-f0-9]{32}$', task.replace('-', '')) or re.match(r'^[a-f0-9-]{36}$', task):
                        uuid_tasks.append(task)
                    else:
                        title_tasks.append(task)

                self.logger.debug(
                    f"Task classification: {len(uuid_tasks)} UUID tasks, "
                    f"{len(title_tasks)} title tasks"
                )

                # Query 1: UUID-style exact matches
                if uuid_tasks:
                    placeholders = ','.join('?' * len(uuid_tasks))

                    # Get counts by status
                    query = f"""
                        SELECT status, COUNT(*) as count
                        FROM micro_tasks
                        WHERE id IN ({placeholders})
                        GROUP BY status
                    """
                    async with db.execute(query, uuid_tasks) as cursor:
                        async for row in cursor:
                            status = row['status']
                            count = row['count']

                            stats.total_tasks += count

                            if status == 'completed':
                                stats.completed = count
                            elif status == 'active':
                                stats.active = count
                            elif status == 'failed':
                                stats.failed = count

                    # Get task titles
                    query = f"""
                        SELECT title, status, completed_at
                        FROM micro_tasks
                        WHERE id IN ({placeholders})
                        ORDER BY
                            CASE status
                                WHEN 'completed' THEN 1
                                WHEN 'active' THEN 2
                                ELSE 3
                            END,
                            completed_at DESC
                        LIMIT 10
                    """
                    async with db.execute(query, uuid_tasks) as cursor:
                        async for row in cursor:
                            stats.task_titles.append(row['title'])

                # Query 2: Keyword-based LIKE matches (Context7 pattern: extract keywords from long titles)
                if title_tasks:
                    # Extract meaningful keywords from long TodoWrite titles
                    import re
                    all_keywords = set()

                    for title in title_tasks:
                        # Extract keywords: words 4+ chars, exclude common words
                        words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())

                        # Filter out common words and keep meaningful ones
                        common_words = {
                            'this', 'that', 'with', 'from', 'they', 'have', 'been',
                            'were', 'said', 'each', 'which', 'their', 'time', 'will',
                            'about', 'would', 'could', 'should', 'other', 'after',
                            'first', 'into', 'present', 'solution', 'trade', 'offs'
                        }

                        meaningful_words = [w for w in words if w not in common_words and len(w) >= 4]
                        all_keywords.update(meaningful_words)

                    # Context7 pattern: Use keyword-based matching for better recall
                    if all_keywords:
                        # Limit keywords to most relevant ones to avoid too broad queries
                        keywords = list(all_keywords)[:8]  # Top 8 keywords

                        self.logger.debug(f"Extracted keywords from titles: {keywords}")

                        # Build keyword-based OR conditions
                        keyword_conditions = ' OR '.join(['title LIKE ?'] * len(keywords))
                        keyword_params = [f'%{keyword}%' for keyword in keywords]

                        # Get counts by status
                        query = f"""
                            SELECT status, COUNT(*) as count
                            FROM micro_tasks
                            WHERE {keyword_conditions}
                            GROUP BY status
                        """
                        async with db.execute(query, keyword_params) as cursor:
                            async for row in cursor:
                                status = row['status']
                                count = row['count']

                                stats.total_tasks += count

                                if status == 'completed':
                                    stats.completed = count
                                elif status == 'active':
                                    stats.active = count
                                elif status == 'failed':
                                    stats.failed = count

                        # Get task titles (distinct, limit to avoid duplicates)
                        query = f"""
                            SELECT DISTINCT title, status, completed_at
                            FROM micro_tasks
                            WHERE {keyword_conditions}
                            ORDER BY
                                CASE status
                                    WHEN 'completed' THEN 1
                                    WHEN 'active' THEN 2
                                    ELSE 3
                                END,
                                completed_at DESC
                            LIMIT 10
                        """
                        async with db.execute(query, keyword_params) as cursor:
                            async for row in cursor:
                                title = row['title']
                                if title not in stats.task_titles:  # Deduplicate
                                    stats.task_titles.append(title)

            self.logger.debug(
                f"Task stats (tracking-based): {stats.total_tasks} total, "
                f"{stats.completed} completed, from {len(session_data.active_tasks)} tracked items "
                f"({len(uuid_tasks)} UUID + {len(title_tasks)} titles)"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to extract task stats (tracking): {e}")
            return stats

    async def _get_task_stats_by_time(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> TaskStats:
        """
        Extract task statistics based on TIME RANGE (legacy fallback).

        Fallback method for sessions without active_tasks tracking.
        Uses created_at BETWEEN time range query.

        Args:
            start_time: Session start timestamp
            end_time: Session end timestamp (default: now)

        Returns:
            TaskStats with aggregated counts and task titles

        Note:
            This is the OLD behavior - kept for backward compatibility.
            Subject to timezone bugs and includes tasks not actively worked on.
        """
        if end_time is None:
            end_time = datetime.now()

        stats = TaskStats()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get counts by status
                async with db.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM micro_tasks
                    WHERE created_at BETWEEN ? AND ?
                    GROUP BY status
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        status = row['status']
                        count = row['count']

                        stats.total_tasks += count

                        if status == 'completed':
                            stats.completed = count
                        elif status == 'active':
                            stats.active = count
                        elif status == 'failed':
                            stats.failed = count

                # Get task titles (top 10 completed)
                async with db.execute(
                    """
                    SELECT title
                    FROM micro_tasks
                    WHERE created_at BETWEEN ? AND ?
                      AND status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT 10
                    """,
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    async for row in cursor:
                        stats.task_titles.append(row['title'])

            self.logger.debug(
                f"Task stats (time-based fallback): {stats.total_tasks} total, "
                f"{stats.completed} completed"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to extract task stats (time): {e}")
            return stats

    async def get_task_stats(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        session_data: Optional[SessionData] = None
    ) -> TaskStats:
        """
        Extract task statistics using HYBRID approach (tracking + fallback).

        FASE 3 Enhancement: Prioritize session tracking over time-based queries.

        Strategy:
        1. TRY: Session tracking (if session_data.active_tasks exists)
        2. FALLBACK: Time-based query (for backward compatibility)

        Args:
            start_time: Session start timestamp (for fallback)
            end_time: Session end timestamp (for fallback, default: now)
            session_data: Session metadata with active_tasks (NEW)

        Returns:
            TaskStats with aggregated counts and task titles

        Note:
            Backward compatible - old code can still call without session_data.
            New code should pass session_data for tracking-based queries.
        """
        # STRATEGY 1: Try session tracking (Memory Bank pattern)
        if session_data and session_data.active_tasks:
            self.logger.debug("Using tracking-based query (Memory Bank pattern)")
            return await self._get_task_stats_by_tracking(session_data)

        # STRATEGY 2: Fallback to time-based query (backward compat)
        self.logger.warning(
            "No active_tasks tracked - falling back to time-based query "
            "(subject to timezone bugs)"
        )
        return await self._get_task_stats_by_time(start_time, end_time)


if __name__ == "__main__":
    # Test script
    import asyncio

    async def test():
        print("DevStream Session Data Extractor Test - FASE 3 Enhanced")
        print("=" * 60)

        extractor = SessionDataExtractor()

        # Test with last hour of data
        from datetime import timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        print(f"\n1. Testing enhanced memory stats extraction...")
        print(f"   Time range: {start_time} to {end_time}")

        # Test legacy approach
        memory_stats = await extractor.get_memory_stats(start_time, end_time)
        print(f"   Legacy - Total records: {memory_stats.total_records}")
        print(f"   Legacy - Files modified: {memory_stats.files_modified}")
        print(f"   Legacy - Decisions: {memory_stats.decisions_made}")
        print(f"   Legacy - Learnings: {memory_stats.learnings_captured}")

        # Test enhanced approach with sample session data
        sample_session = SessionData(
            session_id="test-session-123",
            session_name="Test Session",
            started_at=start_time,
            ended_at=end_time,
            tokens_used=1000,
            active_tasks=["test-task-1", "test-task-2"],
            completed_tasks=[],
            active_files=[],
            status="completed"
        )

        print(f"\n2. Testing Context7 sqlite-utils time-window queries...")
        session_records = extractor.extract_session_data_sqlite_utils(sample_session)
        print(f"   Session records found: {len(session_records)}")

        if session_records:
            print(f"   Sample record types: {[r.get('content_type', 'unknown') for r in session_records[:5]]}")

        print(f"\n3. Testing real data pattern analysis...")
        if session_records:
            pattern = extractor.analyze_real_patterns(session_records)
            print(f"   Total activities: {pattern.total_activities}")
            print(f"   Unique files: {pattern.unique_files}")
            print(f"   Code changes detected: {len(pattern.code_changes)}")
            print(f"   Task completions: {len(pattern.task_completions)}")
            print(f"   Decision points: {len(pattern.decision_points)}")
            print(f"   Learning moments: {len(pattern.learning_moments)}")

        print(f"\n4. Testing enhanced memory stats with real patterns...")
        if session_records:
            enhanced_stats, enhanced_pattern = await extractor.get_enhanced_memory_stats(sample_session)
            print(f"   Enhanced - Total records: {enhanced_stats.total_records}")
            print(f"   Enhanced - Files modified: {enhanced_stats.files_modified}")
            print(f"   Enhanced - Unique files from patterns: {enhanced_pattern.unique_files}")
            print(f"   Enhanced - Real code changes: {len(enhanced_pattern.code_changes)}")

        print(f"\n5. Testing file path extraction...")
        test_content = "PostToolUse Edit file /Users/fulvio/test.py - Function modified"
        file_paths = extractor._extract_file_paths_from_content(test_content)
        print(f"   Test content: '{test_content}'")
        print(f"   Extracted files: {file_paths}")

        print(f"\n6. Testing code change detection...")
        code_tests = [
            "Modified function in test.py",
            "Created new class TestClass",
            "Updated import statements",
            "This is just a log message"
        ]
        for test in code_tests:
            is_code = extractor._is_code_change_content(test)
            print(f"   '{test[:30]}...' -> {'CODE' if is_code else 'NOT CODE'}")

        print(f"\n7. Testing task completion detection...")
        task_tests = [
            "Task completed successfully",
            "Phase 1 implementation finished",
            "✅ All tests passed",
            "In progress working on it"
        ]
        for test in task_tests:
            is_complete = extractor._is_task_completion_content(test)
            print(f"   '{test[:30]}...' -> {'COMPLETE' if is_complete else 'NOT COMPLETE'}")

        print(f"\n8. Testing task stats extraction...")
        task_stats = await extractor.get_task_stats(start_time, end_time)
        print(f"   Total tasks: {task_stats.total_tasks}")
        print(f"   Completed: {task_stats.completed}")
        print(f"   Active: {task_stats.active}")
        print(f"   Failed: {task_stats.failed}")

        print("\n" + "=" * 60)
        print("FASE 3 Test completed!")
        print("\nKey Enhancements:")
        print("✅ Context7 sqlite-utils time-window queries")
        print("✅ Real data pattern recognition")
        print("✅ Enhanced file path extraction")
        print("✅ Code change detection")
        print("✅ Task completion analysis")
        print("✅ Backward compatibility maintained")

    asyncio.run(test())