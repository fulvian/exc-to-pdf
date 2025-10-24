#!/usr/bin/env python3
"""
DevStream SessionSummaryManager - B2 Behavioral Refinement

Centralized session summary management with:
- Async/await Context7 patterns (aiosqlite)
- Hybrid search integration for retrieval
- Enhanced session goal inference
- Clean separation of concerns

Architecture:
- Extract: Query semantic memory for session activities
- Analyze: Infer goal, extract tasks/files/decisions
- Generate: Create Context7-compliant structured summary
- Store: Save to semantic memory with type "context"
- Display: Retrieve and show in SessionStart

Context7 Patterns:
- aiosqlite async patterns for database access
- Hybrid search for retrieval (RRF algorithm)
- LangMem + Anthropic episodic memory structure
"""

import asyncio
import aiosqlite
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import DevStream utilities
import sys
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from logger import get_devstream_logger


class SessionSummaryManager:
    """
    Manages session summary lifecycle: extraction, analysis, generation, storage, retrieval.

    Responsibilities:
    - Extract session activities from semantic memory
    - Infer session goal from multiple context sources
    - Generate Context7-compliant structured summaries
    - Store summaries in semantic memory
    - Retrieve summaries for SessionStart display
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize SessionSummaryManager.

        Args:
            db_path: Optional database path override
        """
        self.structured_logger = get_devstream_logger('session_summary_manager')
        self.logger = self.structured_logger.logger

        # Database path
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = project_root / "data" / "devstream.db"

        self.db_path = db_path
        self.logger.debug(f"SessionSummaryManager initialized with db_path={db_path}")

    async def extract_session_data(
        self,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Extract session activities from semantic memory.

        Args:
            hours_back: Hours to look back for memories
            limit: Maximum memories to retrieve

        Returns:
            List of memory records sorted by recency
        """
        if not self.db_path.exists():
            self.logger.warning(f"Database not found: {self.db_path}")
            return []

        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%d %H:%M:%S")

            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row

                query = """
                SELECT content, content_type, created_at, keywords
                FROM semantic_memory
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """

                async with db.execute(query, (cutoff_time, limit)) as cursor:
                    rows = await cursor.fetchall()
                    memories = [dict(row) for row in rows]

            self.logger.info(f"Extracted {len(memories)} memories from last {hours_back}h")
            return memories

        except Exception as e:
            self.logger.error(f"Failed to extract session data: {e}")
            return []

    def analyze_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze memories to extract structured session information.

        Args:
            memories: List of memory records

        Returns:
            Dictionary with analyzed session data:
                - completed_tasks: List[str]
                - modified_files: List[str]
                - key_decisions: List[str]
                - errors: List[str]
                - session_context: List[str]
        """
        analysis = {
            "completed_tasks": [],
            "modified_files": set(),
            "key_decisions": [],
            "errors": [],
            "session_context": []
        }

        for memory in memories:
            content = memory['content']
            content_lower = content.lower()
            content_type = memory['content_type']

            # Skip generic session-end markers
            if 'Session Completed' in content and 'DevStream session ended' in content:
                continue

            # Extract completed tasks (TodoWrite completions)
            if 'todo' in content_lower and any(kw in content_lower for kw in ['completed', 'done', '✅']):
                task_lines = [
                    line.strip('- *✅').strip()
                    for line in content.split('\n')
                    if line.strip().startswith(('- ', '* ', '✅'))
                    and len(line.strip()) > 5
                ]
                analysis["completed_tasks"].extend(task_lines[:5])

            # Extract modified files
            if any(keyword in content_lower for keyword in ['edit', 'write', 'modified', 'updated', 'file']):
                file_patterns = [
                    r'`([^`]+\.(py|md|json|ts|js|yaml|yml))`',  # Backtick-quoted files
                    r'\.claude/hooks/[^\s]+\.py',
                    r'/[a-zA-Z0-9_/]+\.(py|md|json|ts|js)',
                    r'[a-zA-Z0-9_]+\.(py|md|json)',
                ]
                for pattern in file_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches:
                            if isinstance(match, tuple):
                                analysis["modified_files"].add(match[0])
                            else:
                                analysis["modified_files"].add(match)

            # Extract key decisions
            if content_type == 'decision':
                sentences = content.split('.')
                for sentence in sentences[:3]:
                    sentence = sentence.strip()
                    if len(sentence) > 30 and not sentence.startswith('#'):
                        analysis["key_decisions"].append(sentence)
                        break

            # Extract errors and issues
            if any(marker in content for marker in ['❌', '⚠️']) or 'error' in content_lower or 'failed' in content_lower:
                error_lines = [line.strip() for line in content.split('\n') if '❌' in line or 'error' in line.lower()]
                for error_line in error_lines[:3]:
                    if len(error_line) > 20:
                        analysis["errors"].append(error_line[:200])

            # Capture documentation and learning content for context
            if content_type in ['documentation', 'learning'] and len(content) > 100:
                lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
                if lines:
                    analysis["session_context"].append(lines[0][:150])

        # Deduplicate and clean
        analysis["completed_tasks"] = list(dict.fromkeys(analysis["completed_tasks"]))[:6]
        analysis["modified_files"] = sorted(list(analysis["modified_files"]))[:12]
        analysis["key_decisions"] = list(dict.fromkeys(analysis["key_decisions"]))[:4]
        analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:3]
        analysis["session_context"] = analysis["session_context"][:2]

        return analysis

    def infer_session_goal(
        self,
        completed_tasks: List[str],
        modified_files: List[str],
        key_decisions: List[str],
        session_context: List[str]
    ) -> str:
        """
        Infer session goal using multiple context sources.

        Args:
            completed_tasks: List of task descriptions
            modified_files: List of file paths
            key_decisions: List of decision texts
            session_context: List of context snippets

        Returns:
            Inferred goal description
        """
        # Combine all available context
        all_text = " ".join(completed_tasks + [str(c) for c in session_context]).lower()
        file_patterns = " ".join(modified_files).lower()

        # Check for specific patterns with priority
        if 'summary' in file_patterns and 'stop' in file_patterns:
            return "Enhance session summary generation system"
        elif 'context7' in all_text or 'best practice' in all_text:
            return "Implement Context7 best practices"
        elif 'hook' in file_patterns and any(kw in all_text for kw in ['implement', 'create', 'add']):
            return "Implement hook system functionality"
        elif any(keyword in all_text for keyword in ['vec0', 'sqlite', 'extension', 'database']):
            return "Fix database and extension integration"
        elif any(keyword in all_text for keyword in ['fix', 'bug', 'error', 'issue', 'resolve']):
            if 'critical' in all_text or 'production' in all_text:
                return "Resolve critical production issues"
            return "Fix bugs and technical issues"
        elif any(keyword in all_text for keyword in ['implement', 'add', 'create', 'build']):
            return "Implement new features and functionality"
        elif any(keyword in all_text for keyword in ['refactor', 'improve', 'optimize', 'enhance']):
            return "Refactor and optimize code"
        elif any(keyword in all_text for keyword in ['test', 'validate', 'verify']):
            return "Test and validate implementation"
        elif any(keyword in all_text for keyword in ['document', 'docs', 'readme']):
            return "Update documentation"

        # Fallback: use first task or first context item
        if completed_tasks:
            return completed_tasks[0][:100]
        elif session_context:
            return str(session_context[0])[:100]

        return "Development work"

    def generate_structured_summary(
        self,
        session_goal: str,
        completed_tasks: List[str],
        modified_files: List[str],
        key_decisions: List[str],
        errors: List[str],
        session_context: List[str],
        total_memories: int
    ) -> str:
        """
        Generate Context7-compliant structured summary.

        Format follows episodic memory structure: observation → thoughts → action → result
        Optimized for session continuity and LLM retrieval.

        Args:
            session_goal: Inferred session goal
            completed_tasks: List of completed task descriptions
            modified_files: List of modified file paths
            key_decisions: List of key decision texts
            errors: List of error messages
            session_context: List of session context snippets
            total_memories: Total memories analyzed

        Returns:
            Formatted markdown summary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_only = datetime.now().strftime("%Y-%m-%d")
        time_only = datetime.now().strftime("%H:%M")

        summary_parts = [
            "# 📋 Session Summary",
            f"\n**Session**: {date_only} ending at {time_only}",
            f"**Goal**: {session_goal}\n"
        ]

        # What Happened (narrative - MemoChat pattern)
        narrative = self._generate_session_narrative(
            completed_tasks, modified_files, key_decisions, errors, session_context
        )
        if narrative:
            summary_parts.append("## 🎯 What Happened")
            summary_parts.append(narrative)
            summary_parts.append("")

        # Completed Work
        if completed_tasks:
            summary_parts.append("## ✅ Completed Work")
            for i, task in enumerate(completed_tasks, 1):
                task_clean = task.strip('- *✅').strip()
                if len(task_clean) > 5:
                    summary_parts.append(f"{i}. {task_clean}")
            summary_parts.append("")

        # Files Modified
        if modified_files:
            summary_parts.append(f"## 📁 Files Modified ({len(modified_files)})")
            for file in modified_files[:10]:
                summary_parts.append(f"- `{file}`")
            if len(modified_files) > 10:
                summary_parts.append(f"\n_(and {len(modified_files) - 10} more files)_")
            summary_parts.append("")

        # Technical Decisions
        if key_decisions:
            summary_parts.append("## 🔍 Technical Decisions")
            for i, decision in enumerate(key_decisions, 1):
                decision_clean = decision.strip()
                if '\n' in decision_clean:
                    lines = [l.strip() for l in decision_clean.split('\n') if l.strip()]
                    summary_parts.append(f"{i}. **{lines[0]}**")
                    if len(lines) > 1:
                        summary_parts.append(f"   {lines[1][:200]}")
                else:
                    summary_parts.append(f"{i}. {decision_clean[:300]}")
            summary_parts.append("")

        # Known Issues
        if errors:
            summary_parts.append("## ⚠️ Known Issues")
            for error in errors:
                error_clean = error.strip('- ').strip()
                if error_clean:
                    summary_parts.append(f"- {error_clean[:200]}")
            summary_parts.append("")

        # Impact & Next Steps
        summary_parts.append("## 🚀 Impact & Next Steps")
        impact = self._generate_impact_statement(completed_tasks, modified_files, key_decisions)
        summary_parts.append(f"**Impact**: {impact}\n")

        summary_parts.append("**Next Session Should**:")
        next_steps = self._generate_smart_next_steps(completed_tasks, errors, session_goal)
        for step in next_steps:
            summary_parts.append(f"- {step}")
        summary_parts.append("")

        # Session Metrics
        if total_memories > 0 or modified_files or key_decisions:
            summary_parts.append("## 📊 Session Metrics")
            if total_memories > 0:
                summary_parts.append(f"- Memories Analyzed: {total_memories}")
            if modified_files:
                summary_parts.append(f"- Files Modified: {len(modified_files)}")
            if key_decisions:
                summary_parts.append(f"- Decisions Made: {len(key_decisions)}")
            if completed_tasks:
                summary_parts.append(f"- Tasks Completed: {len(completed_tasks)}")
            summary_parts.append("- Status: ✅ Ready for continuation")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def _generate_session_narrative(
        self,
        completed_tasks: List[str],
        modified_files: List[str],
        key_decisions: List[str],
        errors: List[str],
        session_context: List[str]
    ) -> Optional[str]:
        """Generate a narrative summary of what happened (MemoChat pattern)."""
        narratives = []

        # Opening: What was attempted
        if session_context:
            setup = str(session_context[0])[:150]
            if not setup.endswith('.'):
                setup += '.'
            narratives.append(setup)

        # Action: What was done
        if modified_files and key_decisions:
            action = f"Modified {len(modified_files)} files implementing {len(key_decisions)} technical decisions."
            narratives.append(action)
        elif modified_files:
            action = f"Modified {len(modified_files)} files across the codebase."
            narratives.append(action)
        elif completed_tasks:
            action = f"Completed {len(completed_tasks)} development tasks."
            narratives.append(action)

        # Outcome: Result
        if errors:
            outcome = f"Encountered {len(errors)} issues that need attention."
            narratives.append(outcome)
        elif completed_tasks or modified_files:
            outcome = "Work completed successfully and ready for continuation."
            narratives.append(outcome)

        return " ".join(narratives) if narratives else None

    def _generate_impact_statement(
        self,
        completed_tasks: List[str],
        modified_files: List[str],
        key_decisions: List[str]
    ) -> str:
        """Generate an impact statement describing the significance of the session's work."""
        if not (completed_tasks or modified_files or key_decisions):
            return "Exploratory session, no major changes"

        # Determine impact level
        if len(modified_files) > 8 or len(key_decisions) > 3:
            level = "Significant changes"
        elif len(modified_files) > 4 or len(key_decisions) > 1:
            level = "Moderate updates"
        else:
            level = "Minor modifications"

        # Identify what changed
        components = []
        if any('hook' in f.lower() for f in modified_files):
            components.append("hook system")
        if any('context' in f.lower() for f in modified_files):
            components.append("context management")
        if any('memory' in f.lower() or 'summary' in f.lower() for f in modified_files):
            components.append("memory/summary system")
        if any('test' in f.lower() for f in modified_files):
            components.append("testing infrastructure")

        if components:
            component_str = ", ".join(components[:3])
            return f"{level} to {component_str}. Ready for integration and testing."

        return f"{level} made. System ready for continuation."

    def _generate_smart_next_steps(
        self,
        completed_tasks: List[str],
        errors: List[str],
        session_goal: str
    ) -> List[str]:
        """Generate intelligent next steps based on session outcome."""
        steps = []

        # If there are errors, prioritize them
        if errors:
            steps.append("Address known issues and errors")

        # Based on goal, suggest logical next actions
        goal_lower = session_goal.lower()

        if 'implement' in goal_lower or 'create' in goal_lower:
            steps.append("Test implemented functionality")
            if 'hook' in goal_lower:
                steps.append("Validate hook execution in real scenarios")
            steps.append("Update documentation for new features")
        elif 'fix' in goal_lower or 'bug' in goal_lower:
            steps.append("Verify fixes work in production")
            steps.append("Add regression tests")
        elif 'refactor' in goal_lower:
            steps.append("Review refactored code for issues")
            steps.append("Update related tests")
        elif 'test' in goal_lower:
            steps.append("Analyze test results")
            steps.append("Fix any failing tests")
        elif 'summary' in goal_lower or 'session' in goal_lower:
            steps.append("Test summary quality with real workflow")
            steps.append("Monitor session continuity")

        # Always suggest review as a step
        if not any('review' in s.lower() for s in steps):
            steps.append("Review completed work")

        # Generic continuation
        if not steps:
            steps.append("Continue with pending tasks")

        return steps[:5]  # Max 5 steps

    async def store_summary(self, summary: str) -> Tuple[bool, Optional[str]]:
        """
        Store session summary in semantic memory.

        Args:
            summary: Formatted summary text

        Returns:
            Tuple of (success: bool, memory_id: Optional[str])
        """
        if not self.db_path.exists():
            self.logger.warning(f"Database not found: {self.db_path}")
            return False, None

        try:
            # Generate MD5 hash for ID
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            memory_id = hashlib.md5(f"session-summary-{timestamp_str}".encode()).hexdigest()

            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT INTO semantic_memory (id, content, content_type, keywords, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    summary,
                    "context",
                    "session-end,summary,devstream",
                    timestamp_str
                ))
                await db.commit()

            self.logger.info(f"Stored session summary: {memory_id[:8]}..., {len(summary)} chars")
            return True, memory_id

        except Exception as e:
            self.logger.error(f"Failed to store summary: {e}")
            return False, None

    async def generate_and_store_summary(self) -> Tuple[bool, str]:
        """
        Complete workflow: extract → analyze → generate → store.

        Returns:
            Tuple of (success: bool, summary: str)
        """
        try:
            # Extract session data
            memories = await self.extract_session_data()
            if not memories:
                self.logger.warning("No memories found, generating fallback summary")
                summary = self._generate_fallback_summary()
                return False, summary

            # Analyze memories
            analysis = self.analyze_memories(memories)

            # Infer session goal
            session_goal = self.infer_session_goal(
                analysis["completed_tasks"],
                analysis["modified_files"],
                analysis["key_decisions"],
                analysis["session_context"]
            )

            # Generate structured summary
            summary = self.generate_structured_summary(
                session_goal=session_goal,
                completed_tasks=analysis["completed_tasks"],
                modified_files=analysis["modified_files"],
                key_decisions=analysis["key_decisions"],
                errors=analysis["errors"],
                session_context=analysis["session_context"],
                total_memories=len(memories)
            )

            # Store summary
            success, memory_id = await self.store_summary(summary)

            return success, summary

        except Exception as e:
            self.logger.error(f"Failed to generate and store summary: {e}")
            summary = self._generate_fallback_summary()
            return False, summary

    def _generate_fallback_summary(self) -> str:
        """Generate minimal fallback summary when memory extraction fails."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# 📋 Session Summary

**Ended**: {timestamp}

DevStream session completed. Unable to extract detailed summary from memory.

## 🚀 Next Steps
- Review recent changes
- Continue with pending tasks
"""


# Example usage
if __name__ == "__main__":
    async def test():
        manager = SessionSummaryManager()
        success, summary = await manager.generate_and_store_summary()
        print(f"Success: {success}")
        print(f"\nSummary:\n{summary}")

    asyncio.run(test())
