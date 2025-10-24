#!/usr/bin/env python3
"""
DevStream SessionEnd Hook - Session Summary & Memory Storage
Lightweight session wrap-up executed on Claude Code session termination.

B2 Behavioral Refinement: Uses SessionSummaryManager for centralized summary logic.
"""

import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

# Logging setup
LOG_DIR = Path.home() / ".claude" / "logs" / "devstream"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "session_end.log"


def log(message: str) -> None:
    """Log message to file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass  # Silent fail on logging errors


async def extract_session_summary() -> str:
    """
    Extract structured summary using SessionSummaryManager.
    Delegates to centralized manager for consistency.

    Returns:
        Formatted summary string with session context
    """
    try:
        # Import SessionSummaryManager
        sys.path.insert(0, str(Path(__file__).parent.parent / 'sessions'))
        from session_summary_manager import SessionSummaryManager

        # Create manager and generate summary
        manager = SessionSummaryManager()
        success, summary = await manager.generate_and_store_summary()

        if success:
            log(f"✅ Session summary generated successfully")
        else:
            log(f"⚠️ Summary generated with warnings")

        return summary

    except Exception as e:
        log(f"❌ Failed to extract session summary: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return generate_fallback_summary()


def generate_structured_summary(completed_tasks, modified_files, key_decisions, errors, session_context=None,
                                session_start_time=None, total_memories=0) -> str:
    """
    Generate structured summary following Context7 best practices (MemoChat/EM-LLM patterns).

    Format follows episodic memory structure: observation → thoughts → action → result
    Optimized for session continuity and LLM retrieval.

    Args:
        completed_tasks: List of completed task descriptions
        modified_files: List of modified file paths
        key_decisions: List of key decision texts
        errors: List of error messages
        session_context: Optional list of session context snippets
        session_start_time: Optional start time for duration calculation
        total_memories: Total memories analyzed

    Returns:
        Formatted markdown summary optimized for session continuation
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_only = datetime.now().strftime("%Y-%m-%d")
    time_only = datetime.now().strftime("%H:%M")

    summary_parts = [
        "# 📋 Session Summary",
        f"\n**Session**: {date_only} ending at {time_only}",
    ]

    # Infer session goal from all available context
    session_goal = infer_session_goal_enhanced(completed_tasks, modified_files, key_decisions, session_context)

    # What Happened (narrative - MemoChat pattern)
    summary_parts.append(f"**Goal**: {session_goal}\n")

    narrative = generate_session_narrative(completed_tasks, modified_files, key_decisions, errors, session_context)
    if narrative:
        summary_parts.append("## 🎯 What Happened")
        summary_parts.append(narrative)
        summary_parts.append("")

    # Completed Work (actionable results)
    if completed_tasks:
        summary_parts.append("## ✅ Completed Work")
        for i, task in enumerate(completed_tasks[:6], 1):  # Top 6 tasks
            # Clean up task text
            task_clean = task.strip('- *✅').strip()
            if len(task_clean) > 5:
                summary_parts.append(f"{i}. {task_clean}")
        summary_parts.append("")

    # Files Modified (with key highlights)
    if modified_files:
        summary_parts.append("## 📁 Files Modified ({})".format(len(modified_files)))

        # Group files by type for better readability
        key_files = []
        other_files = []

        for file in sorted(modified_files)[:15]:
            if any(keyword in file.lower() for keyword in ['stop', 'session', 'summary', 'context', 'hook']):
                key_files.append(file)
            else:
                other_files.append(file)

        if key_files:
            summary_parts.append("**Key changes**:")
            for file in key_files[:5]:
                summary_parts.append(f"- `{file}`")

        if other_files and len(key_files) < 10:
            remaining_space = 10 - len(key_files)
            if other_files[:remaining_space]:
                if key_files:
                    summary_parts.append("\n**Other files**:")
                for file in other_files[:remaining_space]:
                    summary_parts.append(f"- `{file}`")

        if len(modified_files) > 10:
            summary_parts.append(f"\n_(and {len(modified_files) - 10} more files)_")
        summary_parts.append("")

    # Technical Decisions (processed correctly)
    if key_decisions:
        summary_parts.append("## 🔍 Technical Decisions")
        for i, decision in enumerate(key_decisions[:5], 1):
            # Clean and format decision text
            decision_clean = decision.strip()
            # Handle multi-line decisions
            if '\n' in decision_clean:
                lines = [l.strip() for l in decision_clean.split('\n') if l.strip()]
                summary_parts.append(f"{i}. **{lines[0]}**")
                if len(lines) > 1:
                    summary_parts.append(f"   {lines[1][:200]}")
            else:
                summary_parts.append(f"{i}. {decision_clean[:300]}")
        summary_parts.append("")

    # Known Issues (properly formatted)
    if errors:
        summary_parts.append("## ⚠️ Known Issues")
        for error in errors[:4]:
            # Clean error text
            error_clean = error.strip('- ').strip()
            if error_clean:
                summary_parts.append(f"- {error_clean[:200]}")
        summary_parts.append("")

    # Impact & Next Steps (Context7 pattern: what should happen next)
    summary_parts.append("## 🚀 Impact & Next Steps")

    # Calculate impact
    if completed_tasks or modified_files:
        impact = generate_impact_statement(completed_tasks, modified_files, key_decisions)
        summary_parts.append(f"**Impact**: {impact}\n")

    # Next steps
    summary_parts.append("**Next Session Should**:")
    next_steps = generate_smart_next_steps(completed_tasks, errors, session_goal)
    for step in next_steps:
        summary_parts.append(f"- {step}")
    summary_parts.append("")

    # Session Metrics (for retrieval and analysis)
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


def infer_session_goal_enhanced(completed_tasks, modified_files, key_decisions, session_context) -> str:
    """
    Enhanced session goal inference using multiple context sources.

    Args:
        completed_tasks: List of task descriptions
        modified_files: List of file paths
        key_decisions: List of decision texts
        session_context: List of context snippets

    Returns:
        Inferred goal description
    """
    # Combine all available context
    all_text = " ".join(completed_tasks + [str(c) for c in (session_context or [])]).lower()

    # Analyze file patterns
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


def generate_session_narrative(completed_tasks, modified_files, key_decisions, errors, session_context) -> str:
    """
    Generate a narrative summary of what happened (MemoChat pattern).

    Returns a 2-3 sentence narrative that tells the story of the session.
    """
    narratives = []

    # Opening: What was attempted
    if session_context:
        # Use first context as setup
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


def generate_impact_statement(completed_tasks, modified_files, key_decisions) -> str:
    """
    Generate an impact statement describing the significance of the session's work.
    """
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


def generate_smart_next_steps(completed_tasks, errors, session_goal) -> list:
    """
    Generate intelligent next steps based on session outcome.
    """
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


def infer_session_goal(completed_tasks, modified_files) -> str:
    """
    Legacy function kept for backward compatibility.
    Use infer_session_goal_enhanced() for new code.
    """
    return infer_session_goal_enhanced(completed_tasks, modified_files, [], None)


def generate_fallback_summary() -> str:
    """Generate minimal fallback summary when memory extraction fails."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# 📋 Session Summary

**Ended**: {timestamp}

DevStream session completed. Unable to extract detailed summary from memory.

## 🚀 Next Steps
- Review recent changes
- Continue with pending tasks
"""


async def store_session_end() -> tuple[bool, str]:
    """
    Extract and store session summary using SessionSummaryManager.
    Delegates all logic to centralized manager.

    Returns:
        Tuple of (success: bool, summary: str)
    """
    try:
        # Extract and store summary using manager
        summary = await extract_session_summary()
        return True, summary

    except Exception as e:
        log(f"❌ Failed to store session summary: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        fallback_summary = generate_fallback_summary()
        return False, fallback_summary


async def main_logic():
    """Main logic - silent execution with summary file creation."""
    # Read stdin (SessionEnd hook receives minimal input)
    session_id = "unknown"
    try:
        input_data = json.load(sys.stdin)
        session_id = input_data.get("session_id", "unknown")
        log(f"Session ID: {session_id}")
    except Exception as e:
        log(f"⚠️ Failed to parse stdin: {e}")

    # Store session end marker and get summary
    success, summary = await store_session_end()

    # Write summary to marker file for next SessionStart to display
    summary_file = Path.home() / ".claude" / "state" / "devstream_last_session.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(summary_file, "w") as f:
            f.write(summary)
        log(f"✅ Summary written to {summary_file}")
    except Exception as e:
        log(f"⚠️ Failed to write summary file: {e}")

    if success:
        log("✅ Session end hook completed successfully")
    else:
        log("⚠️ Session end hook completed with warnings")

    # Exit 0: SessionEnd hooks should be silent
    sys.exit(0)


def main():
    """Main entry point for SessionEnd hook following cchooks best practices."""
    # Diagnostic logging - write immediately to ensure we capture execution
    try:
        with open(LOG_FILE, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"[{datetime.now()}] 🔥 SESSIONEND HOOK CALLED!\n")
            f.write(f"[{datetime.now()}] Python: {sys.executable}\n")
            f.write(f"[{datetime.now()}] CWD: {Path.cwd()}\n")
            f.flush()  # Force write
    except Exception:
        pass

    log("SessionEnd hook triggered")

    try:
        # Run async main logic
        import asyncio
        asyncio.run(main_logic())
        # main_logic() exits via sys.exit(0) - this line never reached

    except Exception as e:
        log(f"💥 Unexpected error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        # Show error to user via stderr (cchooks pattern)
        print(f"⚠️  DevStream: Unexpected error: {e}", file=sys.stderr)
        # Exit code 1: non-blocking error, shows stderr to user
        sys.exit(1)


if __name__ == "__main__":
    main()