#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "python-dotenv>=1.0.0",
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
# ]
# ///

"""
DevStream SessionStart Hook - Context Caricamento Iniziale e Task Detection
Context7-compliant session initialization con project context e memory loading.
"""

import json
import sys
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger

class SessionStartHook(DevStreamHookBase):
    """
    SessionStart hook per initialization di sessioni Claude Code con DevStream context.
    Implementa Context7-validated patterns per session setup.
    """

    def __init__(self):
        super().__init__('session_start')
        self.structured_logger = get_devstream_logger('session_start')
        self.start_time = time.time()

    async def process_session_start(self, input_data: dict) -> None:
        """
        Process session start e setup DevStream context.

        Args:
            input_data: JSON input from Claude Code SessionStart
        """
        self.structured_logger.log_hook_start(input_data, {"phase": "session_start"})

        try:
            session_id = input_data.get('session_id', 'unknown')
            cwd = input_data.get('cwd', os.getcwd())

            self.logger.info(f"Starting DevStream session: {session_id}")

            # Check if this is a DevStream project
            is_devstream_project = await self.detect_devstream_project(cwd)

            if not is_devstream_project:
                self.logger.info("Not a DevStream project - minimal initialization")
                self.success_exit()
                return

            # Load project context
            project_context = await self.load_project_context(cwd)

            # Load recent session memory
            session_memory = await self.load_session_memory(session_id)

            # Check for active tasks
            active_tasks = await self.check_active_tasks()

            # Generate session context
            session_context = await self.generate_session_context(
                session_id,
                project_context,
                session_memory,
                active_tasks
            )

            # Inject initial context
            if session_context:
                self.output_context(session_context)

            # Log session initialization
            self.structured_logger.log_context_injection(
                context_type="session_initialization",
                content_size=len(session_context) if session_context else 0,
                keywords=["session-start", "devstream-project", "initialization"]
            )

            # Store session start in memory
            await self.store_session_start(session_id, project_context)

            # Log performance metrics
            execution_time = (time.time() - self.start_time) * 1000
            self.structured_logger.log_performance_metrics(execution_time)

            self.logger.info(f"DevStream session initialized: {session_id}")

        except Exception as e:
            self.structured_logger.log_hook_error(e, {"session_id": session_id})
            raise

    async def detect_devstream_project(self, cwd: str) -> bool:
        """
        Detect if current directory is a DevStream project.

        Args:
            cwd: Current working directory

        Returns:
            True if DevStream project detected
        """
        cwd_path = Path(cwd)

        # Check for DevStream indicators
        devstream_indicators = [
            # Direct DevStream project
            cwd_path.name == 'devstream',
            # DevStream database
            (cwd_path / 'data' / 'devstream.db').exists(),
            # DevStream MCP server
            (cwd_path / 'mcp-devstream-server').exists(),
            # DevStream hooks
            (cwd_path / '.claude' / 'hooks' / 'devstream').exists(),
            # DevStream memory system
            (cwd_path / 'src' / 'devstream' / 'memory').exists(),
        ]

        # Check for DevStream references in CLAUDE.md
        claude_md = cwd_path / 'CLAUDE.md'
        if claude_md.exists():
            try:
                content = claude_md.read_text(encoding='utf-8')
                if any(keyword in content.lower() for keyword in [
                    'devstream', 'memoria semantica', 'mcp-devstream', 'hook system'
                ]):
                    devstream_indicators.append(True)
            except Exception:
                pass

        return any(devstream_indicators)

    async def load_project_context(self, cwd: str) -> Dict[str, Any]:
        """
        Load comprehensive project context.

        Args:
            cwd: Current working directory

        Returns:
            Project context dictionary
        """
        context = get_project_context()
        cwd_path = Path(cwd)

        # Add project-specific information
        context.update({
            "cwd": cwd,
            "project_name": cwd_path.name,
            "is_devstream_project": True,
        })

        # Load CLAUDE.md if exists
        claude_md = cwd_path / 'CLAUDE.md'
        if claude_md.exists():
            try:
                claude_content = claude_md.read_text(encoding='utf-8')
                context["claude_md_size"] = len(claude_content)
                context["has_claude_standards"] = True

                # Extract key sections
                context["claude_methodology"] = self.extract_methodology(claude_content)
            except Exception as e:
                self.logger.warning(f"Failed to read CLAUDE.md: {e}")

        # Check for recent development activity
        context["recent_commits"] = await self.get_recent_git_activity(cwd_path)

        # Check for active development indicators
        context["active_development"] = await self.assess_development_activity(cwd_path)

        return context

    def extract_methodology(self, claude_content: str) -> str:
        """
        Extract methodology from CLAUDE.md content.

        Args:
            claude_content: CLAUDE.md content

        Returns:
            Extracted methodology summary
        """
        # Look for methodology sections
        lines = claude_content.split('\n')
        methodology_lines = []

        in_methodology = False
        for line in lines:
            if 'metodologia' in line.lower() or 'methodology' in line.lower():
                in_methodology = True
                methodology_lines.append(line.strip())
            elif in_methodology and line.strip().startswith('#'):
                break
            elif in_methodology:
                methodology_lines.append(line.strip())

        return '\n'.join(methodology_lines[:10])  # First 10 lines

    async def get_recent_git_activity(self, cwd_path: Path) -> Dict[str, Any]:
        """
        Get recent git activity information.

        Args:
            cwd_path: Project directory path

        Returns:
            Git activity summary
        """
        git_info = {"has_git": False}

        if (cwd_path / '.git').exists():
            git_info["has_git"] = True
            # In real implementation, would use git commands
            git_info["recent_activity"] = "Active development detected"

        return git_info

    async def assess_development_activity(self, cwd_path: Path) -> Dict[str, Any]:
        """
        Assess current development activity.

        Args:
            cwd_path: Project directory path

        Returns:
            Development activity assessment
        """
        activity = {"active_areas": []}

        # Check for recent file modifications (simplified)
        python_files = list(cwd_path.glob('**/*.py'))[:10]
        for py_file in python_files:
            try:
                stat = py_file.stat()
                # Files modified in last 24 hours
                if (time.time() - stat.st_mtime) < 86400:
                    activity["active_areas"].append(str(py_file.relative_to(cwd_path)))
            except Exception:
                pass

        activity["has_recent_activity"] = len(activity["active_areas"]) > 0

        return activity

    async def load_session_memory(self, session_id: str) -> Optional[str]:
        """
        Load recent memory from previous sessions.

        Args:
            session_id: Current session ID

        Returns:
            Session memory context or None
        """
        # Search for recent session memories
        search_params = {
            'query': f'session devstream memory context',
            'limit': 3,
            'content_type': 'context'
        }

        search_response = await self.call_devstream_mcp(
            'devstream_search_memory',
            search_params
        )

        if search_response:
            # Extract relevant session memories
            return self.format_session_memories(search_response)

        return None

    def format_session_memories(self, search_response: Dict[str, Any]) -> str:
        """
        Format session memories for context.

        Args:
            search_response: MCP search response

        Returns:
            Formatted session memory context
        """
        # Extract and format memories (simplified)
        content = search_response.get('content', [])
        if content and len(content) > 0:
            text_content = content[0].get('text', '')
            if text_content:
                return f"📚 Recent Session Context:\n{text_content[:300]}...\n"

        return ""

    async def check_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Check for active DevStream tasks.

        Returns:
            List of active tasks
        """
        # Query DevStream tasks via MCP (placeholder)
        # In real implementation would call devstream_list_tasks
        active_tasks = []

        try:
            # Simulate task check
            active_tasks.append({
                "id": "hook-system-implementation",
                "title": "Hook System Implementation",
                "status": "in_progress",
                "priority": 9
            })
        except Exception as e:
            self.logger.warning(f"Failed to check active tasks: {e}")

        return active_tasks

    async def generate_session_context(
        self,
        session_id: str,
        project_context: Dict[str, Any],
        session_memory: Optional[str],
        active_tasks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive session context.

        Args:
            session_id: Session ID
            project_context: Project context
            session_memory: Session memory
            active_tasks: Active tasks

        Returns:
            Complete session context string
        """
        context_parts = [
            "🚀 DevStream Session Started",
            f"📝 Session: {session_id}",
            f"🏗️ Project: {project_context.get('project_name', 'DevStream')}",
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "🎯 DevStream Methodology Active:",
            "- Research-Driven Development con Context7",
            "- Memory system semantico con sqlite-vec",
            "- Task management con MCP integration",
            "- Hook system per automation completa",
            ""
        ]

        # Add project-specific context
        if project_context.get('has_claude_standards'):
            context_parts.extend([
                "📋 Project Standards:",
                "- Segui CLAUDE.md guidelines",
                "- Context7-compliant implementation",
                "- Structured logging obbligatorio",
                ""
            ])

        # Add active tasks context
        if active_tasks:
            context_parts.extend([
                "📋 Active Tasks:",
            ])
            for task in active_tasks[:3]:  # Max 3 tasks
                status_emoji = "🔄" if task["status"] == "in_progress" else "⏳"
                context_parts.append(
                    f"{status_emoji} {task['title']} (Priority: {task.get('priority', 5)})"
                )
            context_parts.append("")

        # Add session memory if available
        if session_memory:
            context_parts.extend([
                session_memory,
                ""
            ])

        # Add development tips
        context_parts.extend([
            "💡 Quick Tips:",
            "- Use TodoWrite per task tracking granulare",
            "- Store progress in DevStream memory",
            "- Follow Context7 best practices",
            "- Test hook implementations thoroughly",
            "",
            "---"
        ])

        return '\n'.join(context_parts)

    async def store_session_start(
        self,
        session_id: str,
        project_context: Dict[str, Any]
    ) -> None:
        """
        Store session start in memory.

        Args:
            session_id: Session ID
            project_context: Project context
        """
        memory_content = (
            f"SESSION START [{session_id}]: DevStream project session initialized. "
            f"Project: {project_context.get('project_name', 'DevStream')}, "
            f"Methodology: Research-Driven Development, "
            f"Active development: {project_context.get('active_development', {}).get('has_recent_activity', False)}"
        )

        # Store via MCP
        await self.call_devstream_mcp(
            'devstream_store_memory',
            {
                'content': memory_content,
                'content_type': 'context',
                'keywords': ['session-start', 'devstream', 'initialization', session_id[:8]]
            }
        )

        # Log memory operation
        self.structured_logger.log_memory_operation(
            operation="store",
            content_type="context",
            content_size=len(memory_content),
            keywords=['session-start', 'initialization']
        )

async def main():
    """Main hook execution following Context7 patterns."""
    hook = SessionStartHook()

    try:
        # Read JSON input from stdin (Context7 pattern)
        input_data = hook.read_stdin_json()

        # Process session start
        await hook.process_session_start(input_data)

        # Success exit
        hook.success_exit()

    except Exception as e:
        hook.error_exit(f"SessionStart hook failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())