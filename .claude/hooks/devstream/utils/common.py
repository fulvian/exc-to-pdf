#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "python-dotenv>=1.0.0",
#     "aiohttp>=3.8.0",
# ]
# ///

"""
DevStream Hook Utilities - Common Functions
Context7-compliant UV single-file script template.
"""

import json
import sys
import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from mcp_client import get_mcp_client
import aiohttp

# Load environment variables
load_dotenv()

# Import project locator functions (Dependency Injection pattern)
try:
    from .project_locator import get_project_root, get_database_path, is_devstream_available
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent))
    from project_locator import get_project_root, get_database_path, is_devstream_available

class DevStreamHookBase:
    """
    Base class for DevStream hooks following Context7 patterns.
    Provides common functionality per hook execution.
    """

    def __init__(self, hook_type: str):
        """
        Initialize hook base.
        
        Args:
            hook_type: Type of hook (e.g., "PreToolUse", "PostToolUse")
            
        Raises:
            PathValidationError: If database path validation fails
        """
        self.hook_type = hook_type
        
        # Import path validator
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from path_validator import validate_db_path, PathValidationError
        
        # Get path from environment or default
        raw_path = os.getenv(
            'DEVSTREAM_DB_PATH',
            'data/devstream.db'  # Relative path (project-root relative)
        )
        
        # SECURITY: Validate database path
        try:
            self.devstream_db_path = validate_db_path(raw_path)
        except PathValidationError as e:
            # Setup basic logging for error reporting
            self.setup_logging()
            self.logger.error(
                f"Database path validation failed: {e}",
                extra={"raw_path": raw_path, "hook_type": hook_type}
            )
            raise
        
        self.setup_logging()


    def setup_logging(self) -> None:
        """Setup structured logging per DevStream hooks."""
        log_dir = Path.home() / '.claude' / 'logs' / 'devstream'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'{self.hook_type}.log'),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger(f'devstream.{self.hook_type}')

    def read_stdin_json(self) -> Dict[str, Any]:
        """
        Read JSON input from stdin (Context7 pattern).

        Returns:
            Parsed JSON data from Claude Code

        Raises:
            SystemExit: If JSON parsing fails
        """
        try:
            input_data = json.loads(sys.stdin.read())
            self.logger.info(f"Received input: {json.dumps(input_data, indent=2)}")
            return input_data
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON input: {e}")
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            sys.exit(1)

    async def call_devstream_mcp(
        self,
        tool: str,
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Call DevStream MCP server tools.

        Args:
            tool: MCP tool name (e.g., 'devstream_store_memory')
            parameters: Tool parameters

        Returns:
            MCP response or None on failure
        """
        try:
            client = get_mcp_client()
            result = await client.call_tool(tool, parameters)
            self.logger.info(f"DevStream direct call completed: {tool}")
            return result
        except Exception as e:
            self.logger.error(f"Direct DevStream call failed: {e}")
            return None

    def output_context(self, context: str) -> None:
        """
        Output context to Claude Code (Context7 pattern).

        Args:
            context: Context string to inject
        """
        print(context)
        self.logger.info(f"Context injected: {context[:100]}...")

    def output_json(self, data: Dict[str, Any]) -> None:
        """
        Output JSON response (Context7 pattern).

        Args:
            data: JSON response data
        """
        print(json.dumps(data))
        self.logger.info(f"JSON output: {json.dumps(data)}")

    def block_with_reason(self, reason: str) -> None:
        """
        Block operation with reason (Context7 pattern).

        Args:
            reason: Reason for blocking
        """
        self.output_json({
            "decision": "block",
            "reason": reason
        })
        self.logger.warning(f"Blocked operation: {reason}")
        sys.exit(0)

    def approve_with_reason(self, reason: str) -> None:
        """
        Approve operation with reason (Context7 pattern).

        Args:
            reason: Reason for approval
        """
        self.output_json({
            "decision": "approve",
            "reason": reason
        })
        self.logger.info(f"Approved operation: {reason}")
        sys.exit(0)

    def success_exit(self) -> None:
        """Success exit (Context7 pattern)."""
        self.logger.info(f"{self.hook_type} hook completed successfully")
        sys.exit(0)

    def error_exit(self, error: str) -> None:
        """Error exit (Context7 pattern)."""
        self.logger.error(f"{self.hook_type} hook failed: {error}")
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

def get_project_context() -> Dict[str, Any]:
    """
    Get DevStream project context.

    Returns:
        Project context dictionary
    """
    context = {
        "project": "DevStream Development",
        "methodology": "Research-Driven Development con Context7",
        "memory_system": "Semantic memory con sqlite-vec",
        "task_management": "DevStream MCP integration",
        "timestamp": datetime.now().isoformat()
    }

    # Check for CLAUDE.md
    claude_md = Path.cwd() / "CLAUDE.md"
    if claude_md.exists():
        context["standards_file"] = str(claude_md)

    # Check for git repo
    git_dir = Path.cwd() / ".git"
    if git_dir.exists():
        context["git_repo"] = True

    return context


def classify_query_type(query: str) -> str:
    """
    Classify user query type for processing (Context7 pattern).

    Args:
        query: User query string

    Returns:
        Query classification: 'research', 'implementation', 'debugging', 'documentation', 'general'
    """
    import re

    query_lower = query.lower()

    # Research patterns
    research_keywords = ['how to', 'best practice', 'research', 'learn about', 'explain', 'what is', 'documentation']
    if any(keyword in query_lower for keyword in research_keywords):
        return 'research'

    # Implementation patterns
    implementation_keywords = ['implement', 'create', 'build', 'write', 'develop', 'code', 'add feature']
    if any(keyword in query_lower for keyword in implementation_keywords):
        return 'implementation'

    # Debugging patterns
    debugging_keywords = ['fix', 'debug', 'error', 'issue', 'problem', 'broken', 'not working']
    if any(keyword in query_lower for keyword in debugging_keywords):
        return 'debugging'

    # Documentation patterns
    documentation_keywords = ['document', 'readme', 'guide', 'manual', 'docs', 'explain code']
    if any(keyword in query_lower for keyword in documentation_keywords):
        return 'documentation'

    return 'general'


def generate_session_id() -> str:
    """
    Generate unique session identifier (Context7 pattern).

    Returns:
        Unique session ID
    """
    import uuid
    return f"sess-{uuid.uuid4().hex[:12]}"


def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for memory indexing (Context7 pattern).

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    import re

    # Extract words and filter common words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

    # Filter common words and short words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'what', 'how',
        'why', 'when', 'where', 'who', 'which', 'that', 'this', 'these', 'those',
        'i', 'you', 'we', 'they', 'me', 'us', 'them', 'a', 'an'
    }

    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    # Remove duplicates and limit
    return list(dict.fromkeys(keywords))[:max_keywords]

if __name__ == "__main__":
    # Template test when run directly
    print("DevStream Hook Utilities - Template Test")
    print(f"Project Context: {get_project_context()}")
