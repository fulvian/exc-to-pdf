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
DevStream UserPromptSubmit Hook - Automatic Memory Storage
Context7-compliant implementation per DevStream methodology.
"""

import json
import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path

# Import our common utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context

class UserPromptSubmitHook(DevStreamHookBase):
    """
    UserPromptSubmit hook for automatic DevStream memory storage.
    Implements Context7-validated patterns per research.
    """

    def __init__(self):
        super().__init__('user_prompt_submit')

    async def process_prompt(self, input_data: dict) -> None:
        """
        Process user prompt and store in DevStream memory.

        Args:
            input_data: JSON input from Claude Code
        """
        prompt = input_data.get('prompt', '')
        session_id = input_data.get('session_id', 'unknown')

        if not prompt.strip():
            self.logger.info("Empty prompt, skipping memory storage")
            return

        # Get project context
        project_context = get_project_context()

        # Determine content type based on prompt
        content_type = self.classify_prompt(prompt)

        # Store in DevStream memory via MCP
        await self.store_in_memory(
            content=prompt,
            content_type=content_type,
            session_id=session_id,
            context=project_context
        )

        # Inject DevStream context if relevant
        if self.should_inject_context(prompt):
            await self.inject_project_context(project_context)

    def classify_prompt(self, prompt: str) -> str:
        """
        Classify prompt type for memory storage.

        Args:
            prompt: User prompt text

        Returns:
            Content type for DevStream memory
        """
        prompt_lower = prompt.lower()

        # Code-related prompts
        if any(keyword in prompt_lower for keyword in [
            'implement', 'function', 'class', 'code', 'debug', 'fix',
            'refactor', 'optimize', 'test'
        ]):
            return 'code'

        # Task management prompts
        if any(keyword in prompt_lower for keyword in [
            'task', 'todo', 'plan', 'phase', 'milestone', 'sprint'
        ]):
            return 'context'

        # Documentation prompts
        if any(keyword in prompt_lower for keyword in [
            'document', 'explain', 'readme', 'guide', 'help'
        ]):
            return 'documentation'

        # Learning/research prompts
        if any(keyword in prompt_lower for keyword in [
            'learn', 'research', 'understand', 'analyze', 'study'
        ]):
            return 'learning'

        # Default to context
        return 'context'

    def should_inject_context(self, prompt: str) -> bool:
        """
        Determine if DevStream context should be injected.

        Args:
            prompt: User prompt text

        Returns:
            True if context injection needed
        """
        prompt_lower = prompt.lower()

        # Context injection triggers
        context_keywords = [
            'devstream', 'memory', 'task', 'hook', 'implementation',
            'metodologia', 'context7', 'research', 'plan'
        ]

        return any(keyword in prompt_lower for keyword in context_keywords)

    async def store_in_memory(
        self,
        content: str,
        content_type: str,
        session_id: str,
        context: dict
    ) -> None:
        """
        Store prompt in DevStream memory via MCP.

        Args:
            content: Prompt content
            content_type: Classified content type
            session_id: Claude session ID
            context: Project context
        """
        # Extract keywords for semantic search
        keywords = self.extract_keywords(content)

        # MCP call parameters
        mcp_params = {
            'content': f"USER PROMPT [{session_id}]: {content}",
            'content_type': content_type,
            'keywords': keywords
        }

        # Store via MCP (placeholder - actual MCP integration needed)
        response = await self.call_devstream_mcp(
            'devstream_store_memory',
            mcp_params
        )

        if response:
            self.logger.info(f"Stored prompt in memory: {content[:50]}...")
        else:
            self.logger.error("Failed to store prompt in memory")

    def extract_keywords(self, content: str) -> list:
        """
        Extract keywords from prompt content.

        Args:
            content: Prompt text

        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction (could be enhanced with NLP)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
        }

        words = content.lower().split()
        keywords = [
            word.strip('.,!?;:') for word in words
            if len(word) > 3 and word not in stop_words
        ]

        # Limit to 10 most relevant keywords
        return list(set(keywords))[:10]

    async def inject_project_context(self, context: dict) -> None:
        """
        Inject DevStream project context.

        Args:
            context: Project context dictionary
        """
        context_text = f"""
DevStream Project Context:
- Methodology: {context['methodology']}
- Memory System: {context['memory_system']}
- Task Management: {context['task_management']}
- Standards: Follow CLAUDE.md guidelines
- Hook System: Context7-compliant implementation
"""

        self.output_context(context_text.strip())

async def main():
    """Main hook execution following Context7 patterns."""
    hook = UserPromptSubmitHook()

    try:
        # Read JSON input from stdin (Context7 pattern)
        input_data = hook.read_stdin_json()

        # Process the prompt
        await hook.process_prompt(input_data)

        # Success exit
        hook.success_exit()

    except Exception as e:
        hook.error_exit(f"UserPromptSubmit hook failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())