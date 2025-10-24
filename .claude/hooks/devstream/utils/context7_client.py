#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp>=3.8.0",
# ]
# ///

"""
DevStream Context7 Client - MCP Wrapper per Context7 Integration
Context7-compliant library documentation retrieval con fallback robusto.
"""

import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class Context7Result:
    """Context7 search result."""
    library_id: str
    docs: str
    snippets: List[str]
    success: bool
    error: Optional[str] = None


class Context7Client:
    """
    Client wrapper for Context7 MCP integration.
    Provides library resolution, documentation retrieval, and graceful fallback.
    """

    def __init__(self, mcp_client: Optional[Any]):
        """
        Initialize Context7 client.

        Args:
            mcp_client: MCP client instance (optional)
        """
        self.mcp_client = mcp_client
        self.enabled = mcp_client is not None
        self.disabled_reason = None if self.enabled else "MCP client unavailable"

    async def should_trigger_context7(self, query: str) -> bool:
        """
        Detect if query would benefit from Context7 research.

        Args:
            query: User query or context

        Returns:
            True if Context7 should be triggered
        """
        if not self.enabled:
            return False

        triggers = [
            r"how to.*(?:implement|use|setup|configure)",
            r"best practice",
            r"(?:example|sample).*(?:code|implementation)",
            r"documentation.*(?:for|about)",
            # Common libraries/frameworks
            r"\b(react|vue|angular|django|fastapi|flask|nextjs|nuxt|svelte)\b",
            r"\b(pytest|jest|vitest|mocha|cypress)\b",
            r"\b(sqlalchemy|prisma|mongoose|sequelize)\b",
            r"\b(numpy|pandas|matplotlib|scikit-learn)\b",
        ]

        return any(re.search(pattern, query, re.IGNORECASE) for pattern in triggers)

    def extract_library_name(self, query: str) -> Optional[str]:
        """
        Extract library/framework name from query.

        Args:
            query: User query

        Returns:
            Library name or None
        """
        # Common patterns for library mentions
        patterns = [
            r"(?:using|with|for|in)\s+(\w+)",
            r"(\w+)\s+(?:library|framework|package)",
            r"import\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_topic(self, query: str) -> Optional[str]:
        """
        Extract specific topic from query.

        Args:
            query: User query

        Returns:
            Topic string or None
        """
        # Extract key concepts
        topics = []

        # Look for action words
        actions = re.findall(
            r"\b(hook|routing|state|component|api|auth|database|query|test)\w*\b",
            query,
            re.IGNORECASE
        )
        topics.extend(actions)

        if topics:
            return " ".join(topics[:3])  # Max 3 topics

        return None

    async def resolve_library(self, library_name: str) -> Optional[str]:
        """
        Resolve library name to Context7 library ID.

        Args:
            library_name: Library/framework name

        Returns:
            Context7 library ID or None
        """
        if not self.enabled or self.mcp_client is None:
            return None

        try:
            result = await self.mcp_client.call_tool(
                "mcp__context7__resolve-library-id",
                {"libraryName": library_name}
            )

            # Parse result to extract library ID
            # Result format varies, typically contains library ID in text
            if result and isinstance(result, dict):
                # Try to extract library ID from result
                content = str(result)
                # Look for /org/project pattern
                match = re.search(r'/[\w-]+/[\w-]+', content)
                if match:
                    return match.group(0)

            return None

        except Exception as e:
            return None

    async def get_documentation(
        self,
        library_id: str,
        topic: Optional[str] = None,
        tokens: int = 5000  # CRITICAL: Match CLAUDE.md spec (5000 tokens for Context7)
    ) -> Optional[str]:
        """
        Get documentation for library from Context7.

        Args:
            library_id: Context7 library ID (format: /org/project)
            topic: Optional specific topic
            tokens: Maximum tokens to retrieve

        Returns:
            Documentation string or None
        """
        if not self.enabled or self.mcp_client is None:
            return None

        try:
            params = {
                "context7CompatibleLibraryID": library_id,
                "tokens": tokens
            }

            if topic:
                params["topic"] = topic

            result = await self.mcp_client.call_tool(
                "mcp__context7__get-library-docs",
                params
            )

            if result and isinstance(result, dict):
                # Extract documentation content
                return str(result.get("content", ""))

            return None

        except Exception as e:
            return None

    async def search_and_retrieve(
        self,
        query: str,
        library_name: Optional[str] = None
    ) -> Context7Result:
        """
        Search Context7 and retrieve documentation (main method).

        Args:
            query: User query or context
            library_name: Optional explicit library name

        Returns:
            Context7Result with documentation or error
        """
        if not self.enabled:
            return Context7Result(
                library_id="",
                docs="",
                snippets=[],
                success=False,
                error=self.disabled_reason or "Context7 disabled"
            )

        # Extract library if not provided
        if not library_name:
            library_name = self.extract_library_name(query)

        if not library_name:
            return Context7Result(
                library_id="",
                docs="",
                snippets=[],
                success=False,
                error="No library detected"
            )

        # Resolve library
        library_id = await self.resolve_library(library_name)
        if not library_id:
            return Context7Result(
                library_id="",
                docs="",
                snippets=[],
                success=False,
                error=f"Library '{library_name}' not found"
            )

        # Get documentation
        topic = self.extract_topic(query)
        docs = await self.get_documentation(library_id, topic)

        if not docs:
            return Context7Result(
                library_id=library_id,
                docs="",
                snippets=[],
                success=False,
                error="Documentation retrieval failed"
            )

        return Context7Result(
            library_id=library_id,
            docs=docs,
            snippets=[],  # TODO: Extract code snippets from docs
            success=True
        )

    def format_docs_for_context(self, result: Context7Result) -> str:
        """
        Format Context7 documentation for Claude context injection.

        Args:
            result: Context7 result

        Returns:
            Formatted context string
        """
        if not result.success or not result.docs:
            return ""

        # Format with clear headers
        formatted = f"""
# Context7 Documentation: {result.library_id}

{result.docs}

---
*Source: Context7 Library Documentation*
"""
        return formatted.strip()


# Utility function for quick access
async def quick_context7_search(
    mcp_client: Any,
    query: str
) -> Optional[str]:
    """
    Quick Context7 search helper.

    Args:
        mcp_client: MCP client instance
        query: Search query

    Returns:
        Formatted documentation or None
    """
    client = Context7Client(mcp_client)
    result = await client.search_and_retrieve(query)

    if result.success:
        return client.format_docs_for_context(result)

    return None
