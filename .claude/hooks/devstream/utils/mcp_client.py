#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "aiohttp>=3.8.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
DevStream MCP Client - Direct DB + Context7 MCP Hybrid
======================================================

- Operazioni DevStream (memoria, task, checkpoint, piani) → `DevStreamDirectClient`
  per evitare il vecchio server MCP.
- Tool Context7 (`mcp__context7__*`) → MCP ibrido (direct client + fallback npx)
  attraverso `Context7HybridManager`, preservando il comportamento originario.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent))
from logger import get_devstream_logger
from path_validator import validate_db_path, PathValidationError
from direct_client import DevStreamDirectClient, DatabaseException
from context7_hybrid_manager import Context7HybridManager, Context7Error


class DevStreamMCPClient:
    """
    Client ibrido:
      • DevStream → Direct DB
      • Context7 → Hybrid Manager (MCP + Direct HTTP con fallback automatico)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        # Determina percorso DB (multi-progetto incluso)
        raw_path = db_path or os.getenv('DEVSTREAM_DB_PATH', 'data/devstream.db')
        project_root = os.getenv('DEVSTREAM_PROJECT_ROOT')

        try:
            self.db_path = validate_db_path(raw_path, project_root)
        except PathValidationError as exc:
            logger = get_devstream_logger('mcp_client')
            logger.logger.error(
                "Database path validation failed",
                extra={"raw_path": raw_path, "project_root": project_root, "error": str(exc)}
            )
            raise

        self.logger = get_devstream_logger('mcp_client')

        # Direct DB client per le API DevStream
        self.direct_client = DevStreamDirectClient(self.db_path)

        # Context7 hybrid manager (usa feature flag: direct/mcp/rollout)
        # Abilitiamo fallback MCP by default per resilienza
        self.context7_manager = Context7HybridManager(direct_enabled=None, mcp_fallback=True)

    # ------------------------------------------------------------------
    # Operazioni DevStream via Direct Client
    # ------------------------------------------------------------------
    async def store_memory(
        self,
        content: str,
        content_type: str,
        keywords: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.store_memory(
                content=content,
                content_type=content_type,
                keywords=keywords,
                session_id=session_id,
                hook_name="mcp_client_store"
            )
        except DatabaseException as exc:
            self.logger.logger.error(
                "store_memory_failed",
                extra={
                    "content_preview": content[:120],
                    "content_type": content_type,
                    "session_id": session_id,
                    "error": str(exc)
                }
            )
            return None

    async def search_memory(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.search_memory(
                query=query,
                content_type=content_type,
                limit=limit,
                hook_name="mcp_client_search"
            )
        except DatabaseException as exc:
            self.logger.logger.error(
                "search_memory_failed",
                extra={
                    "query": query[:120],
                    "content_type": content_type,
                    "limit": limit,
                    "error": str(exc)
                }
            )
            return None

    async def trigger_checkpoint(
        self,
        reason: str = "tool_trigger"
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.trigger_checkpoint(reason=reason)
        except DatabaseException as exc:
            self.logger.logger.error(
                "trigger_checkpoint_failed",
                extra={"reason": reason, "error": str(exc)}
            )
            return None

    async def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        priority: int,
        phase_name: str,
        project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.create_task(
                title=title,
                description=description,
                task_type=task_type,
                priority=priority,
                phase_name=phase_name,
                project=project
            )
        except DatabaseException as exc:
            self.logger.logger.error(
                "create_task_failed",
                extra={
                    "title": title,
                    "task_type": task_type,
                    "priority": priority,
                    "phase_name": phase_name,
                    "project": project,
                    "error": str(exc)
                }
            )
            return None

    async def list_tasks(
        self,
        status: Optional[str] = None,
        project: Optional[str] = None,
        priority: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.list_tasks(
                status=status,
                project=project,
                priority=priority
            )
        except DatabaseException as exc:
            self.logger.logger.error(
                "list_tasks_failed",
                extra={
                    "status": status,
                    "project": project,
                    "priority": priority,
                    "error": str(exc)
                }
            )
            return None

    async def update_task(
        self,
        task_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.direct_client.update_task(
                task_id=task_id,
                status=status,
                notes=notes
            )
        except DatabaseException as exc:
            self.logger.logger.error(
                "update_task_failed",
                extra={
                    "task_id": task_id,
                    "status": status,
                    "notes": notes,
                    "error": str(exc)
                }
            )
            return None

    async def create_implementation_plan(
        self,
        task_id: str,
        model_type: str,
        plan_content: str,
        plan_file_path: str,
        handoff_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        plan_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        plan_metadata = metadata.copy() if metadata else {}
        plan_metadata.update({
            "plan_file_path": plan_file_path,
            "model_type": model_type,
            "handoff_prompt_present": bool(handoff_prompt)
        })
        metadata_json = json.dumps(plan_metadata)

        try:
            with self.direct_client.connection_manager.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO implementation_plans (
                        id,
                        task_id,
                        title,
                        content,
                        model_type,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'draft', ?, ?)
                    """,
                    (
                        plan_id,
                        task_id,
                        Path(plan_file_path).name if plan_file_path else f"{task_id}.md",
                        plan_content,
                        model_type,
                        created_at,
                        created_at
                    )
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_memory (
                        id,
                        content,
                        content_type,
                        keywords,
                        session_id,
                        created_at,
                        updated_at,
                        access_count,
                        relevance_score,
                        source,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1.0, ?, ?)
                    """,
                    (
                        plan_id,
                        plan_content,
                        "implementation_plan",
                        json.dumps(["implementation-plan", model_type.lower()]),
                        "",
                        created_at,
                        created_at,
                        "implementation_plan_generator",
                        metadata_json
                    )
                )
                conn.commit()

            if handoff_prompt:
                await self.store_memory(
                    content=handoff_prompt,
                    content_type="handoff_prompt",
                    keywords=["handoff", "implementation-plan", task_id],
                    session_id=None
                )

            return {
                "success": True,
                "plan_id": plan_id,
                "plan_file_path": plan_file_path,
                "metadata": plan_metadata
            }
        except Exception as exc:
            self.logger.logger.error(
                "create_implementation_plan_failed",
                extra={
                    "task_id": task_id,
                    "model_type": model_type,
                    "plan_file_path": plan_file_path,
                    "error": str(exc)
                }
            )
            return None

    async def get_implementation_plan(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.direct_client.connection_manager.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        p.id,
                        p.content,
                        p.model_type,
                        p.status,
                        p.updated_at,
                        sm.metadata
                    FROM implementation_plans p
                    LEFT JOIN semantic_memory sm ON sm.id = p.id
                    WHERE p.task_id = ?
                    ORDER BY p.updated_at DESC
                    LIMIT 1
                    """,
                    (task_id,)
                )
                row = cursor.fetchone()

            if not row:
                return None

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            return {
                "plan_id": row["id"],
                "plan_content": row["content"],
                "model_type": row["model_type"],
                "status": row["status"],
                "metadata": metadata,
                "updated_at": row["updated_at"]
            }
        except Exception as exc:
            self.logger.logger.error(
                "get_implementation_plan_failed",
                extra={"task_id": task_id, "error": str(exc)}
            )
            return None

    # ------------------------------------------------------------------
    # Context7 via Hybrid Manager
    # ------------------------------------------------------------------
    async def _call_context7_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "mcp__context7__resolve-library-id":
                library_name = arguments.get("libraryName") or arguments.get("library_name")
                if not library_name:
                    return {"error": "libraryName is required"}
                library_id = await self.context7_manager.resolve_library_id(library_name)
                return {"library_id": library_id}

            if tool_name == "mcp__context7__get-library-docs":
                library_id = (
                    arguments.get("context7CompatibleLibraryID")
                    or arguments.get("library_id")
                    or arguments.get("libraryId")
                )
                if not library_id:
                    return {"content": "", "error": "context7CompatibleLibraryID is required"}
                topic = arguments.get("topic")
                tokens = arguments.get("tokens", 5000)
                docs = await self.context7_manager.get_library_docs(
                    library_id=library_id,
                    topic=topic,
                    tokens=int(tokens) if isinstance(tokens, (int, str)) else 5000
                )
                return {"content": docs, "library_id": library_id}

            return {"error": f"Unsupported Context7 tool: {tool_name}"}

        except Context7Error as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # MCP compatibility interface
    # ------------------------------------------------------------------
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name cannot be empty")
        if arguments is None:
            raise ValueError("arguments cannot be None (use {} for no arguments)")

        normalized = tool_name.split("__", 2)[-1]

        # DevStream operations handled via direct client
        if normalized in {
            "devstream_store_memory",
            "devstream_search_memory",
            "devstream_trigger_checkpoint",
            "devstream_create_task",
            "devstream_update_task",
            "devstream_list_tasks",
            "devstream_create_implementation_plan",
            "devstream_get_implementation_plan"
        }:
            return await self._handle_devstream_tool(normalized, arguments)

        # Context7 tools via hybrid manager
        if tool_name.startswith("mcp__context7__"):
            return await self._call_context7_tool(tool_name, arguments)

        # Unsupported tool
        return {"error": f"Unsupported MCP tool: {tool_name}"}

    async def _handle_devstream_tool(self, normalized_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if normalized_name == "devstream_store_memory":
            result = await self.store_memory(
                content=arguments.get("content", ""),
                content_type=arguments.get("content_type", "context"),
                keywords=arguments.get("keywords"),
                session_id=arguments.get("session_id")
            )
            return result or {"error": "store_memory failed"}

        if normalized_name == "devstream_search_memory":
            result = await self.search_memory(
                query=arguments.get("query", ""),
                content_type=arguments.get("content_type"),
                limit=arguments.get("limit", 10)
            )
            return result or {"error": "search_memory failed"}

        if normalized_name == "devstream_trigger_checkpoint":
            result = await self.trigger_checkpoint(arguments.get("reason", "tool_trigger"))
            return result or {"error": "trigger_checkpoint failed"}

        if normalized_name == "devstream_create_task":
            result = await self.create_task(
                title=arguments.get("title", ""),
                description=arguments.get("description", ""),
                task_type=arguments.get("task_type", "development"),
                priority=arguments.get("priority", 5),
                phase_name=arguments.get("phase_name", "unspecified"),
                project=arguments.get("project")
            )
            return result or {"error": "create_task failed"}

        if normalized_name == "devstream_update_task":
            result = await self.update_task(
                task_id=arguments.get("task_id", ""),
                status=arguments.get("status", "pending"),
                notes=arguments.get("notes")
            )
            return result or {"error": "update_task failed"}

        if normalized_name == "devstream_list_tasks":
            result = await self.list_tasks(
                status=arguments.get("status"),
                project=arguments.get("project"),
                priority=arguments.get("priority")
            )
            return result or {"error": "list_tasks failed"}

        if normalized_name == "devstream_create_implementation_plan":
            result = await self.create_implementation_plan(
                task_id=arguments.get("task_id", ""),
                model_type=arguments.get("model_type", "claude-sonnet-45"),
                plan_content=arguments.get("plan_content", ""),
                plan_file_path=arguments.get("plan_file_path", ""),
                handoff_prompt=arguments.get("handoff_prompt"),
                metadata=arguments.get("metadata")
            )
            return result or {"error": "create_implementation_plan failed"}

        if normalized_name == "devstream_get_implementation_plan":
            result = await self.get_implementation_plan(arguments.get("task_id", ""))
            return result or {"error": "implementation plan not found"}

        return {"error": f"Unsupported DevStream tool: {normalized_name}"}

    async def health_check(self) -> bool:
        try:
            return await self.direct_client.health_check()
        except Exception as exc:
            self.logger.logger.error("health_check_failed", extra={"error": str(exc)})
            return False

    async def shutdown(self) -> None:
        try:
            await self.context7_manager.close()
        except Exception:
            pass

    async def __aenter__(self) -> DevStreamMCPClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()


# Singleton per gli hook
_mcp_client: Optional[DevStreamMCPClient] = None


def get_mcp_client() -> DevStreamMCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = DevStreamMCPClient()
    return _mcp_client


# Convenience functions compatibili con le vecchie API ---------------------------------
async def store_memory_async(
    content: str,
    content_type: str,
    keywords: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    client = get_mcp_client()
    return await client.store_memory(content, content_type, keywords, session_id)


async def search_memory_async(
    query: str,
    content_type: Optional[str] = None,
    limit: int = 10
) -> Optional[Dict[str, Any]]:
    client = get_mcp_client()
    return await client.search_memory(query, content_type, limit)


async def trigger_checkpoint_async(reason: str = "tool_trigger") -> Optional[Dict[str, Any]]:
    client = get_mcp_client()
    return await client.trigger_checkpoint(reason)


if __name__ == "__main__":
    async def smoke_test() -> None:
        client = get_mcp_client()
        print("🧪 Health check:", await client.health_check())
    asyncio.run(smoke_test())
