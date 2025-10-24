#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cchooks>=0.1.4",
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "watchdog>=3.0.0",
#     "sqlite-utils>=3.36.0",
#     "aiofiles>=23.0.0",
#     "ollama>=0.1.0",
#     "sqlite-vec>=0.1.0",
# ]
# ///

"""
DevStream RealTimeDataCapture - FASE 1 Memory Vector Enhancement

Implements Context7 Watchdog pattern for real-time file modification capture.
Filters .py, .md, .ts, .tsx files and stores specific file context instead
of generic checkpoints.

Context7 Pattern Implementation:
- Uses Observer pattern with FileSystemEventHandler
- Filters by file extensions and paths
- Debounces rapid file changes
- Stores real file modifications in semantic memory
- Maintains session-specific context tracking
"""

import asyncio
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime
from dataclasses import dataclass

# Watchdog imports (Context7 pattern: use PollingObserver on macOS to prevent kernel panics)
# See: https://github.com/gorakhargosh/watchdog/blob/master/README.rst
# FSEvents on macOS can cause segmentation faults and kernel panics
import platform
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Add parent directories to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from devstream_base import DevStreamHookBase
from unified_client import get_unified_client
from debouncer import debounce
from logger import get_devstream_logger

# Import crash prevention (Context7 best practice integration)
try:
    from monitoring.crash_prevention import get_crash_monitor, should_disable_file_monitoring
except ImportError:
    # Fallback if crash prevention module not available
    def get_crash_monitor():
        return None
    def should_disable_file_monitoring():
        return False

# Initialize structured logging
logger = get_devstream_logger("real_time_capture").logger


@dataclass
class FileModification:
    """
    Represents a file modification event with metadata.

    Context7 Pattern: Structured data for real-time capture.
    """
    file_path: str
    event_type: str  # created, modified, deleted, moved
    timestamp: datetime
    file_size: Optional[int] = None
    content_preview: Optional[str] = None
    session_id: Optional[str] = None


class RealTimeFileEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for real-time file system monitoring.

    Context7 Pattern: Subclass FileSystemEventHandler with filtering logic.
    """

    def __init__(self, capture_instance: 'RealTimeDataCapture'):
        """
        Initialize the event handler.

        Args:
            capture_instance: Reference to RealTimeDataCapture instance
        """
        super().__init__()
        self.capture = capture_instance

    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handle file modification events.

        Args:
            event: File system event from Watchdog
        """
        if not event.is_directory:
            self.capture._handle_file_event("modified", event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation events.

        Args:
            event: File system event from Watchdog
        """
        if not event.is_directory:
            self.capture._handle_file_event("created", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        Handle file deletion events.

        Args:
            event: File system event from Watchdog
        """
        if not event.is_directory:
            self.capture._handle_file_event("deleted", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """
        Handle file move/rename events.

        Args:
            event: File system event from Watchdog
        """
        if not event.is_directory:
            self.capture._handle_file_event("moved", event.dest_path or event.src_path)


class RealTimeDataCapture:
    """
    Real-time file modification capture using Context7 Watchdog patterns.

    FASE 1 Enhancement: Replaces generic "Task Checkpoint" messages with
    real file modifications and session-specific data.

    Features:
    - Real-time file monitoring with Observer pattern
    - File type filtering (.py, .md, .ts, .tsx)
    - Path exclusion logic
    - Debounced event processing
    - Session-specific context storage
    - Graceful degradation on errors
    """

    # File extensions to monitor (Context7 pattern: specific filtering)
    MONITORED_EXTENSIONS = {'.py', '.md', '.ts', '.tsx'}

    # Paths to exclude from monitoring
    EXCLUDED_PATHS = {
        '.git', 'node_modules', '.venv', '.devstream', '__pycache__',
        'dist', 'build', '.next', 'coverage', '.pytest_cache', '.mypy_cache'
    }

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize real-time data capture.

        Args:
            project_root: Root directory to monitor (defaults to current working directory)
        """
        self.base = DevStreamHookBase("real_time_capture")
        self.unified_client = get_unified_client()

        # Project configuration
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.db_path = str(self.project_root / 'data' / 'devstream.db')

        # Watchdog observer (Context7 best practice: use PollingObserver on macOS)
        # Prevents kernel panics and segmentation faults with FSEvents
        if platform.system() == 'Darwin':  # macOS
            self.observer: Optional[PollingObserver] = None
            self.observer_type = 'polling'  # Safer but less responsive
            logger.info("Using PollingObserver on macOS (prevents FSEvents kernel panics)")
        else:
            self.observer: Optional[Observer] = None
            self.observer_type = 'native'   # Use native observer on other platforms
            logger.info("Using native Observer")

        self.event_handler = RealTimeFileEventHandler(self)

        # Debounced event tracking (Context7 pattern: reduce noise)
        self._debounced_events: Dict[str, float] = {}

        # Session tracking
        self.current_session_id: Optional[str] = None
        self.monitored_files: Set[str] = set()

        # State management
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # Crash prevention (Context7 integration)
        self.crash_monitor = get_crash_monitor()
        if self.crash_monitor:
            logger.info("Crash prevention monitor integrated")

        logger.info("RealTimeDataCapture initialized",
                   project_root=str(self.project_root),
                   monitored_extensions=list(self.MONITORED_EXTENSIONS),
                   observer_type=self.observer_type)

    def _should_monitor_file(self, file_path: str) -> bool:
        """
        Determine if file should be monitored based on extension and path.

        Context7 Pattern: Multi-dimensional filtering to reduce noise.

        Args:
            file_path: Path to file to check

        Returns:
            True if file should be monitored, False otherwise
        """
        file_path_obj = Path(file_path)

        # Check file extension
        if file_path_obj.suffix.lower() not in self.MONITORED_EXTENSIONS:
            return False

        # Check excluded paths
        path_parts = file_path_obj.parts
        for excluded in self.EXCLUDED_PATHS:
            if excluded in path_parts:
                return False

        return True

    async def _get_current_session_id(self) -> Optional[str]:
        """
        Get current active session ID from work_sessions table.

        Returns:
            Current session ID if found, None otherwise
        """
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT id FROM work_sessions
                    WHERE status = 'active'
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        session_id = row[0]
                        logger.debug("Active session found", session_id=session_id[:8])
                        return session_id

            logger.debug("No active session found")
            return None

        except Exception as e:
            logger.error("Failed to get session ID", error=str(e))
            return None

    def _extract_content_preview(self, file_path: str, max_length: int = 300) -> Optional[str]:
        """
        Extract content preview from file.

        Context7 Pattern: Content preview for memory storage.

        Args:
            file_path: Path to file
            max_length: Maximum preview length

        Returns:
            Content preview or None if file unreadable
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return None

            # Read file content (with encoding handling)
            with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) <= max_length:
                return content

            # Try to break at sentence/line boundary
            preview = content[:max_length]
            last_period = preview.rfind('.')
            last_newline = preview.rfind('\n')

            break_point = max(last_period, last_newline)
            if break_point > max_length * 0.7:  # At least 70% of max length
                return preview[:break_point + 1].strip()

            return preview.strip() + "..."

        except Exception as e:
            logger.debug("Failed to extract content preview",
                        file_path=file_path, error=str(e))
            return None

    def _extract_keywords(self, file_path: str) -> List[str]:
        """
        Extract keywords from file path and content.

        Context7 Pattern: Keyword extraction for semantic search.

        Args:
            file_path: Path to file

        Returns:
            List of keywords
        """
        keywords = []
        file_path_obj = Path(file_path)

        # Add file name without extension
        file_name = file_path_obj.stem
        keywords.append(file_name)

        # Add parent directory if relevant
        parent = file_path_obj.parent.name
        if parent and parent not in ['.', '..']:
            keywords.append(parent)

        # Add language/framework from file extension
        ext = file_path_obj.suffix.lower()
        lang_map = {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'react',
            '.md': 'documentation'
        }
        if ext in lang_map:
            keywords.append(lang_map[ext])

        # Add real-time capture tags
        keywords.extend(["realtime", "modification", "autocapture"])

        return keywords

    async def _store_file_modification(self, modification: FileModification) -> Optional[str]:
        """
        Store file modification in DevStream memory.

        Context7 Pattern: Store specific file context instead of generic checkpoints.

        Args:
            modification: File modification data

        Returns:
            Memory ID if storage successful, None otherwise
        """
        try:
            # Build memory content
            memory_content = f"""# Real-time File Modification

**File**: {modification.file_path}
**Event**: {modification.event_type}
**Timestamp**: {modification.timestamp.isoformat()}
**Session**: {modification.session_id[:8] if modification.session_id else 'unknown'}

## Content Preview

{modification.content_preview or 'No content available'}
"""

            # Extract keywords
            keywords = self._extract_keywords(modification.file_path)
            if modification.session_id:
                keywords.append(f"session:{modification.session_id[:8]}")

            logger.debug("Storing real-time modification",
                        file_path=modification.file_path,
                        event_type=modification.event_type,
                        keywords_count=len(keywords))

            # Store via unified client
            result = await self.unified_client.store_memory(
                content=memory_content,
                content_type="code",
                keywords=keywords
            )

            if result and isinstance(result, dict):
                memory_id = result.get('memory_id')
                if memory_id:
                    logger.info("Real-time modification stored",
                               file_path=Path(modification.file_path).name,
                               memory_id=memory_id[:8])
                    return memory_id

            logger.warning("Failed to store real-time modification",
                          file_path=modification.file_path)
            return None

        except Exception as e:
            logger.error("Error storing file modification",
                        file_path=modification.file_path, error=str(e))
            return None

    def _handle_file_event(self, event_type: str, file_path: str) -> None:
        """
        Handle file system event with debouncing.

        Context7 Pattern: Debounced event processing to reduce noise.

        Args:
            event_type: Type of event (created, modified, deleted, moved)
            file_path: Path to affected file
        """
        # Filter files
        if not self._should_monitor_file(file_path):
            return

        # Add to monitored files set
        self.monitored_files.add(file_path)

        # Simple time-based debouncing (Context7 pattern: reduce noise)
        event_key = f"{file_path}:{event_type}"
        current_time = time.time()

        # Check if we should process this event (debounce with 1 second delay)
        last_processed = self._debounced_events.get(event_key, 0)
        if current_time - last_processed < 1.0:
            logger.debug("Event debounced",
                        event_type=event_type,
                        file_path=Path(file_path).name)
            return

        # Update last processed time
        self._debounced_events[event_key] = current_time

        # Process event asynchronously
        asyncio.create_task(self._process_file_event(event_type, file_path))

        logger.debug("File event queued",
                    event_type=event_type,
                    file_path=Path(file_path).name)

    async def _process_file_event(self, event_type: str, file_path: str) -> None:
        """
        Process a file system event asynchronously.

        Args:
            event_type: Type of event (created, modified, deleted, moved)
            file_path: Path to affected file
        """
        try:
            modification = FileModification(
                file_path=file_path,
                event_type=event_type,
                timestamp=datetime.now(),
                file_size=Path(file_path).stat().st_size if Path(file_path).exists() else None,
                content_preview=self._extract_content_preview(file_path) if event_type != "deleted" else None,
                session_id=await self._get_current_session_id()
            )

            await self._store_file_modification(modification)

        except Exception as e:
            logger.error("Error processing file event",
                        file_path=file_path,
                        event_type=event_type,
                        error=str(e))

    def start_monitoring(self, paths: Optional[List[str]] = None) -> bool:
        """
        Start real-time file monitoring.

        Context7 Pattern: Observer.start() with error handling.

        Args:
            paths: List of paths to monitor (defaults to project root)

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Real-time monitoring already running")
            return True

        # CRASH PREVENTION: Safety check before starting monitoring
        if self.crash_monitor and should_disable_file_monitoring():
            logger.error("File monitoring DISABLED for crash prevention")
            return False

        # Assess current risk level
        if self.crash_monitor:
            risk_metrics = self.crash_monitor.assess_crash_risk()
            if risk_metrics.risk_level.value in ['critical', 'high']:
                logger.warning("High system risk detected for file monitoring",
                             risk_level=risk_metrics.risk_level.value,
                             warnings=risk_metrics.warnings)

        try:
            # Initialize watchdog observer (Context7 best practice for macOS)
            if self.observer_type == 'polling':
                # Use PollingObserver on macOS to prevent kernel panics
                self.observer = PollingObserver()
                logger.info("PollingObserver initialized (macOS FSEvents-safe)")
            else:
                # Use native observer on other platforms
                self.observer = Observer()
                logger.info("Native Observer initialized")

            # Determine paths to monitor
            monitor_paths = paths or [str(self.project_root)]

            # Schedule monitoring for each path
            for path in monitor_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    self.observer.schedule(
                        self.event_handler,
                        str(path_obj),
                        recursive=True
                    )
                    logger.info("Scheduled path monitoring", path=str(path_obj))
                else:
                    logger.warning("Path does not exist, skipping", path=str(path_obj))

            # Start the observer
            self.observer.start()
            self.is_running = True
            self.start_time = datetime.now()

            logger.info("Real-time file monitoring started",
                       paths=monitor_paths,
                       observer_type=type(self.observer).__name__)

            return True

        except Exception as e:
            logger.error("Failed to start real-time monitoring", error=str(e))
            self.is_running = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop real-time file monitoring.

        Context7 Pattern: Observer.stop() with graceful shutdown.

        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Real-time monitoring not running")
            return True

        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5.0)  # Wait up to 5 seconds

                if self.observer.is_alive():
                    logger.warning("Observer did not stop gracefully")
                else:
                    logger.info("Real-time file monitoring stopped")

            self.is_running = False

            # Log monitoring statistics
            if self.start_time:
                duration = datetime.now() - self.start_time
                logger.info("Monitoring session completed",
                           duration=str(duration),
                           files_monitored=len(self.monitored_files))

            return True

        except Exception as e:
            logger.error("Failed to stop real-time monitoring", error=str(e))
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring status information
        """
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "monitored_files_count": len(self.monitored_files),
            "monitored_extensions": list(self.MONITORED_EXTENSIONS),
            "project_root": str(self.project_root),
            "observer_type": type(self.observer).__name__ if self.observer else None
        }


# Singleton instance for global access
_instance: Optional[RealTimeDataCapture] = None


def get_real_time_capture(project_root: Optional[str] = None) -> RealTimeDataCapture:
    """
    Get singleton instance of RealTimeDataCapture.

    Context7 Pattern: Singleton for global file monitoring.

    Args:
        project_root: Root directory to monitor

    Returns:
        RealTimeDataCapture instance
    """
    global _instance
    if _instance is None:
        _instance = RealTimeDataCapture(project_root)
    return _instance


async def start_real_time_monitoring(paths: Optional[List[str]] = None) -> bool:
    """
    Start real-time file monitoring (convenience function).

    Args:
        paths: List of paths to monitor

    Returns:
        True if monitoring started successfully
    """
    capture = get_real_time_capture()
    return capture.start_monitoring(paths)


async def stop_real_time_monitoring() -> bool:
    """
    Stop real-time file monitoring (convenience function).

    Returns:
        True if monitoring stopped successfully
    """
    capture = get_real_time_capture()
    return capture.stop_monitoring()