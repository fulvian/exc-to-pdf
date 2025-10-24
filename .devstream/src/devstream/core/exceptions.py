"""
Custom exceptions per il sistema DevStream.

Hierarchy delle eccezioni:
- DevStreamError (base)
  ├── DatabaseError
  ├── MemoryError
  ├── TaskError
  ├── HookError
  ├── ValidationError
  └── ConfigurationError
"""

from typing import Any, Dict, Optional


class DevStreamError(Exception):
    """Base exception per tutti gli errori DevStream."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class DatabaseError(DevStreamError):
    """Errori relativi al database SQLite."""

    def __init__(
        self,
        message: str,
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if table:
            context["table"] = table
        if query:
            context["query"] = query[:100] + "..." if len(query) > 100 else query
        super().__init__(message, **kwargs, context=context)


class EntityNotFoundError(DatabaseError):
    """Errore quando un'entità richiesta non viene trovata."""

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if entity_type:
            context["entity_type"] = entity_type
        if entity_id:
            context["entity_id"] = entity_id
        super().__init__(message, **kwargs, context=context)


class MemoryError(DevStreamError):
    """Errori relativi al sistema memoria semantica."""

    def __init__(
        self,
        message: str,
        memory_id: Optional[str] = None,
        search_query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if memory_id:
            context["memory_id"] = memory_id
        if search_query:
            context["search_query"] = search_query
        super().__init__(message, **kwargs, context=context)


class TaskError(DevStreamError):
    """Errori relativi alla gestione task."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        phase_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if task_id:
            context["task_id"] = task_id
        if task_type:
            context["task_type"] = task_type
        if phase_id:
            context["phase_id"] = phase_id
        super().__init__(message, **kwargs, context=context)


class HookError(DevStreamError):
    """Errori relativi al sistema hook."""

    def __init__(
        self,
        message: str,
        hook_id: Optional[str] = None,
        event_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if hook_id:
            context["hook_id"] = hook_id
        if event_type:
            context["event_type"] = event_type
        super().__init__(message, **kwargs, context=context)


class ValidationError(DevStreamError):
    """Errori di validazione dati."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]
        super().__init__(message, **kwargs, context=context)


class ConfigurationError(DevStreamError):
    """Errori di configurazione sistema."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_file:
            context["config_file"] = config_file
        super().__init__(message, **kwargs, context=context)


class OllamaError(DevStreamError):
    """Errori relativi a integrazione Ollama."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if model:
            context["model"] = model
        if endpoint:
            context["endpoint"] = endpoint
        super().__init__(message, **kwargs, context=context)


class EmbeddingError(MemoryError):
    """Errori specifici per generazione embedding."""

    def __init__(
        self,
        message: str,
        text_length: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if text_length:
            context["text_length"] = text_length
        if model:
            context["model"] = model
        super().__init__(message, **kwargs, context=context)


class SearchError(MemoryError):
    """Errori specifici per ricerca semantica."""

    def __init__(
        self,
        message: str,
        search_type: Optional[str] = None,
        results_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if search_type:
            context["search_type"] = search_type
        if results_count is not None:
            context["results_count"] = results_count
        super().__init__(message, **kwargs, context=context)