"""Structured logging utilities for Codex adapters."""

from __future__ import annotations

import logging
from typing import Optional

import structlog


_LOGGER_NAME = "devstream.codex"


def get_codex_logger(component: Optional[str] = None) -> "structlog.stdlib.BoundLogger":
    """Return a structlog logger bound to the Codex integration namespace."""
    logger = structlog.get_logger(_LOGGER_NAME)
    if component:
        logger = logger.bind(component=component)
    return logger


def ensure_stdlib_handler(level: int = logging.INFO) -> None:
    """Ensure a standard logging handler exists for Codex adapters."""
    root_logger = logging.getLogger(_LOGGER_NAME)
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


__all__ = ["get_codex_logger", "ensure_stdlib_handler"]
