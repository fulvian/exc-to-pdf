"""Shim CLI per instradare eventi Codex nel runtime DevStream."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Dict, Any

from .config import CodexSettings
from .dispatcher import CodexIntegrationRuntime


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DevStream Codex adapter runner")
    parser.add_argument(
        "--event",
        help="Evento Codex in formato JSON (stringa). Se omesso si legge da stdin (JSON lines).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Percorso a file JSON contenente un singolo evento o una lista di eventi.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disabilita la persistenza dei payload campione su disco.",
    )
    return parser.parse_args(argv)


def _iter_events(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.event:
        yield json.loads(args.event)
        return

    if args.file:
        data = json.loads(args.file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


async def _process_events(runtime: CodexIntegrationRuntime, events: Iterable[Dict[str, Any]]) -> None:
    for raw in events:
        result = await runtime.handle_event(raw)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    settings = CodexSettings()

    override_home = os.environ.get("DEVSTREAM_CODEX_HOME")
    if override_home:
        os.environ["HOME"] = str(Path(override_home).expanduser().resolve())
    elif not (Path.home() / ".claude").exists():
        os.environ.setdefault("HOME", str(Path.cwd().resolve()))

    runtime = CodexIntegrationRuntime(
        settings=settings,
        record_samples=not args.no_sample,
    )

    try:
        asyncio.run(_process_events(runtime, _iter_events(args)))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - fallback CLI
        sys.stderr.write(f"Codex adapter error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
