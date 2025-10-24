#!/usr/bin/env bash
# DevStream Codex adapter entrypoint
set -euo pipefail

PYTHON_BIN="${PYTHON:-$(dirname "$0")/../../.devstream/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

exec "$PYTHON_BIN" -m devstream.integrations.codex.cli_runner "$@"
