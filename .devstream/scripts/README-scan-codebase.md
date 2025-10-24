# DevStream Codebase Scanner

Async Python script to scan the codebase and populate `semantic_memory` with embeddings for semantic search.

## Features

- **Multi-format scanning**: Docs (*.md), code (*.py, *.ts, *.tsx, *.rs, *.go), configs (*.json, *.yaml)
- **Keyword extraction**: AST parsing for Python, regex for other languages, NLP for documentation
- **Ollama integration**: Batch embedding generation (768D vectors via `embeddinggemma:300m`)
- **Database triggers**: Automatic sync to `vec_semantic_memory` (vector search) and `fts_semantic_memory` (keyword search)
- **Rich progress tracking**: Real-time progress bars, ETAs, and summaries
- **Graceful error handling**: Retry logic (3 attempts), encoding detection, partial failure tolerance

## Prerequisites

1. **Python 3.11+** with `.devstream` virtual environment
2. **Ollama server** running (`ollama serve`)
3. **embeddinggemma:300m model** installed (`ollama pull embeddinggemma:300m`)
4. **Dependencies**: `aiohttp`, `rich`, `python-dotenv`, `sqlite-vec`, `ollama`

## Usage

### Basic Usage

```bash
# Scan all files (docs + code)
.devstream/bin/python scripts/scan-codebase.py

# Scan only documentation
.devstream/bin/python scripts/scan-codebase.py --docs-only

# Scan only code files
.devstream/bin/python scripts/scan-codebase.py --code-only
```

### Advanced Options

```bash
# Custom batch size (default: 10)
.devstream/bin/python scripts/scan-codebase.py --batch-size 20

# Verbose output (detailed logging)
.devstream/bin/python scripts/scan-codebase.py --verbose

# Skip embeddings (testing mode - database insert only)
.devstream/bin/python scripts/scan-codebase.py --skip-embeddings

# Combine options
.devstream/bin/python scripts/scan-codebase.py --docs-only --batch-size 5 --verbose
```

### Help

```bash
.devstream/bin/python scripts/scan-codebase.py --help
```

## What Gets Scanned

### Directories

| Directory | Content Type | File Patterns |
|-----------|-------------|---------------|
| `docs/` | documentation | *.md, *.rst, *.txt |
| `.claude/agents/` | context | *.md |
| `.claude/hooks/` | code | *.py |
| `mcp-devstream-server/` | code | *.ts, *.js, *.json, *.yaml |
| `scripts/` | code | *.py |

### Excluded

- `.git/`, `node_modules/`, `.venv/`, `.devstream/`, `__pycache__/`, `dist/`, `build/`, `data/`
- Files > 5MB
- Binary files

## Output

### Console Output

```
🔍 DevStream Codebase Scanner
─────────────────────────────

🤖 Testing Ollama connection...
✅ Ollama connected

📁 Scanning files...
✅ Found 27 files

🔑 Extracting keywords...
  Extracting keywords... ━━━━━━━━━━━━━━━━━ 100%
✅ Keywords extracted

🧠 Generating embeddings...
  Generating embeddings... ━━━━━━━━━━━━━━━━━ 100% 0:00:15
✅ Embeddings generated: 27/27

💾 Inserting into database...
✅ Inserted 27 records
✅ Database updated: 27 inserted, 0 updated

📊 Summary
┌────────────────────────┬───────┐
│ Metric                 │ Count │
├────────────────────────┼───────┤
│ Files Scanned          │    27 │
│ Memories Created       │    27 │
│ Embeddings Generated   │    27 │
│ Database Inserts       │    27 │
│ Database Updates       │     0 │
└────────────────────────┴───────┘

✅ Scan completed successfully
```

### Database Tables Updated

1. **semantic_memory**: Main memory storage (content, keywords, embeddings)
2. **vec_semantic_memory**: Vector search virtual table (sqlite-vec)
3. **fts_semantic_memory**: Full-text search virtual table (FTS5)

Triggers automatically sync records between tables.

## Embedding Details

- **Model**: `embeddinggemma:300m` (Ollama)
- **Dimensions**: 768D
- **Batch size**: 10 (configurable via `--batch-size`)
- **Retry logic**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Content limit**: 5000 chars per file (for embedding generation)

## Testing Semantic Search

After scanning, test semantic search:

```python
from sqlite_vec_helper import get_db_connection_with_vec
from ollama_client import OllamaEmbeddingClient
import json

# Generate query embedding
client = OllamaEmbeddingClient()
query = "DevStream agent architecture"
query_embedding = client.generate_embedding(query)

# Semantic search
conn = get_db_connection_with_vec('data/devstream.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT memory_id, distance, content_preview
    FROM vec_semantic_memory
    WHERE embedding MATCH ?
      AND k = 5
    ORDER BY distance
''', (json.dumps(query_embedding),))

for memory_id, distance, preview in cursor.fetchall():
    print(f"{memory_id}: {distance:.4f}")
    print(f"  {preview[:80]}...")
```

## Performance

- **Scan speed**: ~50-100 files/min (depends on file size)
- **Embedding generation**: ~2-5 seconds per batch (10 files)
- **Database insert**: ~1-2 seconds per batch

**Total time for 27 docs**: ~15-20 seconds

## Error Handling

### Graceful Degradation

- File read errors → Skip file, continue scanning
- Ollama connection failure → Exit with error code 2 (unless `--skip-embeddings`)
- Individual embedding failure → Mark as None, continue batch
- Database insert failure → Log error, mark in stats

### Exit Codes

- **0**: Success (all operations completed)
- **1**: Partial success (>50% embeddings failed or database errors)
- **2**: Failure (Ollama unavailable, no files found)

## Troubleshooting

### Ollama Not Available

```bash
# Start Ollama server
ollama serve

# Verify model installed
ollama list | grep embeddinggemma
```

### Permission Errors

```bash
# Ensure .devstream venv has correct permissions
chmod +x .devstream/bin/python
```

### Database Schema Issues

If `vec_semantic_memory` table is missing:

```bash
# Reinitialize database with vec0 extension
.devstream/bin/python -c "
from sqlite_vec_helper import get_db_connection_with_vec
conn = get_db_connection_with_vec('data/devstream.db')
# Schema will be created if missing
"
```

## Architecture

### Class Structure

```
FileScanner
  ├── scan_directory() → List[ScannedFile]
  ├── should_scan_file() → bool
  └── read_file_content() → Optional[str]

KeywordExtractor
  ├── extract_python_keywords() → List[str]
  ├── extract_generic_keywords() → List[str]
  └── extract_doc_keywords() → List[str]

EmbeddingGenerator
  ├── test_connection() → bool
  ├── generate_batch() → List[Optional[List[float]]]
  └── generate_all() → List[MemoryRecord]

DatabaseInserter
  └── insert_batch() → int

main() → int (exit code)
```

### Data Flow

```
Files → FileScanner → ScannedFile[]
                           ↓
                    KeywordExtractor
                           ↓
                    EmbeddingGenerator (Ollama)
                           ↓
                    MemoryRecord[] (with embeddings)
                           ↓
                    DatabaseInserter
                           ↓
     ┌──────────────────┴──────────────────┐
     ↓                  ↓                   ↓
semantic_memory  vec_semantic_memory  fts_semantic_memory
(main storage)   (vector search)      (keyword search)
```

## Type Safety

- Full type hints (mypy --strict compliant)
- Pydantic-style dataclasses (`@dataclass`)
- Optional types for graceful None handling
- Enum for ContentType (code/documentation/context)

## Logging

- Structured logging via `ollama_client.py`
- Rich console output for user feedback
- Verbose mode (`--verbose`) for debugging
- Statistics tracking (success/failed counts)

## Future Enhancements

- [ ] Incremental scanning (only modified files)
- [ ] File hash deduplication (already stores hash)
- [ ] Multi-model support (nomic-embed-text, all-minilm)
- [ ] Parallel embedding generation (asyncio batches)
- [ ] Progress persistence (resume interrupted scans)
- [ ] Custom directory configuration (YAML/JSON)

## Related Documentation

- [DevStream Memory System](../docs/architecture/memory_and_context_system.md)
- [Ollama Client](../.claude/hooks/devstream/utils/ollama_client.py)
- [SQLite-Vec Helper](../.claude/hooks/devstream/utils/sqlite_vec_helper.py)
- [Database Schema](../schema/schema.sql)

## License

MIT License - Part of DevStream project
