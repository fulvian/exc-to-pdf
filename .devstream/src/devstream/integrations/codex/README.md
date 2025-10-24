# DevStream Codex Integration (WIP)

This package mirrors the Claude Code hook pipeline for Codex CLI by wrapping existing DevStream services.

## Guarantees
- No mutations to `.claude` hook scripts; adapters only import and reuse their logic.
- Protocol enforcement, Context7 research, semantic memory capture, and checkpointing remain available when Codex emits the corresponding events.
- Failures degrade gracefully: Codex commands must never block on DevStream automation.
- Configuration stays isolated via `DEVSTREAM_CODEX_*` environment variables and optional `config/codex/*.yml` overlays.

## Status
Implementation in progress. Run `make dev` to install optional Codex extras before wiring adapters.

## Runtime dispatcher
Usa `CodexIntegrationRuntime` per collegare gli eventi CLI: passagli il dizionario dell'evento e riceverai l'azione da intraprendere (prompt protocollo, contesto pre-tool, ID memoria, sommario sessione). Gli adapter possono essere sovrascritti per testare con stub o per implementazioni personalizzate.

## CLI shim
Esegui `scripts/codex/relay.sh` per instradare eventi JSON (singoli o stream) verso il runtime DevStream. Le risposte sono restituite su stdout; i payload campione vengono salvati per impostazione predefinita in `data/codex_event_samples.jsonl`.
- Override della home: impostare `DEVSTREAM_CODEX_HOME` per forzare i log in una directory scrivibile (utile in ambienti sandbox).

### Esecuzione esempi
```
DEVSTREAM_CODEX_HOME=data/codex_home \
DEVSTREAM_MEMORY_ENABLED=false \
DEVSTREAM_PROTOCOL_ENFORCEMENT_ENABLED=false \
scripts/codex/relay.sh --file events_demo.json
```
*Nota*: disabilitare memoria/protocollo è consigliato solo per simulazioni offline; per la parità completa con Claude occorre avviare l’MCP DevStream server.
```
