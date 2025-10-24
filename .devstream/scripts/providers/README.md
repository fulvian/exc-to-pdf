# LLM Provider Scripts

Provider-specific configuration scripts for DevStream multi-provider LLM integration.

## Available Providers (Phase 1)

- **z.ai** (`z.ai.sh`) - Native Anthropic API compatibility, GLM models
- **Synthetic** (`synthetic.sh`) - Anthropic-compatible endpoint, custom models

## Usage

These scripts are sourced automatically by `../start-devstream.sh`:

```bash
# Use z.ai
./scripts/start-devstream.sh z.ai

# Use Synthetic
./scripts/start-devstream.sh synthetic
```

## Provider Scripts

Each script:
1. Sources `.env.llm-providers` configuration
2. Validates API keys
3. Tests connectivity (Synthetic only - quick validation)
4. Exports `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`
5. Prints configuration summary

## Configuration

Edit `.env.llm-providers` in project root to configure providers.

## Adding New Providers

1. Create `scripts/providers/<provider-name>.sh`
2. Follow the pattern from existing scripts
3. Update `start-devstream.sh` to recognize new provider
4. Document in this README

## Environment Variables

Provider scripts set these variables for Claude Code:

- `ANTHROPIC_BASE_URL` - API endpoint URL
- `ANTHROPIC_API_KEY` - API authentication token

## Validation Levels

- **z.ai**: Key existence check only (fast startup)
- **Synthetic**: Full API connectivity test (ensures service availability)

## Troubleshooting

### z.ai connection issues
- Verify `ZAI_API_KEY` in `.env.llm-providers`
- Check z.ai service status

### Synthetic connection issues
- Verify `SYNTHETIC_API_KEY` in root `.env`
- Check `SYNTHETIC_BASE_URL` in `.env.llm-providers`
- Ensure API endpoint is reachable (connectivity test runs automatically)
