#!/bin/bash
# z.ai Provider Configuration for DevStream

# Color codes (if not already set by parent script)
RED=${RED:-'\033[0;31m'}
GREEN=${GREEN:-'\033[0;32m'}
YELLOW=${YELLOW:-'\033[1;33m'}
NC=${NC:-'\033[0m'}

# Source root .env first (single source of truth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Then source provider config (for BASE_URL and model mappings)
if [ -f "$PROJECT_ROOT/.env.llm-providers" ]; then
    source "$PROJECT_ROOT/.env.llm-providers"
fi

# Validate API key (existence only, no connectivity test)
if [ -z "$ZAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: ZAI_API_KEY not set in root .env${NC}"
    exit 1
fi

# Export Anthropic-compatible env vars
export ANTHROPIC_BASE_URL="${ZAI_BASE_URL}"
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_API_KEY="${ZAI_API_KEY}"

# Output configuration
echo -e "${GREEN}✅ z.ai provider configured${NC}"
echo -e "${GREEN}   Base URL: ${ANTHROPIC_BASE_URL}${NC}"
echo -e "${GREEN}   Models:${NC}"
echo -e "${GREEN}     - Opus   → ${ZAI_MODEL_OPUS}${NC}"
echo -e "${GREEN}     - Sonnet → ${ZAI_MODEL_SONNET}${NC}"
echo -e "${GREEN}     - Haiku  → ${ZAI_MODEL_HAIKU}${NC}"
