#!/bin/bash

# start-devstream.sh - Multi-Provider LLM Launcher for Claude Code
# Usage: ./scripts/start-devstream.sh [provider]
# Providers: z.ai|zai, synthetic, nanogpt, openrouter, anthropic (default)

# 2.1 - Skeleton & Error Handling
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 2.2 - Provider Selection Logic
PROVIDER="${1:-${DEVSTREAM_LLM_PROVIDER:-anthropic}}"

# Normalize provider name (handle aliases)
case "${PROVIDER}" in
    z.ai|zai)
        PROVIDER="z.ai"
        ;;
    synthetic)
        PROVIDER="synthetic"
        ;;
    nanogpt)
        echo -e "${RED}❌ Error: nanogpt provider not yet implemented (Phase 2)${NC}"
        echo -e "${YELLOW}Available providers: z.ai, synthetic, anthropic${NC}"
        exit 1
        ;;
    openrouter)
        echo -e "${RED}❌ Error: openrouter provider not yet implemented (Phase 2)${NC}"
        echo -e "${YELLOW}Available providers: z.ai, synthetic, anthropic${NC}"
        exit 1
        ;;
    anthropic|*)
        PROVIDER="anthropic"
        ;;
esac

# 2.3 - Validation Functions

# Validate API key is present
validate_api_key() {
    local key_name="$1"
    local key_value="$2"

    if [ -z "${key_value}" ]; then
        echo -e "${RED}❌ Error: ${key_name} is not set${NC}"
        echo -e "${YELLOW}Please configure ${key_name} in .env.llm-providers or .env${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ ${key_name} validated${NC}"
}

# Test Synthetic API connectivity
test_synthetic_api() {
    local base_url="$1"
    local api_key="$2"

    echo -e "${YELLOW}Testing Synthetic API connectivity...${NC}"

    # Minimal test payload
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${base_url}/v1/messages" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${api_key}" \
        -d '{
            "model": "hf:zai-org/GLM-4.6",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "test"}]
        }' \
        --connect-timeout 10 \
        --max-time 15 2>/dev/null || echo "000")

    # Accept 200 (success), 401 (auth error but reachable), 400 (bad request but reachable)
    case "${response_code}" in
        200|401|400)
            echo -e "${GREEN}✅ Synthetic API is reachable (HTTP ${response_code})${NC}"
            return 0
            ;;
        000)
            echo -e "${RED}❌ Error: Cannot reach Synthetic API (connection timeout/failed)${NC}"
            echo -e "${YELLOW}Check your network connection and SYNTHETIC_BASE_URL${NC}"
            exit 1
            ;;
        *)
            echo -e "${RED}❌ Error: Synthetic API returned unexpected status (HTTP ${response_code})${NC}"
            echo -e "${YELLOW}This may indicate a server issue${NC}"
            exit 1
            ;;
    esac
}

# 2.4 - Provider Configuration & Claude Code Launcher

echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  DevStream Multi-Provider LLM Launcher${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""

case "${PROVIDER}" in
    z.ai)
        echo -e "${YELLOW}Loading z.ai configuration...${NC}"

        # Load from .env.llm-providers
        if [ -f "${PROJECT_ROOT}/.env.llm-providers" ]; then
            # shellcheck source=/dev/null
            source "${PROJECT_ROOT}/.env.llm-providers"
        else
            echo -e "${RED}❌ Error: .env.llm-providers not found${NC}"
            echo -e "${YELLOW}Run: cp .env.llm-providers.example .env.llm-providers${NC}"
            exit 1
        fi

        # Validate API key
        validate_api_key "ZAI_API_KEY" "${ZAI_API_KEY:-}"

        # Export for Claude Code
        export ANTHROPIC_BASE_URL="${ZAI_BASE_URL}"
        unset ANTHROPIC_AUTH_TOKEN
        export ANTHROPIC_API_KEY="${ZAI_API_KEY}"

        # Output configuration
        echo ""
        echo -e "${GREEN}✅ Provider: z.ai (GLM Models)${NC}"
        echo -e "${GREEN}   Base URL: ${ZAI_BASE_URL}${NC}"
        echo -e "${GREEN}   Model Mappings:${NC}"
        echo -e "${GREEN}     • claude-opus-4-5     → glm-4.6${NC}"
        echo -e "${GREEN}     • claude-sonnet-4-5   → glm-4.6${NC}"
        echo -e "${GREEN}     • claude-haiku-4-5    → glm-4-flash${NC}"
        echo ""
        ;;

    synthetic)
        echo -e "${YELLOW}Loading Synthetic configuration...${NC}"

        # Source .env from project root
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            # shellcheck source=/dev/null
            source "${PROJECT_ROOT}/.env"
        else
            echo -e "${RED}❌ Error: .env not found in project root${NC}"
            exit 1
        fi

        # Load .env.llm-providers for base URL
        if [ -f "${PROJECT_ROOT}/.env.llm-providers" ]; then
            # shellcheck source=/dev/null
            source "${PROJECT_ROOT}/.env.llm-providers"
        else
            echo -e "${RED}❌ Error: .env.llm-providers not found${NC}"
            exit 1
        fi

        # Validate API key
        validate_api_key "SYNTHETIC_API_KEY" "${SYNTHETIC_API_KEY:-}"

        # Test API connectivity
        test_synthetic_api "${SYNTHETIC_BASE_URL}" "${SYNTHETIC_API_KEY}"

        # Start Synthetic Proxy (model name translator)
        echo -e "${YELLOW}🚀 Starting Synthetic proxy server...${NC}"

        # Export env vars for proxy
        export SYNTHETIC_API_KEY
        export SYNTHETIC_MODEL

        # Kill any existing proxy on port 3100
        lsof -ti:3100 | xargs kill -9 2>/dev/null || true

        # Start proxy in background
        PROXY_SCRIPT="${PROJECT_ROOT}/scripts/synthetic-proxy.js"
        PROXY_LOG="${PROJECT_ROOT}/.logs/synthetic-proxy.log"
        mkdir -p "${PROJECT_ROOT}/.logs"

        node "$PROXY_SCRIPT" > "$PROXY_LOG" 2>&1 &
        PROXY_PID=$!

        # Wait for proxy to be ready
        sleep 2

        # Verify proxy is running
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo -e "${RED}❌ Error: Proxy failed to start${NC}"
            echo -e "${YELLOW}Check logs: ${PROXY_LOG}${NC}"
            exit 1
        fi

        echo -e "${GREEN}✅ Proxy running (PID: ${PROXY_PID}, Port: 3100)${NC}"

        # Setup cleanup trap
        trap "echo -e '\n${YELLOW}🛑 Stopping Synthetic proxy...${NC}'; kill $PROXY_PID 2>/dev/null; exit" INT TERM EXIT

        # Export for Claude Code (point to local proxy)
        export ANTHROPIC_BASE_URL="http://localhost:3100"
        export ANTHROPIC_AUTH_TOKEN="synthetic-proxy"

        # Backup settings.json
        SETTINGS_FILE="$HOME/.claude/settings.json"
        BACKUP_FILE="$HOME/.claude/settings.json.backup-synthetic"

        if [ -f "$SETTINGS_FILE" ]; then
            cp "$SETTINGS_FILE" "$BACKUP_FILE"
        fi

        # Output configuration
        echo ""
        echo -e "${GREEN}✅ Provider: Synthetic.new (via Proxy)${NC}"
        echo -e "${GREEN}   Proxy URL: http://localhost:3100${NC}"
        echo -e "${GREEN}   Target: ${SYNTHETIC_BASE_URL}${NC}"
        echo -e "${GREEN}   Model: ${SYNTHETIC_MODEL}${NC}"
        echo -e "${GREEN}   Tier: ${SYNTHETIC_TIER:-free}${NC}"
        echo -e "${YELLOW}   📋 Proxy logs: ${PROXY_LOG}${NC}"
        echo -e "${YELLOW}   ℹ️  Settings backup: ${BACKUP_FILE}${NC}"
        echo ""
        ;;

    anthropic)
        echo -e "${YELLOW}Using Anthropic (default)...${NC}"

        # Validate ANTHROPIC_API_KEY from environment
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo -e "${RED}❌ Error: ANTHROPIC_API_KEY not set${NC}"
            echo -e "${YELLOW}Export ANTHROPIC_API_KEY or use alternative provider${NC}"
            exit 1
        fi

        # No base URL override needed
        echo ""
        echo -e "${GREEN}✅ Provider: Anthropic (Official)${NC}"
        echo -e "${GREEN}   Using native Claude API${NC}"
        echo ""
        ;;
esac

# Launch Claude Code
echo -e "${GREEN}Launching Claude Code...${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""

cd "${PROJECT_ROOT}"
claude
