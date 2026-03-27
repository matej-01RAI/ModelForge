import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider detection ────────────────────────────────────────────────────
# Explicit override via PROVIDER env var, or auto-detect from available keys.
# Supported: "anthropic", "azure", "claude-code"

PROVIDER = os.getenv("PROVIDER", "").lower()

# Anthropic direct API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250514")

# Azure AI Foundry
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")
AZURE_AI_MODEL = os.getenv("AZURE_AI_MODEL", "claude-sonnet")
AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview")

# Claude Code CLI
CLAUDE_CODE_MODEL = os.getenv("CLAUDE_CODE_MODEL", "sonnet")  # "opus", "sonnet", "haiku"

# Auto-detect provider if not explicitly set
if not PROVIDER:
    if ANTHROPIC_API_KEY:
        PROVIDER = "anthropic"
    elif AZURE_AI_ENDPOINT and AZURE_AI_API_KEY:
        PROVIDER = "azure"
    else:
        # Default: check if claude CLI is available (claude-code provider)
        PROVIDER = "claude-code"

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspaces")
