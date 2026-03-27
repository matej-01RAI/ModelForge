"""LLM factory — creates the right ChatModel based on the configured provider."""

import shutil
import config


def create_llm(temperature: float = 0.1, max_tokens: int = 16384):
    """Create a LangChain ChatModel based on the configured provider.

    Supports three providers:
      - "anthropic"   → Direct Anthropic API (ANTHROPIC_API_KEY)
      - "azure"       → Azure AI Foundry (AZURE_AI_ENDPOINT + AZURE_AI_API_KEY)
      - "claude-code" → Claude Code CLI via langchain-claude-code

    Args:
        temperature: Sampling temperature (0.0–1.0).
        max_tokens: Maximum tokens in the response.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    provider = config.PROVIDER

    if provider == "anthropic":
        return _create_anthropic_llm(temperature, max_tokens)
    elif provider == "azure":
        return _create_azure_llm(temperature, max_tokens)
    elif provider == "claude-code":
        return _create_claude_code_llm(temperature, max_tokens)
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Set PROVIDER to 'anthropic', 'azure', or 'claude-code', "
            f"or set ANTHROPIC_API_KEY / AZURE_AI_ENDPOINT+AZURE_AI_API_KEY."
        )


def get_provider_display_name() -> str:
    """Return a human-readable string describing the active provider + model."""
    provider = config.PROVIDER
    if provider == "anthropic":
        return f"{config.ANTHROPIC_MODEL} via Anthropic API"
    elif provider == "azure":
        return f"{config.AZURE_AI_MODEL} via Azure AI Foundry"
    elif provider == "claude-code":
        return f"{config.CLAUDE_CODE_MODEL} via Claude Code CLI"
    return f"unknown ({provider})"


# ── Provider implementations ──────────────────────────────────────────────


def _create_anthropic_llm(temperature, max_tokens):
    """Direct Anthropic API — simplest path, just needs ANTHROPIC_API_KEY."""
    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Get your key at https://console.anthropic.com/settings/keys"
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=config.ANTHROPIC_MODEL,
        anthropic_api_key=config.ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_azure_llm(temperature, max_tokens):
    """Azure AI Foundry — uses AnthropicFoundry client for correct routing."""
    if not config.AZURE_AI_ENDPOINT or not config.AZURE_AI_API_KEY:
        raise ValueError(
            "AZURE_AI_ENDPOINT and AZURE_AI_API_KEY must both be set for Azure provider."
        )

    from langchain_anthropic import ChatAnthropic
    from anthropic import AnthropicFoundry

    llm = ChatAnthropic(
        model=config.AZURE_AI_MODEL,
        anthropic_api_key=config.AZURE_AI_API_KEY,
        anthropic_api_url=config.AZURE_AI_ENDPOINT,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Swap the internal client with AnthropicFoundry for Azure routing
    _ = llm._client
    llm.__dict__["_client"] = AnthropicFoundry(
        api_key=config.AZURE_AI_API_KEY,
        base_url=config.AZURE_AI_ENDPOINT,
    )
    return llm


def _create_claude_code_llm(temperature, max_tokens):
    """Claude Code CLI — uses langchain-claude-code for Claude Code subscribers."""
    # Check that claude CLI is installed
    if not shutil.which("claude"):
        raise ValueError(
            "Claude Code CLI not found. Install it with:\n"
            "  npm install -g @anthropic-ai/claude-code\n\n"
            "Or use a different provider by setting ANTHROPIC_API_KEY or "
            "AZURE_AI_ENDPOINT + AZURE_AI_API_KEY in your .env file."
        )

    try:
        from langchain_claude_code import ClaudeCodeChatModel
    except ImportError:
        raise ValueError(
            "langchain-claude-code package not installed. Install it with:\n"
            "  pip install langchain-claude-code\n\n"
            "This package requires the Claude Code CLI (npm install -g @anthropic-ai/claude-code)."
        )

    return ClaudeCodeChatModel(
        model=config.CLAUDE_CODE_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
