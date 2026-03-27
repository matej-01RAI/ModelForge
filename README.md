# ML Model Building Agent

An autonomous AI agent that builds machine learning models through conversation. Describe your task, review the plan, and the agent handles everything — from data analysis to model delivery.

Powered by [Claude](https://www.anthropic.com/claude) via [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-studio) and orchestrated with [LangChain](https://www.langchain.com/).

https://github.com/user-attachments/assets/placeholder-demo-video

## How It Works

```
You:   I have sales data in data/sales.csv. Build a churn classifier, target column is "churn".

Agent: [asks 2-3 clarifying questions if needed, then creates a plan]

You:   Looks good, go for it.

Agent: [executes autonomously — you see live progress]
  [0:02] ## Analyzing dataset data/sales.csv
  [0:05] ?? Searching "best models tabular binary classification"
  [0:08] >> Setting up virtual environment
  [0:12] >> Installing scikit-learn pandas xgboost
  [0:20] => Writing src/train_models.py
  [0:45] >> Running train_models.py
  [1:10] => Writing results/report.md

  Done: 8 steps in 1m 12s | tokens: 24,530 total | 21,200 in | 3,330 out | 6 LLM calls
```

The agent works in two phases:

1. **Planner** — Conversational agent that gathers requirements and creates a structured plan
2. **Builder** — Autonomous agent with tools that executes the plan end-to-end

## Features

- **Fully autonomous execution** — analyzes data, researches approaches, writes code, trains models, tunes hyperparameters, and delivers results
- **Live terminal UI** — see what the agent is doing in real-time with status updates and timing
- **Token usage tracking** — monitor per-turn and session-wide token consumption
- **Isolated workspaces** — each project gets its own directory with a dedicated Python virtual environment
- **Web research** — searches for state-of-the-art approaches before building
- **Multi-framework** — scikit-learn, PyTorch, XGBoost, LightGBM, and more (installed as needed)

## Quick Start

```bash
# Clone and set up
git clone https://github.com/matej-01RAI/MLModelBuildingAgent.git
cd MLModelBuildingAgent
bash setup.sh

# Configure (pick one provider — see below)
cp .env.example .env
nano .env

# Run
source venv/bin/activate
python main.py
```

### Requirements

- Python 3.9+
- One of the supported LLM providers (see below)

### Providers

The agent supports three ways to connect to Claude. Set one up in your `.env` file:

#### Option 1: Anthropic API (recommended)

The simplest option. Get an API key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250514
```

#### Option 2: Azure AI Foundry

For enterprise users with Azure deployments.

```env
AZURE_AI_ENDPOINT=https://your-resource.services.ai.azure.com/anthropic/
AZURE_AI_API_KEY=your-api-key
AZURE_AI_MODEL=claude-sonnet
```

#### Option 3: Claude Code CLI

Use your existing [Claude Code](https://claude.ai/code) subscription — no separate API key needed.

```bash
# Prerequisites
npm install -g @anthropic-ai/claude-code
pip install langchain-claude-code
```

```env
PROVIDER=claude-code
CLAUDE_CODE_MODEL=sonnet   # opus, sonnet, or haiku
```

> **Auto-detection:** If you don't set `PROVIDER` explicitly, the agent picks the first available: `ANTHROPIC_API_KEY` → Anthropic, `AZURE_AI_*` → Azure, fallback → Claude Code CLI.

## Commands

| Command | Description |
|---------|-------------|
| `/quit`, `/exit` | Exit the agent |
| `/clear` | Clear conversation history and start fresh |
| `/workspace` | Show the workspace directory path |
| `/tokens` | Show session token usage (input, output, cache, LLM calls) |
| `/build` | Skip approval prompt and start building immediately |

## Architecture

```
MLModelBuildingAgent/
├── main.py                    # Terminal UI, spinner, token tracking, chat loop
├── config.py                  # Environment variable loading
├── setup.sh                   # One-command setup script
├── requirements.txt           # Pinned Python dependencies
├── agent/
│   ├── ml_agent.py            # Builder agent (LangChain AgentExecutor + tools)
│   ├── planning_agent.py      # Planner agent (conversational, no tools)
│   └── tools/
│       ├── terminal.py        # Shell command execution (600s timeout)
│       ├── file_manager.py    # Read, write, and list files
│       ├── web_research.py    # DuckDuckGo search + URL fetching
│       └── data_analyzer.py   # Auto-analyze CSV/JSON/Parquet datasets
└── workspaces/                # Generated at runtime — one directory per project
```

### Agent Tools

| Tool | Icon | What it does |
|------|------|-------------|
| `run_terminal_command` | `>>` | Execute shell commands — create venvs, install packages, run scripts |
| `read_file` | `<<` | Read file contents (up to 500KB) |
| `write_file` | `=>` | Create or overwrite files, auto-creates parent directories |
| `list_directory` | `[]` | Browse directory contents |
| `analyze_dataset` | `##` | Auto-analyze datasets — types, stats, correlations, class balance |
| `search_web` | `??` | Search DuckDuckGo for ML papers, benchmarks, best practices |
| `fetch_url` | `<>` | Fetch and extract text from web pages |

### Workspace Structure

Each project the agent builds gets an isolated workspace:

```
workspaces/my_project/
├── venv/       # Dedicated Python virtual environment
├── data/       # Copy of training data
├── src/        # Training scripts
├── models/     # Saved models (.pkl, .pth)
└── results/    # Metrics, plots, reports
```

## Security Considerations

This agent executes code autonomously on your machine. Understand these trade-offs before running it:

- **Shell execution** — The `run_terminal_command` tool runs arbitrary shell commands with `shell=True`. This is by design (the agent needs to install packages, run scripts, etc.), but it means the agent has the same filesystem and network access as your user account.
- **File access** — File tools can read/write anywhere your user has permissions. The agent is instructed to work within `workspaces/`, but this is a convention, not a hard sandbox.
- **Network access** — The web research tools can make outgoing HTTP requests. Private/internal network URLs are blocked by default (SSRF protection), but the terminal tool has unrestricted network access.
- **API key** — Your Azure AI API key is stored in `.env` (gitignored). Never commit this file.

**Recommendations:**
- Run the agent in a container or VM if you need stronger isolation
- Review the agent's plan before approving builds on sensitive systems
- Monitor token usage with `/tokens` to control costs

## Customization

### Changing the LLM

Switch providers by updating your `.env` file — see [Providers](#providers) above. The LLM is created by `agent/llm_factory.py`, which both agents share. To add a completely new provider (e.g., OpenAI, local models), add a new factory function there.

### Adding Tools

1. Create a tool file in `agent/tools/` using the `@tool` decorator
2. Import and add it to the `tools` list in `agent/ml_agent.py`
3. Add display icons/verbs in `LiveStatusCallback` in `main.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Tuning Agent Behavior

- **Builder prompt** — `SYSTEM_PROMPT` in `agent/ml_agent.py` controls autonomous execution behavior
- **Planner prompt** — `PLANNING_PROMPT` in `agent/planning_agent.py` controls question-asking and plan format
- **Temperature** — Builder uses 0.1 (deterministic), planner uses 0.3 (slightly creative)
- **Max iterations** — Builder allows up to 50 tool calls per turn

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
