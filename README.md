# ModelForge

Give an AI your dataset.
It builds the model.

No manual feature engineering.
No pipeline setup.
No guesswork.

![ModelForge Demo](demo.gif)


## What Is This?

ModelForge is an AI agent that takes your dataset and:
- designs a modeling approach
- builds the model
- creates a data processing pipeline
- generates documentation

All automatically.


## Why This Matters

Building ML models is still too manual.

You need to:
- explore the data
- choose features
- design a pipeline
- tune the model

ModelForge does this for you.


## But Here's the Interesting Part

Once the model is built...

you can ask it **why** it works (using [ModelLens](https://getmodellens.com/)).


## Example

Give it a dataset.

It will:
1. analyze the data
2. propose a modeling plan
3. build the model
4. create a pipeline
5. document everything

Then you can ask:
- Why did you predict this?
- What features matter most?
- What would change the prediction?

```
dataset --> ModelForge --> model --> ModelLens --> explanation
```



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

1. **Planner** -- Conversational agent that gathers requirements and creates a structured plan
2. **Builder** -- Autonomous agent with tools that executes the plan end-to-end


## Features

- **Fully autonomous execution** -- analyzes data, researches approaches, writes code, trains models, tunes hyperparameters, and delivers results
- **Live terminal UI** -- see what the agent is doing in real-time with status updates and timing
- **Token usage tracking** -- monitor per-turn and session-wide token consumption
- **Isolated workspaces** -- each project gets its own directory with a dedicated Python virtual environment
- **Web research** -- searches for state-of-the-art approaches before building
- **Multi-framework** -- scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, and more (installed as needed)
- **Auto-generated prediction API** -- creates a FastAPI endpoint ready to serve predictions
- **Experiment tracking** -- logs metrics, parameters, and artifacts for every model (with optional MLflow integration)
- **Model cards** -- generates standardized documentation for every trained model
- **Data versioning** -- hashes datasets for reproducibility tracking across builds
- **Resumable builds** -- saves build state so you can resume interrupted or incremental builds
- **Cost estimation** -- shows estimated token usage and cost before building starts
- **Deep learning support** -- generates PyTorch/TensorFlow scaffolds for complex tasks
- **Fairness checks** -- detects demographic bias across protected attributes
- **Session history** -- saves and restores conversations across sessions


## Quick Start

```bash
# Clone and set up
git clone https://github.com/matej-01RAI/ModelForge.git
cd ModelForge
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

Use your existing [Claude Code](https://claude.ai/code) subscription -- no separate API key needed.

```bash
# Prerequisites
npm install -g @anthropic-ai/claude-code
pip install langchain-claude-code
```

```env
PROVIDER=claude-code
CLAUDE_CODE_MODEL=sonnet   # opus, sonnet, or haiku
```

> **Auto-detection:** If you don't set `PROVIDER` explicitly, the agent picks the first available: `ANTHROPIC_API_KEY` -> Anthropic, `AZURE_AI_*` -> Azure, fallback -> Claude Code CLI.


## Commands

| Command | Description |
|---------|-------------|
| `/quit`, `/exit` | Exit the agent |
| `/clear` | Clear conversation history and start fresh |
| `/workspace` | Show the workspace directory path |
| `/tokens` | Show session token usage (input, output, cache, LLM calls) |
| `/build` | Skip approval prompt and start building immediately |
| `/sessions` | List saved sessions |
| `/load <id>` | Restore a previous session |
| `/save` | Save current session to disk |


## Architecture

```
ModelForge/
├── main.py                    # Terminal UI, spinner, token tracking, chat loop
├── config.py                  # Environment variable loading
├── setup.sh                   # One-command setup script
├── requirements.txt           # Python dependencies
├── agent/
│   ├── ml_agent.py            # Builder agent (LangChain AgentExecutor + tools)
│   ├── planning_agent.py      # Planner agent (conversational, no tools)
│   ├── llm_factory.py         # LLM provider abstraction
│   ├── session_history.py     # Session persistence across conversations
│   └── tools/
│       ├── terminal.py        # Shell command execution (600s timeout)
│       ├── file_manager.py    # Read, write, and list files
│       ├── web_research.py    # DuckDuckGo search + URL fetching
│       ├── data_analyzer.py   # Auto-analyze CSV/JSON/Parquet datasets
│       ├── api_generator.py   # FastAPI prediction endpoint generator
│       ├── experiment_tracker.py  # MLflow + JSON experiment logging
│       ├── model_card.py      # Standardized model documentation
│       ├── workspace_state.py # Save/load build state for resume
│       ├── cost_estimator.py  # Pre-build token/cost estimation
│       ├── data_versioner.py  # Dataset hashing for reproducibility
│       ├── deep_learning.py   # PyTorch/TensorFlow scaffold generator
│       └── fairness_check.py  # Demographic bias detection
├── workspaces/                # Generated at runtime — one directory per project
└── sessions/                  # Saved conversation sessions
```

### Agent Tools

| Tool | Icon | What it does |
|------|------|-------------|
| `run_terminal_command` | `>>` | Execute shell commands -- create venvs, install packages, run scripts |
| `read_file` | `<<` | Read file contents (up to 500KB) |
| `write_file` | `=>` | Create or overwrite files, auto-creates parent directories |
| `list_directory` | `[]` | Browse directory contents |
| `analyze_dataset` | `##` | Auto-analyze datasets -- types, stats, correlations, class balance |
| `search_web` | `??` | Search DuckDuckGo for ML papers, benchmarks, best practices |
| `fetch_url` | `<>` | Fetch and extract text from web pages |
| `generate_predict_api` | `~~` | Generate FastAPI prediction endpoint for trained models |
| `log_experiment` | `++` | Log metrics, params, and artifacts (JSON + optional MLflow) |
| `compare_experiments` | `==` | Compare all logged experiments in a table |
| `generate_model_card` | `[]` | Generate standardized model documentation |
| `save_build_state` | `->` | Save build progress for resume capability |
| `load_build_state` | `<-` | Load previous build state to resume |
| `version_dataset` | `##` | Hash dataset for reproducibility tracking |
| `check_dataset_changed` | `??` | Check if dataset has changed since last version |
| `generate_dl_scaffold` | `NN` | Generate PyTorch/TensorFlow training boilerplate |
| `check_fairness` | `**` | Run demographic bias checks on predictions |

### Workspace Structure

Each project the agent builds gets an isolated workspace:

```
workspaces/my_project/
├── venv/              # Dedicated Python virtual environment
├── data/              # Copy of training data
├── src/               # Training scripts + predict_api.py
├── models/            # Saved models (.pkl, .pth, .keras)
├── results/           # Metrics, plots, reports, fairness_report.md
├── experiments/       # Experiment logs (JSON + optional MLflow)
├── .versions/         # Dataset version hashes
├── .build_state.json  # Resumable build state
└── MODEL_CARD.md      # Standardized model documentation
```


## Security Considerations

This agent executes code autonomously on your machine. Understand these trade-offs before running it:

- **Shell execution** -- The `run_terminal_command` tool runs arbitrary shell commands with `shell=True`. This is by design (the agent needs to install packages, run scripts, etc.), but it means the agent has the same filesystem and network access as your user account.
- **File access** -- File tools can read/write anywhere your user has permissions. The agent is instructed to work within `workspaces/`, but this is a convention, not a hard sandbox.
- **Network access** -- The web research tools can make outgoing HTTP requests. Private/internal network URLs are blocked by default (SSRF protection), but the terminal tool has unrestricted network access.
- **API key** -- Your API key is stored in `.env` (gitignored). Never commit this file.

**Recommendations:**
- Run the agent in a container or VM if you need stronger isolation
- Review the agent's plan before approving builds on sensitive systems
- Monitor token usage with `/tokens` to control costs


## Customization

### Changing the LLM

Switch providers by updating your `.env` file -- see [Providers](#providers) above. The LLM is created by `agent/llm_factory.py`, which both agents share. To add a completely new provider (e.g., OpenAI, local models), add a new factory function there.

### Adding Tools

1. Create a tool file in `agent/tools/` using the `@tool` decorator
2. Import and add it to the `tools` list in `agent/ml_agent.py`
3. Add display icons/verbs in `LiveStatusCallback` in `main.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Tuning Agent Behavior

- **Builder prompt** -- `SYSTEM_PROMPT` in `agent/ml_agent.py` controls autonomous execution behavior
- **Planner prompt** -- `PLANNING_PROMPT` in `agent/planning_agent.py` controls question-asking and plan format
- **Temperature** -- Builder uses 0.1 (deterministic), planner uses 0.3 (slightly creative)
- **Max iterations** -- Builder allows up to 50 tool calls per turn


## Related

[ModelLens](https://getmodellens.com/) -- talk to your ML models


## Philosophy

Instead of building models manually and trying to interpret them later, we move towards:

**AI builds the model, and you can ask it what it learned.**


## Tested

Validated across 7 end-to-end builds covering classification, regression, multi-class, small datasets, and deep learning:

| Dataset | Task | Best Model | Key Metric |
|---------|------|------------|------------|
| Heart Disease (303 rows) | Binary classification | KNN | 88.5% accuracy, 100% recall |
| California Housing (20K rows) | Regression | XGBoost | R²=0.848 |
| Wine Quality (1.6K rows) | Multi-class (6 classes) | Random Forest | 66.9% accuracy |
| Iris (150 rows) | Small dataset | SVM (RBF) | 96.7% accuracy |
| Diabetes (442 rows) | Small regression | Lasso | R²=0.496 |
| California Housing (20K rows) | PyTorch neural net | 4-layer DNN | R²=0.806 |
| Wine Quality (1.6K rows) | TensorFlow neural net | Keras DNN | 65.3% accuracy |

Every build auto-generated: model card, prediction API, experiment logs, data versioning, and build state.


## Status

Active development. Core pipeline is stable and tested. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for details.


## License

[MIT](LICENSE)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
