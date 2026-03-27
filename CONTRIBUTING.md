# Contributing to ML Model Building Agent

Thanks for your interest in contributing! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/MLModelBuildingAgent.git`
3. Run `bash setup.sh` to set up the development environment
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
bash setup.sh
source venv/bin/activate
cp .env.example .env
# Edit .env with your Azure AI Foundry credentials
python main.py
```

## Project Structure

```
agent/
  ml_agent.py          # Builder agent (LangChain + tools)
  planning_agent.py    # Planning agent (conversational)
  tools/
    terminal.py        # Shell command execution
    file_manager.py    # File read/write/list
    web_research.py    # Web search + URL fetch
    data_analyzer.py   # Dataset analysis
main.py                # CLI interface + status display
config.py              # Environment configuration
```

## How to Contribute

### Adding a New Tool

1. Create a new file in `agent/tools/` (e.g., `my_tool.py`)
2. Define your tool using the `@tool` decorator from `langchain_core.tools`
3. Import and add it to the `tools` list in `agent/ml_agent.py`
4. Add an icon and verb in `LiveStatusCallback` in `main.py`
5. Update the system prompt in `ml_agent.py` if needed

### Adding a New Feature

1. Open an issue first to discuss the change
2. Keep changes focused — one feature per PR
3. Update the README if your change affects user-facing behavior
4. Test your changes end-to-end with a real dataset

### Improving Prompts

The agent behavior is driven by two system prompts:
- `SYSTEM_PROMPT` in `agent/ml_agent.py` — controls the builder agent
- `PLANNING_PROMPT` in `agent/planning_agent.py` — controls the planner

When tweaking prompts, test with several different task types (classification, regression, small dataset, large dataset) to avoid regressions.

## Code Style

- Python 3.9+ compatible
- Keep functions small and focused
- Use type hints where they add clarity
- No unnecessary abstractions — simple is better

## Pull Requests

1. Make sure the agent runs end-to-end without errors
2. Describe what you changed and why
3. Include a before/after if you changed UI or status output
4. Keep PRs small and reviewable

## Reporting Issues

When opening an issue, please include:
- Python version (`python --version`)
- OS (macOS, Linux, Windows)
- The full error traceback if applicable
- Steps to reproduce

## Security

If you find a security vulnerability, please **do not** open a public issue. Instead, email the maintainers directly or use GitHub's private vulnerability reporting feature.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
