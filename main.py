#!/usr/bin/env python3
"""ML Model Building Agent - Terminal Chat Interface.

An autonomous AI agent that builds machine learning models.
Uses Claude Sonnet (Azure AI Foundry) + LangChain for orchestration.
"""

import os
import sys
import signal
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from agent.ml_agent import create_agent
import config

custom_theme = Theme({
    "user": "bold cyan",
    "agent": "bold green",
    "system": "bold yellow",
    "error": "bold red",
})

console = Console(theme=custom_theme)


WELCOME_BANNER = """
# ML Model Building Agent

An autonomous AI agent that builds the best possible ML models for your data.

**Capabilities:**
- Analyzes your dataset and researches best approaches
- Builds models with PyTorch, TensorFlow, Keras, scikit-learn
- Tunes hyperparameters automatically
- Searches papers & benchmarks for state-of-the-art techniques
- Creates its own virtual environment and manages dependencies

**Commands:**
- Type your request to start building a model
- `/quit` or `/exit` to exit
- `/clear` to clear conversation history
- `/workspace` to show current workspace directory

**Example prompts:**
- "Build a classifier for this CSV: /path/to/data.csv, target column is 'label'"
- "Create a time series forecasting model for stock prices in data.parquet"
- "Build the best regression model for predicting house prices from housing.csv"
"""


def handle_sigint(sig, frame):
    console.print("\n\n[system]Interrupted. Type /quit to exit.[/system]")


def main():
    signal.signal(signal.SIGINT, handle_sigint)

    console.print(Panel(Markdown(WELCOME_BANNER), border_style="green", title="MLModelBuildingAgent"))

    # Validate config
    if not config.AZURE_AI_ENDPOINT or not config.AZURE_AI_API_KEY:
        console.print("[error]Missing Azure AI configuration. Copy .env.example to .env and fill in your credentials.[/error]")
        sys.exit(1)

    console.print(f"[system]Workspace: {config.WORKSPACE_DIR}[/system]")
    console.print(f"[system]Model: {config.AZURE_AI_MODEL} via Azure AI Foundry[/system]")
    console.print()

    # Create agent
    try:
        executor = create_agent()
        console.print("[system]Agent initialized successfully.[/system]\n")
    except Exception as e:
        console.print(f"[error]Failed to initialize agent: {e}[/error]")
        sys.exit(1)

    chat_history = []

    while True:
        try:
            console.print("[user]You:[/user] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[system]Goodbye![/system]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit"):
            console.print("[system]Goodbye![/system]")
            break
        elif user_input.lower() == "/clear":
            chat_history.clear()
            console.print("[system]Conversation history cleared.[/system]\n")
            continue
        elif user_input.lower() == "/workspace":
            console.print(f"[system]Workspace: {config.WORKSPACE_DIR}[/system]\n")
            continue

        # Run agent
        console.print()
        try:
            result = executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            output = result.get("output", "")

            # Display the response
            console.print(Panel(
                Markdown(output),
                border_style="green",
                title="Agent",
                title_align="left",
            ))

            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": output})

            # Show intermediate steps count
            steps = result.get("intermediate_steps", [])
            if steps:
                console.print(f"[system]({len(steps)} tool calls executed)[/system]")

        except KeyboardInterrupt:
            console.print("\n[system]Agent interrupted. You can continue chatting.[/system]")
        except Exception as e:
            console.print(f"[error]Agent error: {e}[/error]")
            console.print("[system]You can try again or rephrase your request.[/system]")

        console.print()


if __name__ == "__main__":
    main()
