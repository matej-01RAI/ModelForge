"""Workspace state tool - saves and loads build state for resume/incremental builds."""

import os
import json
import time
from langchain_core.tools import tool


@tool
def save_build_state(workspace_dir: str, state_data: str) -> str:
    """Save the current build state so it can be resumed later. Call this after each
    major phase completes (data analysis, environment setup, training, etc.).

    Args:
        workspace_dir: Path to the project workspace directory.
        state_data: JSON string with build state. Should include:
            - phase: Current phase name (e.g., "data_analysis", "training", "tuning", "delivery")
            - completed_steps: List of completed steps
            - dataset_path: Path to the dataset
            - target_column: Target column name
            - problem_type: Classification/regression/etc.
            - models_trained: List of model names already trained
            - best_model: Current best model name and metrics
            - notes: Any notes about the current state
    """
    abs_workspace = os.path.expanduser(workspace_dir)
    state_file = os.path.join(abs_workspace, ".build_state.json")

    try:
        state = json.loads(state_data)
    except (json.JSONDecodeError, TypeError):
        return f"[ERROR] Invalid state JSON: {state_data}"

    state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["workspace"] = abs_workspace

    os.makedirs(abs_workspace, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    return f"Build state saved to {state_file}"


@tool
def load_build_state(workspace_dir: str) -> str:
    """Load the saved build state from a workspace. Use this to check what was
    already done and resume from where the previous build left off.

    Args:
        workspace_dir: Path to the project workspace directory.
    """
    abs_workspace = os.path.expanduser(workspace_dir)
    state_file = os.path.join(abs_workspace, ".build_state.json")

    if not os.path.isfile(state_file):
        return "[INFO] No saved build state found. This is a fresh workspace."

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        lines = ["## Saved Build State\n"]
        lines.append(f"**Last updated:** {state.get('last_updated', 'unknown')}")
        lines.append(f"**Current phase:** {state.get('phase', 'unknown')}")
        lines.append(f"**Dataset:** {state.get('dataset_path', 'unknown')}")
        lines.append(f"**Target:** {state.get('target_column', 'unknown')}")
        lines.append(f"**Problem type:** {state.get('problem_type', 'unknown')}")

        completed = state.get("completed_steps", [])
        if completed:
            lines.append(f"\n**Completed steps:** {', '.join(completed)}")

        models = state.get("models_trained", [])
        if models:
            lines.append(f"**Models trained:** {', '.join(models)}")

        best = state.get("best_model", {})
        if best:
            lines.append(f"**Best model:** {json.dumps(best)}")

        notes = state.get("notes", "")
        if notes:
            lines.append(f"\n**Notes:** {notes}")

        lines.append(f"\n**Raw state:**\n```json\n{json.dumps(state, indent=2)}\n```")

        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] Failed to load build state: {e}"
