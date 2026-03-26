"""Terminal tool - executes shell commands in the agent's workspace."""

import subprocess
import os
from langchain_core.tools import tool


@tool
def run_terminal_command(command: str, workspace_dir: str = "") -> str:
    """Execute a shell command in the workspace directory. Use this to:
    - Create and manage Python virtual environments
    - Install packages with pip
    - Run Python scripts for training models
    - Inspect files and directories
    - Run any system command needed for ML workflow

    Args:
        command: The shell command to execute.
        workspace_dir: Working directory. If empty, uses current directory.
    """
    cwd = workspace_dir if workspace_dir and os.path.isdir(workspace_dir) else os.getcwd()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=600,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[EXIT CODE: {result.returncode}]"
        # Truncate very long outputs
        if len(output) > 15000:
            output = output[:7000] + "\n\n... [TRUNCATED] ...\n\n" + output[-7000:]
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 600 seconds."
    except Exception as e:
        return f"[ERROR] {e}"
