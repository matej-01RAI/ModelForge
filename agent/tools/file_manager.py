"""File management tools for reading/writing code and data files."""

import os
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file. Use this to inspect training data, scripts, configs, etc.

    Args:
        file_path: Absolute or relative path to the file to read.
    """
    try:
        path = os.path.expanduser(file_path)
        if not os.path.isfile(path):
            return f"[ERROR] File not found: {path}"
        size = os.path.getsize(path)
        if size > 500_000:
            return f"[WARNING] File is {size:,} bytes. Reading first 50KB.\n" + open(
                path, "r", errors="replace"
            ).read(50_000)
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.
    Use this to create Python training scripts, config files, etc.

    Args:
        file_path: Path where the file should be written.
        content: The full content to write to the file.
    """
    try:
        path = os.path.expanduser(file_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content):,} chars to {path}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def list_directory(dir_path: str) -> str:
    """List files and directories at the given path.

    Args:
        dir_path: Path to the directory to list.
    """
    try:
        path = os.path.expanduser(dir_path)
        if not os.path.isdir(path):
            return f"[ERROR] Not a directory: {path}"
        entries = sorted(os.listdir(path))
        result = []
        for e in entries[:200]:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                result.append(f"  [DIR]  {e}/")
            else:
                size = os.path.getsize(full)
                result.append(f"  [FILE] {e} ({size:,} bytes)")
        if len(entries) > 200:
            result.append(f"  ... and {len(entries) - 200} more entries")
        return "\n".join(result) or "(empty directory)"
    except Exception as e:
        return f"[ERROR] {e}"
