"""Data versioning tool - hashes datasets and logs versions for reproducibility."""

import os
import hashlib
import json
import time
from langchain_core.tools import tool


def _hash_file(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file without loading it entirely into memory."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@tool
def version_dataset(workspace_dir: str, dataset_path: str, description: str = "") -> str:
    """Hash a dataset file and log its version. This creates a reproducibility record
    linking the dataset to the models trained on it. Call this before training starts.

    Args:
        workspace_dir: Path to the project workspace directory.
        dataset_path: Path to the dataset file.
        description: Optional description of this dataset version.
    """
    abs_workspace = os.path.expanduser(workspace_dir)
    abs_dataset = os.path.expanduser(dataset_path)

    if not os.path.isabs(abs_dataset):
        abs_dataset = os.path.join(abs_workspace, abs_dataset)

    if not os.path.isfile(abs_dataset):
        return f"[ERROR] Dataset not found: {abs_dataset}"

    # Compute hash
    file_hash = _hash_file(abs_dataset)
    file_size = os.path.getsize(abs_dataset)
    file_name = os.path.basename(abs_dataset)

    # Create version record
    version_record = {
        "file_name": file_name,
        "file_path": abs_dataset,
        "hash_sha256": file_hash,
        "size_bytes": file_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
    }

    # Save to versions log
    versions_dir = os.path.join(abs_workspace, ".versions")
    os.makedirs(versions_dir, exist_ok=True)

    log_file = os.path.join(versions_dir, "dataset_versions.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(version_record) + "\n")

    # Also save current version pointer
    current_file = os.path.join(versions_dir, "current_dataset.json")
    with open(current_file, "w") as f:
        json.dump(version_record, f, indent=2)

    return (
        f"Dataset versioned successfully.\n"
        f"  File: {file_name}\n"
        f"  Size: {file_size:,} bytes\n"
        f"  SHA-256: {file_hash[:16]}...{file_hash[-8:]}\n"
        f"  Log: {log_file}\n"
    )


@tool
def check_dataset_changed(workspace_dir: str, dataset_path: str) -> str:
    """Check if a dataset has changed since it was last versioned. Useful for
    deciding whether to retrain a model.

    Args:
        workspace_dir: Path to the project workspace directory.
        dataset_path: Path to the dataset file to check.
    """
    abs_workspace = os.path.expanduser(workspace_dir)
    abs_dataset = os.path.expanduser(dataset_path)

    if not os.path.isabs(abs_dataset):
        abs_dataset = os.path.join(abs_workspace, abs_dataset)

    if not os.path.isfile(abs_dataset):
        return f"[ERROR] Dataset not found: {abs_dataset}"

    current_hash = _hash_file(abs_dataset)

    # Load last known version
    current_file = os.path.join(abs_workspace, ".versions", "current_dataset.json")
    if not os.path.isfile(current_file):
        return (
            f"[INFO] No previous version found. This dataset has not been versioned yet.\n"
            f"Current hash: {current_hash[:16]}...{current_hash[-8:]}"
        )

    try:
        with open(current_file, "r") as f:
            last_version = json.load(f)

        last_hash = last_version.get("hash_sha256", "")
        last_time = last_version.get("timestamp", "unknown")

        if current_hash == last_hash:
            return (
                f"[OK] Dataset has NOT changed since last version.\n"
                f"  Hash: {current_hash[:16]}...{current_hash[-8:]}\n"
                f"  Last versioned: {last_time}"
            )
        else:
            return (
                f"[CHANGED] Dataset HAS changed since last version!\n"
                f"  Previous hash: {last_hash[:16]}...{last_hash[-8:]} (from {last_time})\n"
                f"  Current hash:  {current_hash[:16]}...{current_hash[-8:]}\n"
                f"  Retraining is recommended."
            )
    except Exception as e:
        return f"[ERROR] Could not check version: {e}"
