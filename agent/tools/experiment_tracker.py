"""Experiment tracking tool - logs metrics, params, and artifacts with MLflow."""

import os
import json
import time
from langchain_core.tools import tool


@tool
def log_experiment(
    workspace_dir: str,
    experiment_name: str,
    model_name: str,
    metrics: str,
    params: str = "",
    artifact_paths: str = "",
) -> str:
    """Log an ML experiment with metrics, parameters, and artifacts. Creates a structured
    experiment log in the workspace. If MLflow is installed, also logs to MLflow tracking.

    Args:
        workspace_dir: Path to the project workspace directory.
        experiment_name: Name of the experiment (e.g., "iris_classifier").
        model_name: Name/type of the model (e.g., "RandomForest", "XGBoost").
        metrics: JSON string of metrics (e.g., '{"accuracy": 0.95, "f1": 0.94}').
        params: JSON string of hyperparameters (e.g., '{"n_estimators": 100, "max_depth": 5}').
        artifact_paths: Comma-separated list of artifact file paths to log (e.g., "models/best_model.pkl,results/confusion_matrix.png").
    """
    abs_workspace = os.path.expanduser(workspace_dir)

    # Parse inputs
    try:
        metrics_dict = json.loads(metrics)
    except (json.JSONDecodeError, TypeError):
        return f"[ERROR] Invalid metrics JSON: {metrics}"

    params_dict = {}
    if params:
        try:
            params_dict = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return f"[ERROR] Invalid params JSON: {params}"

    artifacts = [a.strip() for a in artifact_paths.split(",") if a.strip()] if artifact_paths else []

    # Create experiment log directory
    experiments_dir = os.path.join(abs_workspace, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    # Generate run ID
    run_id = f"run_{int(time.time())}_{model_name.lower().replace(' ', '_')}"

    # Save structured experiment log
    experiment_log = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_dict,
        "params": params_dict,
        "artifacts": artifacts,
    }

    # Append to experiments log file
    log_file = os.path.join(experiments_dir, "experiments.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(experiment_log) + "\n")

    # Save individual run
    run_file = os.path.join(experiments_dir, f"{run_id}.json")
    with open(run_file, "w") as f:
        json.dump(experiment_log, f, indent=2)

    # Try MLflow logging
    mlflow_status = _try_mlflow_log(abs_workspace, experiment_name, model_name, metrics_dict, params_dict, artifacts, run_id)

    # Generate comparison table if multiple runs exist
    comparison = _generate_comparison(log_file)

    return (
        f"Experiment logged successfully.\n"
        f"  Run ID: {run_id}\n"
        f"  Log file: {log_file}\n"
        f"  Run details: {run_file}\n"
        f"  MLflow: {mlflow_status}\n\n"
        f"{comparison}"
    )


@tool
def compare_experiments(workspace_dir: str) -> str:
    """Compare all logged experiments in a workspace. Shows a table of all runs
    with their metrics side by side for easy comparison.

    Args:
        workspace_dir: Path to the project workspace directory.
    """
    abs_workspace = os.path.expanduser(workspace_dir)
    log_file = os.path.join(abs_workspace, "experiments", "experiments.jsonl")

    if not os.path.isfile(log_file):
        return "[INFO] No experiments logged yet. Use log_experiment to track runs."

    return _generate_comparison(log_file)


def _try_mlflow_log(workspace_dir, experiment_name, model_name, metrics, params, artifacts, run_id):
    """Try to log to MLflow if installed. Returns status string."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.path.join(workspace_dir, "mlruns"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_id):
            mlflow.set_tag("model_name", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for artifact in artifacts:
                abs_path = os.path.join(workspace_dir, artifact)
                if os.path.isfile(abs_path):
                    mlflow.log_artifact(abs_path)

        return "Logged to MLflow successfully"
    except ImportError:
        return "MLflow not installed (local JSON logging only). Install with: pip install mlflow"
    except Exception as e:
        return f"MLflow logging failed: {e} (local JSON log saved)"


def _generate_comparison(log_file):
    """Generate a comparison table from experiment logs."""
    if not os.path.isfile(log_file):
        return ""

    runs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not runs:
        return "No experiments to compare."

    # Collect all metric keys
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.get("metrics", {}).keys())
    all_metrics = sorted(all_metrics)

    # Build comparison table
    lines = ["## Experiment Comparison\n"]
    header = f"| {'Model':<25} | {'Run ID':<35} | " + " | ".join(f"{m:<12}" for m in all_metrics) + " |"
    separator = "|" + "-" * 27 + "|" + "-" * 37 + "|" + "|".join("-" * 14 for _ in all_metrics) + "|"
    lines.append(header)
    lines.append(separator)

    for run in runs:
        model = run.get("model_name", "?")[:25]
        rid = run.get("run_id", "?")[:35]
        values = []
        for m in all_metrics:
            v = run.get("metrics", {}).get(m, "")
            if isinstance(v, float):
                values.append(f"{v:<12.4f}")
            else:
                values.append(f"{str(v):<12}")
        row = f"| {model:<25} | {rid:<35} | " + " | ".join(values) + " |"
        lines.append(row)

    # Highlight best model per metric
    if len(runs) > 1:
        lines.append("")
        lines.append("### Best per metric:")
        for m in all_metrics:
            best_run = max(runs, key=lambda r: r.get("metrics", {}).get(m, float("-inf")) if isinstance(r.get("metrics", {}).get(m, 0), (int, float)) else float("-inf"))
            best_val = best_run.get("metrics", {}).get(m, "?")
            lines.append(f"  {m}: {best_run.get('model_name', '?')} ({best_val})")

    return "\n".join(lines)
