"""Fairness and bias detection tool - checks model predictions for demographic bias."""

import os
import json
from langchain_core.tools import tool


@tool
def check_fairness(
    workspace_dir: str,
    dataset_path: str,
    target_column: str,
    protected_columns: str,
    predictions_path: str = "",
    model_path: str = "",
) -> str:
    """Run fairness and bias checks on model predictions across protected attributes.
    Checks for demographic parity, equalized odds, and group-level performance disparities.

    Args:
        workspace_dir: Path to the project workspace directory.
        dataset_path: Path to the dataset (CSV) with true labels and protected attributes.
        target_column: Name of the target/label column.
        protected_columns: Comma-separated list of protected attribute columns (e.g., "gender,race,age_group").
        predictions_path: Optional path to a CSV with predictions. If empty, will use model_path to generate predictions.
        model_path: Optional path to the model file (pickle). Used if predictions_path is empty.
    """
    import subprocess

    abs_workspace = os.path.expanduser(workspace_dir)
    abs_dataset = os.path.expanduser(dataset_path)
    if not os.path.isabs(abs_dataset):
        abs_dataset = os.path.join(abs_workspace, abs_dataset)

    if not os.path.isfile(abs_dataset):
        return f"[ERROR] Dataset not found: {abs_dataset}"

    protected_cols = [c.strip() for c in protected_columns.split(",") if c.strip()]
    if not protected_cols:
        return "[ERROR] Must provide at least one protected column."

    # Build fairness analysis script
    script = f'''
import sys
import json
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] pandas/numpy not installed.")
    sys.exit(1)

dataset_path = {repr(abs_dataset)}
target_column = {repr(target_column)}
protected_columns = {repr(protected_cols)}
predictions_path = {repr(os.path.join(abs_workspace, predictions_path) if predictions_path else "")}
model_path = {repr(os.path.join(abs_workspace, model_path) if model_path else "")}

# Load data
df = pd.read_csv(dataset_path)

# Get predictions
if predictions_path and predictions_path.strip():
    pred_df = pd.read_csv(predictions_path)
    predictions = pred_df.iloc[:, 0].values
elif model_path and model_path.strip():
    import joblib
    model = joblib.load(model_path)
    feature_cols = [c for c in df.columns if c != target_column and c not in protected_columns]
    X = df[feature_cols]
    # Simple encoding for categorical
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    predictions = model.predict(X)
else:
    print("[ERROR] Must provide either predictions_path or model_path.")
    sys.exit(1)

y_true = df[target_column].values

# Convert to binary if needed (for classification metrics)
is_binary = len(set(y_true)) <= 2

report = []
report.append("# Fairness Analysis Report\\n")

for prot_col in protected_columns:
    if prot_col not in df.columns:
        report.append(f"## {{prot_col}}: Column not found in dataset\\n")
        continue

    report.append(f"## Protected Attribute: {{prot_col}}\\n")
    groups = df[prot_col].unique()

    group_metrics = {{}}
    for group in groups:
        mask = df[prot_col] == group
        group_true = y_true[mask]
        group_pred = predictions[mask]
        n = mask.sum()

        if n == 0:
            continue

        # Accuracy
        acc = np.mean(group_true == group_pred)

        # Positive prediction rate (demographic parity)
        if is_binary:
            pos_rate = np.mean(group_pred == 1) if len(set(group_pred)) <= 2 else np.mean(group_pred)
            # True positive rate (equalized odds)
            tp_mask = group_true == 1
            tpr = np.mean(group_pred[tp_mask] == 1) if tp_mask.sum() > 0 else 0
            # False positive rate
            fp_mask = group_true == 0
            fpr = np.mean(group_pred[fp_mask] == 1) if fp_mask.sum() > 0 else 0
        else:
            pos_rate = None
            tpr = None
            fpr = None

        group_metrics[str(group)] = {{
            "n": int(n),
            "accuracy": round(float(acc), 4),
            "positive_rate": round(float(pos_rate), 4) if pos_rate is not None else None,
            "true_positive_rate": round(float(tpr), 4) if tpr is not None else None,
            "false_positive_rate": round(float(fpr), 4) if fpr is not None else None,
        }}

    # Display results
    report.append("| Group | N | Accuracy | Pos Rate | TPR | FPR |")
    report.append("|-------|---|----------|----------|-----|-----|")
    for group, m in group_metrics.items():
        pr = f"{{m['positive_rate']:.4f}}" if m["positive_rate"] is not None else "N/A"
        tpr = f"{{m['true_positive_rate']:.4f}}" if m["true_positive_rate"] is not None else "N/A"
        fpr = f"{{m['false_positive_rate']:.4f}}" if m["false_positive_rate"] is not None else "N/A"
        report.append(f"| {{group}} | {{m['n']}} | {{m['accuracy']:.4f}} | {{pr}} | {{tpr}} | {{fpr}} |")

    report.append("")

    # Fairness metrics
    if is_binary and len(group_metrics) >= 2:
        pos_rates = [m["positive_rate"] for m in group_metrics.values() if m["positive_rate"] is not None]
        accs = [m["accuracy"] for m in group_metrics.values()]

        if pos_rates:
            dp_ratio = min(pos_rates) / max(pos_rates) if max(pos_rates) > 0 else 0
            report.append(f"**Demographic Parity Ratio:** {{dp_ratio:.4f}} (1.0 = perfect parity, >0.8 generally acceptable)")

        acc_diff = max(accs) - min(accs)
        report.append(f"**Accuracy Disparity:** {{acc_diff:.4f}} (0.0 = no disparity)")

        tprs = [m["true_positive_rate"] for m in group_metrics.values() if m["true_positive_rate"] is not None]
        if tprs:
            eo_diff = max(tprs) - min(tprs)
            report.append(f"**Equalized Odds (TPR gap):** {{eo_diff:.4f}} (0.0 = perfect equalized odds)")

        # Flag concerns
        report.append("")
        if pos_rates and dp_ratio < 0.8:
            report.append("⚠️  **WARNING:** Demographic parity ratio below 0.8 — potential bias detected.")
        if acc_diff > 0.1:
            report.append("⚠️  **WARNING:** Accuracy disparity above 10% — model performs unevenly across groups.")
        if tprs and eo_diff > 0.1:
            report.append("⚠️  **WARNING:** TPR gap above 10% — equalized odds violation detected.")

        if (not pos_rates or dp_ratio >= 0.8) and acc_diff <= 0.1 and (not tprs or eo_diff <= 0.1):
            report.append("✓ No major fairness concerns detected.")

    report.append("")

# Save report
output = "\\n".join(report)
print(output)

# Save as file
with open("{os.path.join(abs_workspace, 'results', 'fairness_report.md')}", "w") as f:
    f.write(output)

# Save metrics as JSON
with open("{os.path.join(abs_workspace, 'results', 'fairness_metrics.json')}", "w") as f:
    json.dump({{"protected_columns": protected_columns, "groups": group_metrics}}, f, indent=2)
'''

    # Ensure results dir exists
    os.makedirs(os.path.join(abs_workspace, "results"), exist_ok=True)

    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[EXIT CODE: {result.returncode}]"
        return output.strip() or "[ERROR] No output from fairness check"
    except subprocess.TimeoutExpired:
        return "[ERROR] Fairness check timed out after 120s."
    except Exception as e:
        return f"[ERROR] {e}"
