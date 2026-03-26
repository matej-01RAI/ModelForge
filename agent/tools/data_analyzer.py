"""Data analysis tool - analyzes datasets to inform model building decisions."""

import os
import json
from langchain_core.tools import tool


@tool
def analyze_dataset(file_path: str, workspace_dir: str = "") -> str:
    """Analyze a dataset file (CSV, JSON, Parquet) to understand its structure,
    statistics, and characteristics. This helps decide which model to build.

    Returns: column types, shape, missing values, basic statistics, class distribution
    for classification tasks, and correlation info.

    Args:
        file_path: Path to the dataset file.
        workspace_dir: Workspace directory (for resolving relative paths).
    """
    import subprocess

    path = os.path.expanduser(file_path)
    if not os.path.isabs(path) and workspace_dir:
        path = os.path.join(workspace_dir, path)

    if not os.path.isfile(path):
        return f"[ERROR] File not found: {path}"

    # Build a Python script that analyzes the data
    analysis_script = f'''
import sys
import json

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] pandas/numpy not installed. Install them first with: pip install pandas numpy")
    sys.exit(1)

path = {repr(path)}
ext = path.rsplit(".", 1)[-1].lower()

try:
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext == "json" or ext == "jsonl":
        df = pd.read_json(path, lines=(ext == "jsonl"))
    elif ext == "parquet":
        df = pd.read_parquet(path)
    elif ext in ("xls", "xlsx"):
        df = pd.read_excel(path)
    else:
        print(f"[WARNING] Unknown extension .{{ext}}, trying CSV")
        df = pd.read_csv(path)
except Exception as e:
    print(f"[ERROR] Could not read file: {{e}}")
    sys.exit(1)

report = []
report.append(f"## Dataset: {{path}}")
report.append(f"Shape: {{df.shape[0]:,}} rows x {{df.shape[1]}} columns")
report.append(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}} MB")

report.append("\\n### Column Types")
for col in df.columns:
    dtype = str(df[col].dtype)
    nunique = df[col].nunique()
    nulls = df[col].isnull().sum()
    report.append(f"  {{col}}: {{dtype}} ({{nunique}} unique, {{nulls}} nulls)")

report.append("\\n### Missing Values")
missing = df.isnull().sum()
if missing.any():
    for col in missing[missing > 0].index:
        pct = missing[col] / len(df) * 100
        report.append(f"  {{col}}: {{missing[col]}} ({{pct:.1f}}%)")
else:
    report.append("  No missing values")

report.append("\\n### Numeric Statistics")
desc = df.describe()
report.append(desc.to_string())

report.append("\\n### First 5 Rows")
report.append(df.head().to_string())

# Detect potential target columns
report.append("\\n### Potential Target Columns (low cardinality)")
for col in df.columns:
    if df[col].nunique() <= 20 and df[col].nunique() > 1:
        vc = df[col].value_counts()
        report.append(f"  {{col}} ({{df[col].nunique()}} classes): {{dict(vc.head(10))}}")

# Correlation for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) >= 2:
    report.append("\\n### Top Correlations")
    corr = df[num_cols].corr()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            pairs.append((num_cols[i], num_cols[j], abs(corr.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for c1, c2, v in pairs[:10]:
        report.append(f"  {{c1}} <-> {{c2}}: {{v:.3f}}")

print("\\n".join(report))
'''

    try:
        result = subprocess.run(
            ["python3", "-c", analysis_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR] {result.stderr}"
        return output.strip() or "[ERROR] No output from analysis script"
    except subprocess.TimeoutExpired:
        return "[ERROR] Analysis timed out after 120s. The dataset may be too large."
    except Exception as e:
        return f"[ERROR] {e}"
