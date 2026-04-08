"""API generator tool - creates FastAPI prediction endpoints from trained models."""

import os
from langchain_core.tools import tool


@tool
def generate_predict_api(workspace_dir: str, model_path: str, feature_names: str, model_type: str = "sklearn") -> str:
    """Generate a FastAPI prediction API for a trained model. Creates a ready-to-run
    predict_api.py file that loads the model and serves predictions via HTTP.

    Args:
        workspace_dir: Path to the project workspace directory.
        model_path: Relative path to the saved model file within the workspace (e.g., models/best_model.pkl).
        feature_names: Comma-separated list of input feature names (e.g., "age,income,score").
        model_type: Type of model - "sklearn" for pickle/joblib, "pytorch" for .pth files, "tensorflow" for .h5/.keras files.
    """
    features = [f.strip() for f in feature_names.split(",") if f.strip()]
    if not features:
        return "[ERROR] feature_names must be a comma-separated list of input features."

    abs_workspace = os.path.expanduser(workspace_dir)
    abs_model = os.path.join(abs_workspace, model_path)
    if not os.path.isfile(abs_model):
        return f"[ERROR] Model file not found: {abs_model}"

    # Build feature fields for Pydantic model
    feature_fields = "\n".join(f"    {f}: float" for f in features)
    feature_dict = "{" + ", ".join(f'"{f}": data.{f}' for f in features) + "}"

    if model_type == "sklearn":
        loader_code = _sklearn_loader(model_path)
    elif model_type == "pytorch":
        loader_code = _pytorch_loader(model_path)
    elif model_type == "tensorflow":
        loader_code = _tensorflow_loader(model_path)
    else:
        return f"[ERROR] Unknown model_type: {model_type}. Use 'sklearn', 'pytorch', or 'tensorflow'."

    api_code = f'''"""ModelForge Prediction API - Auto-generated FastAPI endpoint."""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="ModelForge Prediction API",
    description="Auto-generated prediction endpoint for trained ML model.",
    version="1.0.0",
)

# ── Request/Response schemas ────────────────────────────────────────────

class PredictionRequest(BaseModel):
{feature_fields}

class PredictionResponse(BaseModel):
    prediction: str | float | int
    probability: list[float] | None = None

# ── Model loading ───────────────────────────────────────────────────────

{loader_code}

# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = {feature_dict}
    X = np.array([[features[k] for k in [{", ".join(f'"{f}"' for f in features)}]]])

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]

    # Try to get probability
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(X)[0].tolist()
        except Exception:
            pass

    return PredictionResponse(
        prediction=prediction if not isinstance(prediction, np.generic) else prediction.item(),
        probability=probability,
    )

@app.post("/predict/batch")
def predict_batch(data: list[PredictionRequest]):
    """Predict for multiple samples at once."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = []
    for d in data:
        features = {feature_dict.replace("data.", "d.")}
        rows.append([features[k] for k in [{", ".join(f'"{f}"' for f in features)}]])

    X = np.array(rows)
    if scaler is not None:
        X = scaler.transform(X)

    predictions = model.predict(X)
    results = []
    for i, pred in enumerate(predictions):
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X[i:i+1])[0].tolist()
            except Exception:
                pass
        results.append({{
            "prediction": pred if not isinstance(pred, np.generic) else pred.item(),
            "probability": prob,
        }})

    return {{"predictions": results}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    api_path = os.path.join(abs_workspace, "src", "predict_api.py")
    os.makedirs(os.path.dirname(api_path), exist_ok=True)
    with open(api_path, "w") as f:
        f.write(api_code)

    # Generate a requirements file for the API
    api_reqs = "fastapi>=0.104.0\nuvicorn>=0.24.0\nnumpy\n"
    if model_type == "sklearn":
        api_reqs += "scikit-learn\njoblib\n"
    elif model_type == "pytorch":
        api_reqs += "torch\n"
    elif model_type == "tensorflow":
        api_reqs += "tensorflow\n"

    reqs_path = os.path.join(abs_workspace, "requirements_api.txt")
    with open(reqs_path, "w") as f:
        f.write(api_reqs)

    return (
        f"Generated prediction API at: {api_path}\n"
        f"API requirements at: {reqs_path}\n\n"
        f"To run:\n"
        f"  pip install -r {reqs_path}\n"
        f"  python {api_path}\n\n"
        f"Endpoints:\n"
        f"  GET  /health         - Health check\n"
        f"  POST /predict        - Single prediction\n"
        f"  POST /predict/batch  - Batch predictions\n"
        f"  GET  /docs           - Interactive API docs (Swagger UI)\n"
    )


def _sklearn_loader(model_path: str) -> str:
    return f'''import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "{model_path}")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {{MODEL_PATH}}")
except Exception as e:
    print(f"WARNING: Could not load model: {{e}}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded from {{SCALER_PATH}}")
except Exception:
    scaler = None'''


def _pytorch_loader(model_path: str) -> str:
    return f'''import torch

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "{model_path}")
scaler = None  # PyTorch models handle their own normalization

try:
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"Model loaded from {{MODEL_PATH}}")
except Exception as e:
    print(f"WARNING: Could not load model: {{e}}")
    model = None'''


def _tensorflow_loader(model_path: str) -> str:
    return f'''import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "{model_path}")
scaler = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {{MODEL_PATH}}")
except Exception as e:
    print(f"WARNING: Could not load model: {{e}}")
    model = None'''
