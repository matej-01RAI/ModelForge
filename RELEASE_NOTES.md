# Release Notes: v2.0 -- The Full Pipeline Release

## Overview

ModelForge v2 transforms the agent from a model builder into a **complete ML pipeline generator**. Every build now produces not just a trained model, but a production-ready prediction API, standardized documentation, experiment logs, and reproducibility tracking -- all automatically.


## New Features

### Auto-Generated Prediction API
After training, the agent generates a FastAPI endpoint (`src/predict_api.py`) with:
- `POST /predict` -- single prediction
- `POST /predict/batch` -- batch predictions
- `GET /health` -- health check
- `GET /docs` -- interactive Swagger UI

Supports sklearn (pickle/joblib), PyTorch (.pth), and TensorFlow (.keras) models.

### Experiment Tracking
Every model the agent trains is logged with metrics, hyperparameters, and artifact paths. Supports:
- **Local JSON logs** -- always available, no dependencies
- **MLflow integration** -- optional, logs to local MLflow tracking server when installed

Use `compare_experiments` to see a side-by-side comparison table of all runs.

### Model Cards
Every build generates a `MODEL_CARD.md` following ML model documentation standards:
- Model details and intended use
- Training data description
- Performance metrics
- Limitations and ethical considerations
- Usage code snippets

### Resumable Builds
Build state is saved after each major phase via `.build_state.json`. If a build is interrupted or you want to iterate on an existing workspace, the agent can pick up where it left off instead of starting from scratch.

### Cost Estimation
Before building starts, ModelForge shows an estimated token count and USD cost based on the plan complexity, number of models, and active provider pricing. Helps you decide before spending.

### Deep Learning Support
The agent can now generate PyTorch and TensorFlow/Keras training scaffolds with:
- Proper data loading and preprocessing
- Model architecture with BatchNorm, dropout, and regularization
- Training loops with early stopping and learning rate scheduling
- Evaluation and model saving

For tabular data, sklearn remains the default. Deep learning activates for larger datasets or when explicitly requested.

### Data Versioning
Datasets are SHA-256 hashed before training and logged in `.versions/`. Use `check_dataset_changed` to detect if data has been modified since the last build -- useful for deciding when to retrain.

### Session History
Conversations are auto-saved after each build and can be restored across sessions:
- `/save` -- manually save current session
- `/sessions` -- list saved sessions
- `/load <id>` -- restore a previous session with full chat history

### Fairness & Bias Detection
When protected attributes are present (gender, race, age group, etc.), the agent can run bias checks:
- Demographic parity ratio
- Equalized odds (TPR gap)
- Accuracy disparity across groups
- Automatic warnings when thresholds are violated


## Test Results

Validated across 7 end-to-end tests covering all major ML scenarios:

| Test | Dataset | Task | Best Model | Key Metric | Tools Used |
|------|---------|------|------------|------------|------------|
| Heart Disease | UCI (303 rows) | Binary classification | KNN | 88.5% acc, 100% recall | 35 steps |
| California Housing | sklearn (20,640 rows) | Regression | XGBoost Tuned | R²=0.848 | 32 steps |
| Wine Quality | UCI (1,599 rows) | Multi-class (6 classes) | Random Forest | 66.9% acc | 38 steps |
| Iris | sklearn (150 rows) | Small dataset classification | SVM (RBF) | 96.7% acc | 10-fold CV |
| Diabetes | sklearn (442 rows) | Small regression | Lasso Tuned | R²=0.496 | 9 experiments |
| California Housing | sklearn (20,640 rows) | PyTorch neural net | 4-layer DNN | R²=0.806 | PyTorch |
| Wine Quality | UCI (1,599 rows) | TensorFlow neural net | Keras DNN | 65.3% acc | 3 iterations |

All v2 features (model card, predict API, experiment tracking, data versioning, build state) were used in every test.


## Breaking Changes

- `requirements.txt` updated -- now requires `langchain-classic>=1.0.0` for LangChain 1.x compatibility
- Agent now has 17 tools (up from 7) -- system prompt significantly expanded


## Files Changed

### New Files (10)
- `agent/tools/api_generator.py` -- FastAPI prediction endpoint generator
- `agent/tools/experiment_tracker.py` -- experiment logging and comparison
- `agent/tools/model_card.py` -- model card documentation generator
- `agent/tools/workspace_state.py` -- build state save/load for resume
- `agent/tools/cost_estimator.py` -- pre-build cost estimation
- `agent/tools/deep_learning.py` -- PyTorch/TensorFlow scaffold generator
- `agent/tools/data_versioner.py` -- dataset hashing and version tracking
- `agent/tools/fairness_check.py` -- demographic bias detection
- `agent/session_history.py` -- session persistence across conversations
- `tests/run_test.py` -- automated test runner

### Modified Files (6)
- `agent/ml_agent.py` -- registered 10 new tools, expanded system prompt with new phases
- `main.py` -- cost estimation display, session commands, auto-save, new tool icons
- `requirements.txt` -- updated for LangChain 1.x compatibility
- `README.md` -- documented all new features, tools, commands, workspace structure
- `CONTRIBUTING.md` -- updated project name
- `setup.sh` -- updated project name
