# Predictive Maintenance Platform for RUL Estimation (NASA C-MAPSS)

An end-to-end predictive maintenance system that estimates Remaining Useful Life (RUL), detects degradation transitions, supports cross-domain adaptation, and serves predictions through both a Streamlit UI and FastAPI.

## Table of Contents

- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Technical Architecture](#technical-architecture)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Challenges Faced](#challenges-faced)
- [Future Roadmap](#future-roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## System Architecture

```text
┌────────────────────────────┐
│  Data Layer                │
│  data/raw, data/processed  │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│  ML Core (src/)            │
│  loading/preprocess/window │
│  changepoint/train/eval    │
│  models + adaptation       │
└───────┬─────────┬──────────┘
        │         │
        │         ├──────────────────────────────┐
        │                                        ▼
        ▼                              ┌──────────────────────┐
┌────────────────────────────┐         │ FastAPI Service      │
│ Streamlit UI (app.py)      │         │ api/main.py          │
│ interactive 9-step workflow│         │ /predict /adapt ...  │
└────────────────────────────┘         └──────────┬───────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────────┐
                                         │ Deployed Artifacts   │
                                         │ models/saved/*.joblib│
                                         │ *.keras, *.weights   │
                                         └──────────────────────┘
```

## Features

- Multi-dataset C-MAPSS workflow (FD001–FD004).
- End-to-end preprocessing: smoothing, imputation, normalization, RUL target generation.
- Baseline LSTM training and domain-adversarial learning (DANN).
- Change-point detection and health-state classification (`Healthy`, `Warning`, `Critical`).
- Model explainability utilities (SHAP-oriented workflow).
- Transfer learning for new machine types with feature alignment.
- Exportable prediction pipelines (`pm_pipeline_*.joblib`).
- Interactive Streamlit app for full workflow execution.
- FastAPI inference endpoints for production-style serving.

## Tech Stack

- **Language:** Python 3.11+
- **ML / Data:** NumPy, Pandas, SciPy, scikit-learn, TensorFlow/Keras, SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **App/API:** Streamlit, FastAPI, Uvicorn, Pydantic
- **Utilities:** Joblib, TQDM, Ruptures
- **Environment / Packaging:** pip, virtualenv, Docker

## Technical Architecture

### 1) Data & Preprocessing Layer

- `src/data_loader.py`: dataset ingestion for C-MAPSS splits.
- `src/preprocessor.py`: filtering, imputation, normalization, piecewise RUL labeling.
- `src/windowing.py`: sequence window generation for temporal models.

### 2) Modeling Layer

- `src/models/lstm_baseline.py`: baseline temporal regressor.
- `src/models/lstm_dann.py`, `src/train.py`: domain-adversarial training components.
- `src/changepoint.py`: CUSUM-based transition detection and health logic.
- `src/evaluate.py`: RMSE / NASA-style score computation.

### 3) Adaptation Layer

- `src/feature_aligner.py`: aligns non-C-MAPSS feature spaces to expected model input.
- `src/fine_tuner.py`: few-shot two-phase fine-tuning.
- `src/models/adaptive_lstm.py`: adapted pipeline abstraction for machine-specific models.

### 4) Serving Layer

- `api/main.py`: REST API endpoints (`/health`, `/models`, `/predict`, `/adapt`, `/predict/adapted`, `/machines`).
- `api/predictor.py`: model registry, loading, prediction orchestration.
- `api/adapt.py`: adaptation and adapted-inference execution.
- `api/schemas.py`: request/response contracts.

### 5) UX Layer

- `app.py`: guided 9-step Streamlit interface from ingestion to export and transfer learning.

## Implementation Details

### Core Data Flow

1. Load raw engine run-to-failure records from `data/raw`.
2. Apply per-engine preprocessing and scale features.
3. Build fixed-length windows for sequence models.
4. Train/evaluate baseline and adaptation-capable models.
5. Export reusable pipelines and model weights.
6. Serve inference via FastAPI and validate via Streamlit UI.

### Model Persistence

- Scalers, aligners, and pipelines are serialized with Joblib.
- Neural network weights are saved in `.keras` or `.weights.h5` formats.
- Runtime inference primarily resolves `models/saved/pm_pipeline_*.joblib`.

### API Inference Contract

- `/predict` expects cycles as a 2D array with 24 ordered features per row:
  - `op_setting_1..3` + `sensor_1..21`.
- The response includes:
  - `rul_prediction`, `health_state`, change-point metadata, and operator-friendly explanation text.

## Project Structure

```text
.
├── app.py                         # Main Streamlit workflow app
├── api/
│   ├── main.py                    # FastAPI routes
│   ├── predictor.py               # Inference/model loading
│   ├── adapt.py                   # Transfer-learning API logic
│   └── schemas.py                 # Pydantic schemas
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── windowing.py
│   ├── changepoint.py
│   ├── evaluate.py
│   ├── explainer.py
│   ├── feature_aligner.py
│   ├── fine_tuner.py
│   ├── train.py
│   └── models/
├── notebooks/                     # Experiment and pipeline notebooks
├── data/
│   ├── raw/
│   └── processed/
├── models/saved/                  # Exported weights/pipelines
├── artifacts/                     # Generated artifacts
├── requirements.txt
└── Dockerfile
```

## Getting Started

### Prerequisites

- Python 3.11+
- macOS/Linux/Windows
- Optional: Docker

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare data

Place C-MAPSS files under `data/raw/`:

- `train_FD001.txt` ... `train_FD004.txt`
- `test_FD001.txt` ... `test_FD004.txt`
- `RUL_FD001.txt` ... `RUL_FD004.txt`

### 3) Run Streamlit app

```bash
streamlit run app.py
```

### 4) Run FastAPI service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) API docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 6) Docker (notebook-centric image)

```bash
docker build -t pm-rul .
docker run --rm -it -p 8888:8888 -v "$PWD":/app pm-rul
```

## Environment Variables

No mandatory environment variables are required by default.

Optional/conventional variables you may use in deployment scripts:

- `PYTHONPATH` (commonly set to project root, e.g., `/app` in Docker)
- `HOST` and `PORT` (if wrapping Uvicorn startup in custom scripts)
- `MODELS_DIR` (if you externalize model directory in your own launcher)

## API Reference

Base URL (local): `http://localhost:8000`

### `GET /health`

Returns service liveness.

### `GET /models`

Returns available canonical dataset models (e.g., `FD001`–`FD004`) found in `models/saved`.

### `POST /predict`

Predict RUL and health state for one unit.

**Request body**

```json
{
  "unit_id": "engine_001",
  "dataset_id": "FD001",
  "readings": [[0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]]
}
```

### `POST /adapt`

Fine-tune from a base dataset model for a new machine domain.

### `POST /predict/adapted`

Predict using an adapted machine-specific pipeline.

### `GET /machines`

List available adapted machine IDs.

## Challenges Faced

- Handling domain shift across FD datasets and real-like machine feature spaces.
- Keeping sequence model input shape consistent while supporting variable external sensors.
- Preserving temporal trends while reducing noise and missing-value impact.
- Exporting reproducible model pipelines that remain compatible between notebooks, Streamlit, and API runtime.
- Balancing model performance with explainability and operational readability.

## Future Roadmap

- Unify config management (single source for paths, thresholds, hyperparameters).
- Add formal automated tests for API schema validation and inference contract checks.
- Add CI pipeline for linting, type checks, and smoke inference tests.
- Improve artifact/version tracking for reproducible experiment lineage.
- Add auth/rate-limiting and stricter CORS policies for production API deployment.
- Expand explainability endpoints and model monitoring hooks.

## Troubleshooting

### Model not found errors

Ensure exported pipelines/weights exist under `models/saved/` and names match expected patterns:

- `pm_pipeline_fd001.joblib` (or canonical equivalents)
- `lstm_target_only_FD001.keras`

### FastAPI cannot start

Install runtime dependencies and verify module path:

```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000
```

### Streamlit cannot load data

Ensure all required C-MAPSS raw files are present in `data/raw/`.

## License

Add your preferred license (e.g., MIT, Apache-2.0) before public distribution.
