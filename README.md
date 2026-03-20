# Predictive Maintenance Platform (NASA C-MAPSS + VHACK Application)

This repository is split into two connected parts:

1. **Model development pipeline** (notebooks + `src/`): ingest, preprocess, train, evaluate, explain, and export artifacts.
2. **Application layer** (`vhack/` and API): consume exported artifacts for backend inference and frontend product workflows.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Technical Architecture](#technical-architecture)
- [Notebook-by-Notebook Workflow and Usage](#notebook-by-notebook-workflow-and-usage)
- [Artifact Contract (What Feeds Frontend/Backend)](#artifact-contract-what-feeds-frontendbackend)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Challenges Faced](#challenges-faced)
- [Future Roadmap](#future-roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## System Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│                    DATA & EXPERIMENT LAYER                       │
│   notebooks/ + src/ + data/raw/ + data/processed/ + artifacts/   │
└───────────────────────────────┬───────────────────────────────────┘
                                │ exports trained weights/pipelines
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    MODEL ARTIFACT LAYER                          │
│      models/saved/*.keras, *.weights.h5, *.joblib, *.npy        │
└───────────────────────────────┬───────────────────────────────────┘
                                │ consumed by
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ API Runtime (api/)           │        │ VHACK Product (vhack/)       │
│ /predict, /adapt, /machines  │        │ backend + frontend + app UI  │
└──────────────────────────────┘        └──────────────────────────────┘
```

---

## Features

- End-to-end RUL pipeline for FD001–FD004 datasets.
- Noise handling, missing-data simulation/imputation, and feature scaling.
- Baseline LSTM training and DANN-based cross-domain adaptation.
- Change-point detection and health-state classification.
- Explainability pipeline for feature attribution and operator-facing interpretation.
- Exported reusable model pipelines for API serving.
- Transfer-learning path for onboarding new machine domains.
- Product-facing VHACK application integration (frontend + backend).

---

## Tech Stack

- **Language:** Python
- **Data/ML:** NumPy, Pandas, SciPy, scikit-learn, TensorFlow/Keras, SHAP, Ruptures
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Serving:** FastAPI, Uvicorn, Pydantic
- **Apps:** Streamlit (root app and VHACK app)
- **Persistence:** Joblib, NPY, Keras/H5 model weights
- **Containerization:** Docker

---

## Technical Architecture

### Core modules (`src/`)

- `data_loader.py`: loads C-MAPSS train/test/RUL splits.
- `preprocessor.py`: smoothing, missing-value handling, normalization, piecewise RUL.
- `windowing.py`: sequence window generation for temporal models.
- `models/`: baseline and adaptation model definitions.
- `train.py`: model training loop(s), including adversarial training flow.
- `changepoint.py`: CUSUM-based transition detection and health state logic.
- `evaluate.py`: RMSE, MAE, NASA score, model comparison utilities.
- `explainer.py`: explainability helpers used in interpretation steps.
- `feature_aligner.py`, `fine_tuner.py`: adaptation/transfer-learning utilities.

### Serving modules (`api/`)

- `main.py`: API routes (`/health`, `/models`, `/predict`, `/adapt`, `/predict/adapted`, `/machines`).
- `predictor.py`: model registry/load/predict orchestration.
- `adapt.py`: machine-specific adaptation and adapted inference.
- `schemas.py`: request/response data contracts.

### Product modules (`vhack/`)

- `vhack/backend/`: backend services and routers.
- `vhack/frontend/`: frontend pages and app-level interactions.
- `vhack/app.py`, `vhack/streamlit_app.py`: app entry points.
- Uses artifacts produced by notebook/model pipeline.

---

## Notebook-by-Notebook Workflow and Usage

This is the exact progression of your model-development lifecycle and how each notebook contributes to the production stack.

### 1) `notebooks/01_data_exploration.ipynb`

**What you did**
- Loaded raw C-MAPSS datasets and validated structure/columns.
- Explored engine lifecycle distributions and sensor behavior.
- Identified dataset differences across FD subsets.

**Why it matters**
- Established baseline understanding of signal quality and operating regimes before preprocessing.

**Used by**
- Guides decisions in Notebook 02 (smoothing, imputation, normalization strategy).

---

### 2) `notebooks/02_preprocessing_noise_handling.ipynb`

**What you did**
- Applied noise reduction and missing-value handling.
- Built normalized training inputs and consistent feature processing.
- Prepared train-ready arrays/tables for downstream sequence modeling.

**Typical outputs**
- Processed datasets in `data/processed/` (for example window-ready arrays/cleaned tables).

**Used by**
- Notebook 03, 04, 05, and downstream API export path.

---

### 3) `notebooks/03_changepoint_anomaly_detection.ipynb`

**What you did**
- Implemented and tuned transition/anomaly detection (CUSUM-style logic).
- Estimated health-transition points across engine trajectories.
- Mapped transitions to health states for operations context.

**Typical outputs**
- Change-point/health diagnostics (plots/tables/artifacts as applicable).

**Used by**
- Prediction explainability and API response enrichment (`health_state`, transition signals).

---

### 4) `notebooks/04_baseline_lstm_rul.ipynb`

**What you did**
- Trained baseline sequence model for RUL prediction.
- Tuned/validated baseline behavior on canonical dataset setup.
- Established baseline metrics for future comparison.

**Typical outputs**
- Baseline model weights under `models/saved/`.

**Used by**
- Notebook 05 (as base for adaptation experiments), 06 (comparison), and 08 (export).

---

### 5) `notebooks/05_lstm_dann_domain_adaptation.ipynb`

**What you did**
- Ran domain-adversarial training for cross-domain robustness.
- Trained feature extractor + regressor + domain classifier setup.
- Benchmarked transfer/generalization behavior across dataset pairs.

**Typical outputs**
- DANN-related weights/adapters/scalers in `models/saved/`.

**Used by**
- Notebook 06 for comparative evaluation and Notebook 08/09 for deployable adaptation paths.

---

### 6) `notebooks/06_model_evaluation_comparison.ipynb`

**What you did**
- Compared baseline vs adaptation model families using common metrics.
- Consolidated RMSE/MAE/NASA score-style reporting.
- Assessed error profiles and trade-offs for deployment selection.

**Typical outputs**
- Evaluation summaries/charts used to choose production candidates.

**Used by**
- Notebook 08 model-export decisions and product readiness checks.

---

### 7) `notebooks/07_interpretability.ipynb`

**What you did**
- Generated feature-attribution analyses (SHAP-style workflow).
- Interpreted key sensor contributions to predicted RUL.
- Produced operator-facing explanatory evidence.

**Typical outputs**
- Explainability visual assets and feature-importance interpretation artifacts.

**Used by**
- API/business explanation logic and stakeholder trust/readability.

---

### 8) `notebooks/08_model_export_fastapi.ipynb`

**What you did**
- Wrapped trained components into deployable pipeline artifacts.
- Exported model/scaler/aligner bundles for runtime loading.
- Validated FastAPI serving contract against exported assets.

**Typical outputs**
- `models/saved/pm_pipeline_*.joblib`
- Runtime-compatible model files required by `api/`.

**Used by**
- `api/main.py` and `api/predictor.py` endpoints.
- Backend flows that power product-facing predictions.

---

### 9) `notebooks/09_mtda_multi_target_extension.ipynb`

**What you did**
- Extended adaptation to multi-target or broader transfer scenarios.
- Tested scaling adaptation logic beyond single source-target setup.
- Investigated robustness for real deployment diversity.

**Typical outputs**
- Extended adaptation artifacts/configuration candidates.

**Used by**
- Future-ready transfer learning pipeline and new-machine onboarding.

---

## Artifact Contract (What Feeds Frontend/Backend)

Your application layer relies on artifacts generated in the notebook pipeline.

### Artifact producers
- Notebooks 04/05/08/09 and related `src/` modules.

### Artifact consumers
- API: `api/predictor.py`, `api/adapt.py`
- Product stack: `vhack/backend/` services and frontend-driven prediction workflows.

### Key runtime expectation
- Exported files in `models/saved/` must match expected names and feature ordering assumptions.

---

## Project Structure

```text
.
├── app.py
├── api/
│   ├── main.py
│   ├── predictor.py
│   ├── adapt.py
│   └── schemas.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_noise_handling.ipynb
│   ├── 03_changepoint_anomaly_detection.ipynb
│   ├── 04_baseline_lstm_rul.ipynb
│   ├── 05_lstm_dann_domain_adaptation.ipynb
│   ├── 06_model_evaluation_comparison.ipynb
│   ├── 07_interpretability.ipynb
│   ├── 08_model_export_fastapi.ipynb
│   └── 09_mtda_multi_target_extension.ipynb
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved/
├── artifacts/
├── vhack/
│   ├── backend/
│   ├── frontend/
│   ├── app.py
│   └── streamlit_app.py
├── requirements.txt
└── Dockerfile
```

---

## Getting Started

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare data

Place C-MAPSS raw files in `data/raw/` (`train_*`, `test_*`, `RUL_*`).

### 3) Run notebook pipeline (recommended order)

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_preprocessing_noise_handling.ipynb`
3. `notebooks/03_changepoint_anomaly_detection.ipynb`
4. `notebooks/04_baseline_lstm_rul.ipynb`
5. `notebooks/05_lstm_dann_domain_adaptation.ipynb`
6. `notebooks/06_model_evaluation_comparison.ipynb`
7. `notebooks/07_interpretability.ipynb`
8. `notebooks/08_model_export_fastapi.ipynb`
9. `notebooks/09_mtda_multi_target_extension.ipynb` (optional/advanced)

### 4) Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Run apps

```bash
streamlit run app.py
streamlit run vhack/streamlit_app.py
```

---

## Environment Variables

No strict mandatory variables are required in the current default setup.

Common deployment variables:
- `PYTHONPATH`
- `HOST`
- `PORT`
- `MODELS_DIR` (if externalized in your launcher/scripts)

---

## API Reference

Base URL (local): `http://localhost:8000`

### `GET /health`
Returns API liveness.

### `GET /models`
Returns available canonical dataset models.

### `POST /predict`
Predicts RUL and health state using exported pipeline artifacts.

### `POST /adapt`
Fine-tunes for a new machine domain using uploaded labeled sequences.

### `POST /predict/adapted`
Runs prediction on an adapted machine-specific model.

### `GET /machines`
Lists adapted machine IDs available for inference.

---

## Challenges Faced

- Cross-domain drift between datasets and machine types.
- Preserving temporal signal while suppressing noise.
- Keeping feature alignment stable when onboarding non-canonical sensors.
- Ensuring artifact compatibility between notebook training and runtime APIs.
- Translating technical model outputs into operator-actionable explanations.

---

## Future Roadmap

- Add stronger experiment tracking/versioned model registry.
- Add CI for API schema checks and artifact load smoke tests.
- Expand explainability and monitoring endpoints.
- Harden production settings (auth, CORS policy, rate limits).
- Consolidate notebook-to-product handoff into one reproducible release pipeline.

---

## Troubleshooting

### Artifacts not found
- Confirm `models/saved/` contains expected exported files from Notebook 08.

### Inference mismatch
- Re-check feature order consistency between training pipeline and API input payloads.

### API startup issues
```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000
```

---

## License

Add a license file before public release (e.g., MIT or Apache-2.0).
