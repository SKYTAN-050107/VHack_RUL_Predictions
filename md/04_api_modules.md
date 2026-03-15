# 04 — API Modules: FastAPI Deployment

> **IDE Agent Instructions:** Create each file at the exact path shown under `### Create File:`. All files go inside the `api/` directory of the project root.

---

## 4.1 — Pydantic Schemas

### Create File: `api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    """
    Request body for the /predict endpoint.

    Provide the full sensor history for ONE engine unit as a 2D list:
      readings[i] = one cycle = [op_setting_1, op_setting_2, op_setting_3,
                                  sensor_1, sensor_2, ..., sensor_21]

    The list must be ordered by cycle (ascending).
    Each inner list must have exactly 24 values.
    """
    unit_id:    str   = Field(..., example="engine_001")
    dataset_id: str   = Field("FD001", example="FD001",
                               description="Which dataset scaler/model to use. "
                                           "One of: FD001, FD002, FD003, FD004")
    readings: List[List[float]] = Field(
        ...,
        description="2D list of shape (n_cycles, 24). "
                    "Columns: op_setting_1-3, sensor_1-21 in order."
    )


class PredictResponse(BaseModel):
    """Response body from the /predict endpoint."""
    unit_id:              str
    rul_prediction:       float  = Field(..., description="Predicted remaining cycles")
    health_state:         str    = Field(..., description="Healthy | Warning | Critical")
    change_point_detected: bool  = Field(..., description="Whether CUSUM flagged a transition")
    change_point_step:    Optional[int] = Field(
        None,
        description="Step index within the last 50 cycles where change was detected"
    )
    explanation:          str    = Field(..., description="Plain-language insight for operators")


class ModelListResponse(BaseModel):
    """Response for the /models endpoint."""
    available_datasets: List[str]
```

---

## 4.2 — Predictor (Inference Logic)

### Create File: `api/predictor.py`

```python
import numpy as np
import joblib
from pathlib import Path
from typing import Optional


# ── In-memory model registry: loaded once per process, reused per request ─────
_MODEL_REGISTRY: dict = {}


def load_pipeline(dataset_id: str, models_dir: str = 'models/saved'):
    """
    Load and cache the PredictiveMaintenancePipeline for a given dataset.

    The pipeline is loaded from disk only once (on first request for that
    dataset_id) and cached in _MODEL_REGISTRY for subsequent requests.

    Args:
        dataset_id : e.g. 'FD001'
        models_dir : Directory containing .joblib files

    Returns:
        PredictiveMaintenancePipeline instance

    Raises:
        FileNotFoundError if the model file does not exist
    """
    key = dataset_id.upper()
    if key not in _MODEL_REGISTRY:
        path = Path(models_dir) / f'pm_pipeline_{key.lower()}.joblib'
        if not path.exists():
            raise FileNotFoundError(
                f"No model found for dataset '{key}'. "
                f"Expected file: {path}. "
                f"Run Notebook 08 to export the model first."
            )
        _MODEL_REGISTRY[key] = joblib.load(path)
    return _MODEL_REGISTRY[key]


def build_explanation(rul: float, health_state: str,
                       change_point_detected: bool) -> str:
    """
    Generate a plain-language insight string for factory operators.

    Args:
        rul                   : Predicted RUL in cycles
        health_state          : 'Healthy', 'Warning', or 'Critical'
        change_point_detected : Whether CUSUM flagged a state transition

    Returns:
        Human-readable string explaining the prediction
    """
    if health_state == 'Critical':
        urgency = (
            f"CRITICAL ALERT: Engine has approximately {rul:.0f} cycles "
            f"remaining before failure. Schedule maintenance IMMEDIATELY."
        )
    elif health_state == 'Warning':
        urgency = (
            f"WARNING: Engine health has degraded. "
            f"Approximately {rul:.0f} cycles remaining. "
            f"Plan maintenance within the next operational window."
        )
    else:
        urgency = (
            f"Engine is operating normally. "
            f"Estimated {rul:.0f} cycles remaining before scheduled maintenance."
        )

    cp_note = ""
    if change_point_detected:
        cp_note = (" A sensor deviation from the healthy baseline was detected "
                   "in recent cycles, confirming the transition to an impaired state.")

    return urgency + cp_note


def run_prediction(unit_id: str, dataset_id: str,
                    readings: list,
                    models_dir: str = 'models/saved') -> dict:
    """
    End-to-end inference: load model → predict → format response.

    Args:
        unit_id    : Engine identifier (for response labelling)
        dataset_id : Which model/scaler to use
        readings   : List of lists, shape (n_cycles, n_features)
        models_dir : Path to saved model directory

    Returns:
        dict matching PredictResponse schema
    """
    pipeline = load_pipeline(dataset_id, models_dir)
    X_raw    = np.array(readings, dtype=np.float32)
    result   = pipeline.predict(X_raw)

    rul          = result['rul_prediction']
    health_state = result['health_state']
    cp_detected  = result['change_point_detected']
    cp_step      = result.get('change_point_step')

    return {
        'unit_id':               unit_id,
        'rul_prediction':        round(rul, 1),
        'health_state':          health_state,
        'change_point_detected': cp_detected,
        'change_point_step':     cp_step,
        'explanation':           build_explanation(rul, health_state, cp_detected)
    }


def list_available_models(models_dir: str = 'models/saved') -> list:
    """Return list of dataset IDs for which .joblib models exist."""
    path = Path(models_dir)
    if not path.exists():
        return []
    return [
        p.stem.replace('pm_pipeline_', '').upper()
        for p in path.glob('pm_pipeline_*.joblib')
    ]
```

---

## 4.3 — FastAPI Application

### Create File: `api/main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas   import PredictRequest, PredictResponse, ModelListResponse
from .predictor import run_prediction, list_available_models


# ── App Configuration ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Modular RUL prediction and health state detection for industrial machinery. "
        "Supports multiple machine types via the dataset_id parameter. "
        "Each dataset_id maps to a separately trained LSTM model and scaler."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns OK if the server is running."""
    return {"status": "ok", "service": "predictive-maintenance-api"}


@app.get("/models", response_model=ModelListResponse, tags=["System"])
def list_models():
    """
    List all available dataset-specific models.

    Returns the dataset IDs for which a trained .joblib pipeline exists
    in the models/saved/ directory.
    """
    available = list_available_models()
    return {"available_datasets": available}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_rul(request: PredictRequest):
    """
    Predict Remaining Useful Life (RUL) and health state for one engine unit.

    **Request body:**
    - `unit_id`    : Identifier string for the engine
    - `dataset_id` : Which machine model to use (e.g. 'FD001')
    - `readings`   : Full sensor history as a 2D list (n_cycles × 24 features)
                     Order: op_setting_1, op_setting_2, op_setting_3,
                            sensor_1 … sensor_21

    **Response:**
    - `rul_prediction`       : Estimated cycles remaining
    - `health_state`         : Healthy | Warning | Critical
    - `change_point_detected`: Whether a health state transition was detected
    - `change_point_step`    : Step index of the transition (if detected)
    - `explanation`          : Plain-language insight for operators
    """
    if not request.readings:
        raise HTTPException(
            status_code=422,
            detail="'readings' must contain at least one cycle of sensor data."
        )
    if any(len(row) != 24 for row in request.readings):
        raise HTTPException(
            status_code=422,
            detail="Each reading must have exactly 24 values: "
                   "op_setting_1-3 followed by sensor_1-21."
        )
    try:
        result = run_prediction(
            unit_id    = request.unit_id,
            dataset_id = request.dataset_id,
            readings   = request.readings
        )
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

---

## 4.4 — How to Run the API

```bash
# From the project root directory:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open: http://localhost:8000/docs (Swagger UI)

---

## 4.5 — Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "engine_001",
    "dataset_id": "FD001",
    "readings": [
      [0.0025, 0.0003, 100.0, 518.67, 641.82, 1589.7, 1400.6, 14.62,
       21.61, 554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66, 2388.02,
       8138.62, 8.4195, 0.03, 392, 2388.0, 100.0, 39.06, 23.419],
      [0.0023, 0.0003, 100.0, 518.67, 642.15, 1591.82, 1403.14, 14.62,
       21.61, 553.75, 2388.07, 9044.07, 1.3, 47.49, 522.28, 2388.03,
       8131.49, 8.4318, 0.03, 392, 2388.0, 100.0, 38.86, 23.415]
    ]
  }'
```

Expected response shape:
```json
{
  "unit_id": "engine_001",
  "rul_prediction": 87.3,
  "health_state": "Healthy",
  "change_point_detected": false,
  "change_point_step": null,
  "explanation": "Engine is operating normally. Estimated 87 cycles remaining before scheduled maintenance."
}
```
