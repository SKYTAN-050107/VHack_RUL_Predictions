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

# ──────────────────────────────────────────────────────────────────────────────
# APPEND THESE LINES TO THE BOTTOM OF: api/main.py
# Do not replace anything — just paste from here to the end of the file
# ──────────────────────────────────────────────────────────────────────────────

from .schemas import AdaptRequest, AdaptResponse, AdaptPredictRequest
from .adapt   import run_adaptation, run_adapted_prediction


@app.post("/adapt", response_model=AdaptResponse, tags=["Transfer Learning"])
def adapt_model(request: AdaptRequest):
    """
    Fine-tune the pretrained LSTM on data from a NEW machine type.

    Accepts a small labelled dataset (minimum 35 cycles) and runs
    two-phase transfer learning:
      - Phase 1: Head-only training (fast, prevents catastrophic forgetting)
      - Phase 2: Full fine-tuning at low LR (adapts LSTM temporal patterns)

    The resulting adapted pipeline is saved as
    models/saved/pm_pipeline_{machine_id}.joblib and can be used
    immediately via POST /predict/adapted.
    """
    try:
        result = run_adaptation(
            machine_id      = request.machine_id,
            base_dataset_id = request.base_dataset_id,
            sensor_names    = request.sensor_names,
            readings        = request.readings,
            rul_labels      = request.rul_labels,
            phase1_epochs   = request.phase1_epochs,
            phase2_epochs   = request.phase2_epochs,
            phase1_lr       = request.phase1_lr,
            phase2_lr       = request.phase2_lr,
        )
        return AdaptResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")


@app.post("/predict/adapted", tags=["Transfer Learning"])
def predict_adapted(request: AdaptPredictRequest):
    """
    Predict RUL using a fine-tuned machine-specific model.

    Use the machine_id returned by POST /adapt.
    Accepts any number of sensor columns — FeatureAligner handles
    the mapping to the LSTM's expected input shape automatically.
    """
    try:
        result = run_adapted_prediction(
            machine_id = request.machine_id,
            readings   = request.readings,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/machines", tags=["Transfer Learning"])
def list_adapted_machines(models_dir: str = "models/saved"):
    """List all fine-tuned machine-specific models available for prediction."""
    import pathlib
    p = pathlib.Path(models_dir)
    if not p.exists():
        return {"adapted_machines": []}
    machines = [
        f.stem.replace("pm_pipeline_", "")
        for f in p.glob("pm_pipeline_*.joblib")
        if f.stem.replace("pm_pipeline_", "").upper() not in ["FD001", "FD002", "FD003", "FD004"]
    ]
    return {"adapted_machines": machines}