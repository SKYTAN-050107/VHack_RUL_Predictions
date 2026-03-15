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
