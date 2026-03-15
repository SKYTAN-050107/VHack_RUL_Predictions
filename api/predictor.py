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
