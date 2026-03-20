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

# ──────────────────────────────────────────────────────────────────────────────
# APPEND THESE LINES TO THE BOTTOM OF: api/schemas.py
# Do not replace anything — just paste from here to the end of the file
# ──────────────────────────────────────────────────────────────────────────────


class AdaptRequest(BaseModel):
    """
    Request body for POST /adapt

    Upload a small labelled dataset from a new machine to fine-tune
    the pretrained LSTM on your specific equipment.

    Fields:
        machine_id      : Your identifier for the new machine (used as filename)
        base_dataset_id : Which pretrained model to start from (e.g. 'FD001')
        sensor_names    : List of sensor column names in your data
                          (order must match columns in readings)
        readings        : 2D list shape (n_cycles, n_sensors) — raw sensor history
        rul_labels      : 1D list of RUL values, one per cycle in readings
        phase1_epochs   : Head-only training epochs (default 30)
        phase2_epochs   : Full fine-tune epochs (default 20, set 0 to skip)
        phase1_lr       : Phase 1 learning rate (default 0.001)
        phase2_lr       : Phase 2 learning rate (default 0.0001)
    """
    machine_id:      str              = Field(...,    example="crusher_unit_A")
    base_dataset_id: str              = Field("FD001", example="FD001")
    sensor_names:    List[str]        = Field(...,    description="Column names for your sensors")
    readings:        List[List[float]]
    rul_labels:      List[float]
    phase1_epochs:   int              = Field(30,    ge=5,  le=200)
    phase2_epochs:   int              = Field(20,    ge=0,  le=100)
    phase1_lr:       float            = Field(1e-3,  gt=0)
    phase2_lr:       float            = Field(1e-4,  gt=0)


class AdaptResponse(BaseModel):
    """Response from POST /adapt"""
    machine_id:        str
    base_dataset_id:   str
    n_training_cycles: int
    n_sensors_input:   int
    alignment_method:  str
    phase1_final_loss: float
    phase2_final_loss: Optional[float]
    pipeline_path:     str
    message:           str


class AdaptPredictRequest(BaseModel):
    """
    Request body for POST /predict/adapted

    Use your fine-tuned machine-specific model to predict RUL.

    Fields:
        machine_id : Must match the machine_id used in /adapt
        readings   : Raw sensor readings, same columns as used during /adapt
    """
    machine_id: str               = Field(..., example="crusher_unit_A")
    readings:   List[List[float]] = Field(..., description="Raw sensor readings, same columns as used during /adapt")