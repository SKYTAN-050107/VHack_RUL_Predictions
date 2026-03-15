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
