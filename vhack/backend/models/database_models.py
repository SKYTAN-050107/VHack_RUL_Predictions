from pydantic import BaseModel, Field
from typing import Optional, List, Union, Any
from datetime import datetime

class Machine(BaseModel):
    id: Optional[int] = None
    name: str
    type: str
    current_rul: Optional[int] = None
    status: str = "Green"  # Red, Yellow, Green
    last_updated: Optional[datetime] = None

class MachineUpdate(BaseModel):
    current_rul: Optional[int] = None
    status: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)


class PredictionResult(BaseModel):
    machine_id: int
    predicted_rul: int
    status: str
    updated_at: datetime
    dataset_id: str
    health_state: str
    change_point_detected: bool
    change_point_step: Optional[int] = None
    explanation: str


class UploadSensorDataResponse(BaseModel):
    message: str
    prediction: PredictionResult


class SensorReading(BaseModel):
    machine_id: int
    source: str
    operating_mode: str
    vibration: Optional[float] = None
    temperature: Optional[float] = None
    load: Optional[float] = None
    anomaly_score: Optional[float] = 0.0
    recorded_at: datetime


class RULTrendPoint(BaseModel):
    machine_id: int
    source: str
    dataset_id: str
    rul_prediction: float
    health_state: str
    status: str
    change_point_detected: bool
    change_point_step: Optional[int] = None
    predicted_at: datetime


class SimulatorStatus(BaseModel):
    running: bool
    interval_seconds: int
    tick: int
    seed: int
    last_tick_at: Optional[datetime] = None


class SHAPFeatureImpact(BaseModel):
    feature: str
    shap_value: float
    direction: str
    rank: int


class SHAPExplanation(BaseModel):
    status: str
    cache_hit: bool = False
    mode: str
    base_value: float
    model_output: float
    dataset_id: str
    top_features: List[SHAPFeatureImpact] = []

class MaintenanceLog(BaseModel):
    id: Optional[int] = None
    machine_id: int
    technician_name: str
    status: str = "Active"  # Active, Completed
    root_cause_prediction: Optional[str] = None # AI's initial prediction
    root_cause_verified: Optional[str] = None # Human verified root cause
    action_taken: Optional[str] = None
    steps: Optional[List[str]] = None
    components: Optional[List[Any]] = None
    estimated_time: Optional[float] = None
    completion_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

class MaintenanceCreate(BaseModel):
    machine_id: int
    technician_name: str
    action_label: str
    root_cause_prediction: str
    steps: List[str]
    components: List[Union[dict, str]]
    estimated_time: float

class Staff(BaseModel):
    id: Optional[int] = None
    name: str
    role: str  # Senior Technician, Junior Technician, Maintenance Manager
    specialty: str  # Mechanical, Electrical, Software
    status: str = "Available"  # Available, Busy, On Leave
    created_at: datetime = Field(default_factory=datetime.now)

class Resource(BaseModel):
    id: Optional[int] = None
    filename: str
    resource_type: str  # technical, financial
    uploaded_at: datetime = Field(default_factory=datetime.now)

class UserAuth(BaseModel):
    email: str
    password: str
