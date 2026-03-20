from pydantic import BaseModel
from typing import List, Optional


class FrequencyPeak(BaseModel):
    frequency_hz: float
    amplitude: float
    label: str


class MLPrediction(BaseModel):
    machine_id: str
    rul_days: float
    health_score: float
    vibration_rms: float
    temperature_celsius: float
    anomaly_score: float       # 0 = normal, 1 = severe
    anomalies: List[str]
    frequency_peaks: List[FrequencyPeak]
    confidence: float          # 0–1
    predicted_failure_mode: str
    timestamp: str
