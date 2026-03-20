from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Machine(BaseModel):
    machine_id: str
    name: str
    machine_type: str
    location: str
    status: str          # "Healthy" | "Warning" | "Critical"
    health_score: float  # 0–100
    rul_days: float
    last_updated: datetime
    installed_date: Optional[datetime] = None
    notes: Optional[str] = None
