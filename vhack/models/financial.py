from pydantic import BaseModel
from typing import Optional, Dict


class FinancialParameters(BaseModel):
    machine_id: str
    hourly_production_value: float
    units_per_hour: int
    unit_price: float
    repair_cost_preventative: float
    repair_cost_failure: float
    mttr_hours: float
    sla_penalty_per_hour: float
    supply_chain_penalty: float
    source: Optional[str] = None   # document name this was extracted from
    confidence: Optional[float] = None


class FinancialCalculation(BaseModel):
    machine_id: str
    production_loss: float
    total_downtime_cost_failure: float
    total_preventative_cost: float
    net_savings: float
    roi_percentage: float
    cost_breakdown: Dict[str, float]
