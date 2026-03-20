"""
Mock API client — mirrors the interface of the real MLBackendClient exactly.
Swap to the real client by setting USE_MOCK_DATA = False in config/settings.py.
"""

import streamlit as st
from datetime import datetime
from typing import List, Optional

from models.machine import Machine
from models.prediction import MLPrediction, FrequencyPeak
from data.mock_machines import MOCK_MACHINES_DATA
from data.mock_predictions import MOCK_PREDICTIONS
from data.mock_sensor_history import get_sensor_history, get_fft_spectrum
from config.settings import CACHE_MACHINE_LIST, CACHE_ML_PREDICTIONS, CACHE_SENSOR_HISTORY


@st.cache_data(ttl=CACHE_MACHINE_LIST)
def _cached_all_machines():
    machines = []
    for m in MOCK_MACHINES_DATA:
        machines.append(Machine(**m))
    return machines


@st.cache_data(ttl=CACHE_ML_PREDICTIONS)
def _cached_prediction(machine_id: str):
    raw = MOCK_PREDICTIONS.get(machine_id)
    if raw is None:
        return None
    data = dict(raw)
    data["frequency_peaks"] = [FrequencyPeak(**fp) for fp in data["frequency_peaks"]]
    return MLPrediction(**data)


@st.cache_data(ttl=CACHE_SENSOR_HISTORY)
def _cached_sensor_history(machine_id: str, status: str, sensor_type: str, days: int):
    return get_sensor_history(machine_id, status, sensor_type, days)


@st.cache_data(ttl=CACHE_SENSOR_HISTORY)
def _cached_fft(machine_id: str, status: str):
    return get_fft_spectrum(machine_id, status)


class MockMLBackendClient:
    """Drop-in replacement for MLBackendClient using in-memory mock data."""

    def fetch_all_machines(self) -> List[Machine]:
        return _cached_all_machines()

    def fetch_machine_info(self, machine_id: str) -> Optional[Machine]:
        machines = _cached_all_machines()
        return next((m for m in machines if m.machine_id == machine_id), None)

    def fetch_ml_prediction(self, machine_id: str) -> Optional[MLPrediction]:
        return _cached_prediction(machine_id)

    def fetch_machine_history(self, machine_id: str, sensor_type: str, days: int = 30):
        machine = self.fetch_machine_info(machine_id)
        status = machine.status if machine else "Healthy"
        return _cached_sensor_history(machine_id, status, sensor_type, days)

    def fetch_fft_spectrum(self, machine_id: str):
        machine = self.fetch_machine_info(machine_id)
        status = machine.status if machine else "Healthy"
        return _cached_fft(machine_id, status)
