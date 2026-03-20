"""
Deterministic, realistic 30-day sensor history generator.
Each machine gets hourly readings for vibration (mm/s), temperature (°C), and load (%).
Critical machines show upward trends and anomaly spikes; Healthy machines are stable.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Machine status → sensor generation parameters
_SENSOR_PARAMS = {
    "Healthy": {
        "vibration": {"base": 0.75, "trend": 0.0005, "noise": 0.07, "n_spikes": 0},
        "temperature": {"base": 48.0, "trend": 0.01,   "noise": 1.2,  "n_spikes": 0},
        "load":        {"base": 72.0, "trend": 0.0,    "noise": 3.0,  "n_spikes": 0},
    },
    "Warning": {
        "vibration": {"base": 1.65, "trend": 0.008, "noise": 0.18, "n_spikes": 2},
        "temperature": {"base": 64.0, "trend": 0.06,  "noise": 2.0,  "n_spikes": 1},
        "load":        {"base": 78.0, "trend": 0.02,  "noise": 4.5,  "n_spikes": 0},
    },
    "Critical": {
        "vibration": {"base": 2.90, "trend": 0.025, "noise": 0.40, "n_spikes": 5},
        "temperature": {"base": 77.5, "trend": 0.15,  "noise": 3.5,  "n_spikes": 3},
        "load":        {"base": 84.0, "trend": 0.05,  "noise": 6.0,  "n_spikes": 1},
    },
}


def _seed(machine_id: str) -> int:
    return abs(hash(machine_id)) % (2**31)


def _generate_series(machine_id: str, status: str, sensor: str, n_points: int) -> np.ndarray:
    np.random.seed(_seed(machine_id) + ord(sensor[0]))
    p = _SENSOR_PARAMS.get(status, _SENSOR_PARAMS["Healthy"])[sensor]

    t = np.arange(n_points)
    values = p["base"] + p["trend"] * t + p["noise"] * np.random.randn(n_points)

    # Daily operational cycle (subtle sinusoidal pattern)
    day_cycle = p["noise"] * 0.5 * np.sin(2 * np.pi * t / 24)
    values += day_cycle

    # Inject anomaly spikes in the latter half of the window
    if p["n_spikes"] > 0:
        mid = n_points // 2
        spike_indices = np.random.choice(range(mid, n_points), size=p["n_spikes"], replace=False)
        values[spike_indices] *= np.random.uniform(1.8, 3.2, size=p["n_spikes"])

    # Physical lower bounds
    lower_bounds = {"vibration": 0.1, "temperature": 35.0, "load": 20.0}
    upper_bounds = {"vibration": 15.0, "temperature": 120.0, "load": 100.0}
    values = np.clip(values, lower_bounds[sensor], upper_bounds[sensor])
    return np.round(values, 3)


def get_sensor_history(machine_id: str, status: str, sensor_type: str, days: int = 30) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [timestamp, <sensor_type>] at hourly resolution.

    sensor_type: "vibration" | "temperature" | "load"
    """
    n_points = days * 24
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=(n_points - i)) for i in range(n_points)]

    col_names = {
        "vibration":  "vibration_rms",
        "temperature": "temperature_celsius",
        "load":        "load_pct",
    }
    col = col_names.get(sensor_type, sensor_type)
    values = _generate_series(machine_id, status, sensor_type, n_points)

    return pd.DataFrame({"timestamp": timestamps, col: values, "machine_id": machine_id})


def get_fft_spectrum(machine_id: str, status: str) -> pd.DataFrame:
    """Returns synthetic FFT spectrum data (frequency_hz, amplitude)."""
    np.random.seed(_seed(machine_id) + 99)

    params = {
        "Healthy": {"shaft_freq": 29.5, "gmf": 0, "noise_floor": 0.05},
        "Warning": {"shaft_freq": 29.5, "gmf": 187.0, "noise_floor": 0.12},
        "Critical": {"shaft_freq": 29.5, "gmf": 187.0, "noise_floor": 0.25},
    }
    p = params.get(status, params["Healthy"])

    freqs = np.linspace(0, 500, 1000)
    amplitude = np.random.uniform(0, p["noise_floor"], len(freqs))

    # Add shaft harmonics (1x, 2x, 3x)
    for harmonic in [1, 2, 3]:
        idx = np.argmin(np.abs(freqs - p["shaft_freq"] * harmonic))
        amplitude[idx] += 0.15 / harmonic

    # Add gear mesh frequency harmonics for degraded machines
    if p["gmf"] > 0:
        severity = {"Warning": 0.35, "Critical": 1.10}.get(status, 0)
        for gmf_harmonic in [1, 2, 3]:
            target = p["gmf"] * gmf_harmonic
            if target < 500:
                idx = np.argmin(np.abs(freqs - target))
                amplitude[idx] += severity / gmf_harmonic
                # Sidebands
                for sb in [-1, 1]:
                    sb_freq = target + sb * p["shaft_freq"]
                    if 0 < sb_freq < 500:
                        sb_idx = np.argmin(np.abs(freqs - sb_freq))
                        amplitude[sb_idx] += (severity / gmf_harmonic) * 0.4

    return pd.DataFrame({"frequency_hz": np.round(freqs, 2), "amplitude": np.round(amplitude, 4)})
