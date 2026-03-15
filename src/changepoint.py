import numpy as np
import pandas as pd
from typing import Optional


def cusum_detector(series: np.ndarray,
                    threshold: float = 5.0,
                    drift: float = 0.5) -> Optional[int]:
    """
    Cumulative Sum (CUSUM) sequential change-point detector.

    Detects when a sensor signal deviates cumulatively from its early-lifecycle
    (healthy) baseline, signalling the transition to an Impaired state.

    Algorithm:
      1. Estimate mean and std from the first 20 samples (healthy baseline).
      2. Normalise the series by this baseline.
      3. Accumulate positive and negative deviations minus the drift allowance.
      4. Trigger when either accumulator exceeds threshold h.

    Args:
        series    : 1D numpy array of (normalised) sensor values
        threshold : Detection threshold (h). Higher = fewer false positives.
                    Recommended range: 3.0 – 8.0
        drift     : Allowance (k). Typically 0.5 × expected shift magnitude.

    Returns:
        Index of first detection, or None if no change is detected.
    """
    # Estimate healthy baseline from early cycles
    n_baseline = min(20, len(series) // 4)
    mean = np.mean(series[:n_baseline])
    std  = np.std(series[:n_baseline]) + 1e-8
    normed = (series - mean) / std

    s_pos, s_neg = 0.0, 0.0
    for i, x in enumerate(normed):
        s_pos = max(0.0, s_pos + x - drift)
        s_neg = max(0.0, s_neg - x - drift)
        if s_pos > threshold or s_neg > threshold:
            return i
    return None


def detect_health_transitions(df: pd.DataFrame,
                               sensor_cols: list,
                               threshold: float = 5.0) -> pd.DataFrame:
    """
    For each engine unit in df, find the earliest cycle where ANY sensor
    crosses the CUSUM detection threshold.

    This cycle marks the transition from a Healthy state to an Impaired state.

    Args:
        df          : Normalised DataFrame with 'unit_id', 'cycle', sensor_cols
        sensor_cols : Sensor columns to monitor
        threshold   : CUSUM threshold (same as cusum_detector)

    Returns:
        DataFrame with columns:
            unit_id                 : engine identifier
            health_transition_cycle : cycle number when impairment first detected
            max_cycle               : total cycles in the engine's life
            rul_at_transition       : remaining useful life at the transition point
    """
    records = []
    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        max_cycle   = group['cycle'].max()
        earliest_cp = max_cycle  # default: no change detected before end of life

        for col in sensor_cols:
            cp = cusum_detector(group[col].values, threshold=threshold)
            if cp is not None:
                cp_cycle    = group['cycle'].iloc[cp]
                earliest_cp = min(earliest_cp, cp_cycle)

        records.append({
            'unit_id':                 unit_id,
            'health_transition_cycle': earliest_cp,
            'max_cycle':               max_cycle,
            'rul_at_transition':       max_cycle - earliest_cp
        })

    return pd.DataFrame(records)


def classify_health_state(rul_prediction: float,
                            change_point_detected: bool,
                            critical_rul: float = 20.0,
                            warning_rul: float = 50.0) -> str:
    """
    Map a RUL prediction and change-point flag to a human-readable health state.

    States:
        'Healthy'   — no change detected and RUL > warning_rul
        'Warning'   — change detected but RUL > critical_rul
        'Critical'  — change detected and RUL <= critical_rul

    Args:
        rul_prediction      : Predicted remaining useful life in cycles
        change_point_detected : Whether CUSUM detected a transition
        critical_rul        : RUL threshold below which state = 'Critical'
        warning_rul         : RUL threshold above critical for 'Warning'

    Returns:
        Health state string
    """
    if not change_point_detected and rul_prediction > warning_rul:
        return 'Healthy'
    elif rul_prediction <= critical_rul:
        return 'Critical'
    else:
        return 'Warning'
