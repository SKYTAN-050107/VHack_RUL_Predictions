import numpy as np
import pandas as pd


def create_windows(df: pd.DataFrame,
                    feature_cols: list,
                    window_size: int = 30,
                    rul_col: str = 'RUL') -> tuple:
    """
    Convert per-engine time series into overlapping sliding windows.

    Implements the time-window function h_t from Section 3.2 of the paper.

    For engine i with T_i cycles:
      - Produces (T_i - window_size) windows of shape (window_size, n_features)
      - If T_i <= window_size: zero-pad on the LEFT to reach window_size + 1

    Args:
        df           : Normalised DataFrame with 'unit_id', feature_cols, rul_col
        feature_cols : Input feature column names
        window_size  : T_w in the paper (default 30)
        rul_col      : Column name for regression target

    Returns:
        X         : np.ndarray of shape (N_total, window_size, n_features)
        y         : np.ndarray of shape (N_total,) — RUL at end of each window
        unit_ids  : list of unit_id for each sample (for traceability)
    """
    X_list, y_list, unit_list = [], [], []
    n_features = len(feature_cols)

    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        features = group[feature_cols].values   # (T, n_features)
        ruls     = group[rul_col].values         # (T,)
        T = len(features)

        # Zero-pad short sequences
        if T <= window_size:
            pad_len  = window_size - T + 1
            features = np.vstack([np.zeros((pad_len, n_features)), features])
            ruls     = np.concatenate([np.zeros(pad_len), ruls])

        for t in range(window_size, len(features)):
            window = features[t - window_size:t]   # (window_size, n_features)
            label  = ruls[t]
            X_list.append(window)
            y_list.append(label)
            unit_list.append(unit_id)

    return np.array(X_list), np.array(y_list), unit_list


def create_windows_inference(df: pd.DataFrame,
                              feature_cols: list,
                              window_size: int = 30) -> tuple:
    """
    Create one window per engine for test-time inference.
    Uses the LAST window_size cycles of each engine.

    Args:
        df           : Normalised test DataFrame with 'unit_id' and feature_cols
        feature_cols : Input feature column names
        window_size  : Must match the value used during training

    Returns:
        X         : np.ndarray of shape (n_engines, window_size, n_features)
        unit_ids  : list of unit_id in the same order as X
    """
    X_list, unit_list = [], []
    n_features = len(feature_cols)

    for unit_id, group in df.groupby('unit_id'):
        group    = group.sort_values('cycle')
        features = group[feature_cols].values
        T        = len(features)

        if T < window_size:
            pad_len  = window_size - T
            features = np.vstack([np.zeros((pad_len, n_features)), features])

        X_list.append(features[-window_size:])
        unit_list.append(unit_id)

    return np.array(X_list), unit_list


def create_windows_for_unit_lifecycle(group: pd.DataFrame,
                                       feature_cols: list,
                                       window_size: int = 30) -> np.ndarray:
    """
    Create ALL windows for a single engine unit (for lifecycle analysis / SHAP).

    Returns:
        X : np.ndarray of shape (n_windows, window_size, n_features)
    """
    features = group.sort_values('cycle')[feature_cols].values
    n_features = len(feature_cols)
    T = len(features)

    if T <= window_size:
        pad_len  = window_size - T + 1
        features = np.vstack([np.zeros((pad_len, n_features)), features])

    X_list = []
    for t in range(window_size, len(features)):
        X_list.append(features[t - window_size:t])

    return np.array(X_list)
