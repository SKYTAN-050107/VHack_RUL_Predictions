# 01 — Source Modules: Data Loading, Preprocessing & Windowing

> **IDE Agent Instructions:** Create each file at the exact path shown under `### Create File:`. All files go inside the `src/` directory of the project root.

---

## 1.1 — Data Loader

### Create File: `src/data_loader.py`

```python
import pandas as pd
import numpy as np

# Standard column names for all C-MAPSS datasets
CMAPSS_COLUMNS = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# All feature columns used as model input
FEATURE_COLS = (
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]


def load_cmapss(dataset_id: str, split: str = 'train',
                data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Load a C-MAPSS dataset by ID (e.g. 'FD001') and split ('train' or 'test').

    Args:
        dataset_id : One of 'FD001', 'FD002', 'FD003', 'FD004'
        split      : 'train' or 'test'
        data_dir   : Path to the directory containing the raw .txt files

    Returns:
        pd.DataFrame with standardised column names and a 'dataset_id' column
    """
    path = f"{data_dir}/{split}_{dataset_id}.txt"
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=CMAPSS_COLUMNS,
        engine='python'
    )
    df['dataset_id'] = dataset_id
    return df


def load_rul_labels(dataset_id: str, data_dir: str = 'data/raw') -> np.ndarray:
    """
    Load ground-truth RUL values for the test split of a C-MAPSS dataset.

    Returns:
        1D numpy array of RUL values, one per test engine unit
    """
    path = f"{data_dir}/RUL_{dataset_id}.txt"
    return pd.read_csv(path, header=None).values.flatten()


def load_all_datasets(data_dir: str = 'data/raw') -> dict:
    """
    Convenience function: load all four C-MAPSS datasets at once.

    Returns:
        dict with keys 'FD001' ... 'FD004', each containing:
            'train' : DataFrame
            'test'  : DataFrame
            'rul'   : np.ndarray of test RUL labels
    """
    datasets = {}
    for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        datasets[ds_id] = {
            'train': load_cmapss(ds_id, 'train', data_dir),
            'test':  load_cmapss(ds_id, 'test',  data_dir),
            'rul':   load_rul_labels(ds_id, data_dir)
        }
    return datasets
```

---

## 1.2 — Preprocessor

### Create File: `src/preprocessor.py`

```python
import numpy as np
import pandas as pd
import joblib
import os
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


# ─── Noise Handling ────────────────────────────────────────────────────────────

def apply_savgol_filter(df: pd.DataFrame,
                         sensor_cols: list,
                         window_length: int = 11,
                         polyorder: int = 3) -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing per engine unit to reduce sensor noise
    while preserving the shape of the degradation trend.

    Operates per-unit to prevent boundary artifacts between different engines.

    Args:
        df            : DataFrame with 'unit_id' column and sensor columns
        sensor_cols   : List of sensor column names to smooth
        window_length : SG filter window (must be odd, >= polyorder + 2)
        polyorder     : Polynomial order for SG filter

    Returns:
        DataFrame with smoothed sensor values (copy of input)
    """
    df = df.copy()
    for unit_id, group in df.groupby('unit_id'):
        for col in sensor_cols:
            values = group[col].values
            if len(values) >= window_length:
                df.loc[group.index, col] = savgol_filter(
                    values, window_length, polyorder
                )
    return df


def snr_db(signal: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in decibels for a 1D signal.
    Noise is estimated as the variance of first-differences.
    """
    signal_power = np.mean(signal ** 2)
    noise_estimate = np.var(np.diff(signal))
    return 10 * np.log10(signal_power / (noise_estimate + 1e-10))


# ─── Missing Data ──────────────────────────────────────────────────────────────

def inject_missing_data(df: pd.DataFrame,
                         sensor_cols: list,
                         missing_rate: float = 0.03,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Synthetically introduce NaN values to simulate real-world sensor dropout.
    Used for testing and demonstrating robustness of imputation.

    Args:
        missing_rate : Fraction of sensor readings to null out (e.g. 0.03 = 3%)
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    for col in sensor_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def impute_missing(df: pd.DataFrame,
                    sensor_cols: list,
                    method: str = 'linear') -> pd.DataFrame:
    """
    Impute missing sensor values per engine unit.

    Strategy:
      - Interior gaps: linear interpolation (preserves trend shape)
      - Edge gaps (start/end of series): backward/forward fill

    Args:
        method : 'linear' for interpolation, 'ffill' for forward fill only
    """
    df = df.copy()
    for unit_id, group in df.groupby('unit_id'):
        idx = group.index
        if method == 'linear':
            df.loc[idx, sensor_cols] = (
                group[sensor_cols]
                .interpolate(method='linear', limit_direction='both')
            )
        else:
            df.loc[idx, sensor_cols] = (
                group[sensor_cols].ffill().bfill()
            )
    return df


def find_constant_sensors(df: pd.DataFrame,
                           sensor_cols: list,
                           threshold: float = 1e-4) -> list:
    """
    Identify sensor columns with near-zero variance (effectively constant).
    These carry no degradation information for that specific dataset
    but are retained to maintain cross-domain feature-space consistency.
    """
    variances = df[sensor_cols].var()
    return list(variances[variances < threshold].index)


# ─── Normalisation ─────────────────────────────────────────────────────────────

def fit_normaliser(df: pd.DataFrame,
                    feature_cols: list,
                    save_path: str = None) -> MinMaxScaler:
    """
    Fit a MinMaxScaler to scale each feature to [0, 1].

    Each dataset is normalised INDEPENDENTLY (not globally) so that
    cross-dataset distribution shift is preserved for DANN training.

    Args:
        save_path : If provided, serialise the fitted scaler to this path
    """
    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return scaler


def apply_normaliser(df: pd.DataFrame,
                      scaler: MinMaxScaler,
                      feature_cols: list) -> pd.DataFrame:
    """Apply a pre-fitted MinMaxScaler to a DataFrame."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ─── RUL Target Construction ───────────────────────────────────────────────────

def add_piecewise_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Construct piecewise linear RUL targets as described in the paper (Section 4.2).

    Rule:
      - RUL = max_rul  while  (max_cycle - current_cycle) >= max_rul  [healthy phase]
      - RUL = (max_cycle - current_cycle)  otherwise                   [degrading phase]

    The constant of max_rul = 125 follows Listou Ellefsen et al. and other
    published benchmarks to allow direct comparison.
    """
    max_cycle = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycle, on='unit_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
    return df.drop(columns='max_cycle')


def full_preprocess_pipeline(df_train: pd.DataFrame,
                              df_test: pd.DataFrame,
                              feature_cols: list,
                              sensor_cols: list,
                              smooth: bool = True,
                              max_rul: int = 125,
                              scaler_save_path: str = None) -> tuple:
    """
    End-to-end preprocessing: smooth → impute → normalise → add RUL targets.

    Returns:
        df_train_processed : Normalised train DataFrame with RUL column
        df_test_processed  : Normalised test DataFrame (no RUL column)
        scaler             : Fitted MinMaxScaler (for deployment use)
    """
    if smooth:
        df_train = apply_savgol_filter(df_train, sensor_cols)
        df_test  = apply_savgol_filter(df_test,  sensor_cols)

    df_train = impute_missing(df_train, sensor_cols)
    df_test  = impute_missing(df_test,  sensor_cols)

    scaler = fit_normaliser(df_train, feature_cols, save_path=scaler_save_path)
    df_train = apply_normaliser(df_train, scaler, feature_cols)
    df_test  = apply_normaliser(df_test,  scaler, feature_cols)

    df_train = add_piecewise_rul(df_train, max_rul=max_rul)

    return df_train, df_test, scaler
```

---

## 1.3 — Windowing

### Create File: `src/windowing.py`

```python
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
```
