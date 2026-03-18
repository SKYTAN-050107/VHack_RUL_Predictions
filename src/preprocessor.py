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
    df[sensor_cols] = df[sensor_cols].astype(float)
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
