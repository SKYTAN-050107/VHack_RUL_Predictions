# 02 — Source Modules: Change-Point Detection, Evaluation & Explainability

> **IDE Agent Instructions:** Create each file at the exact path shown under `### Create File:`. All files go inside the `src/` directory of the project root.

---

## 2.1 — Change-Point Detection

### Create File: `src/changepoint.py`

```python
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
```

---

## 2.2 — Evaluation Metrics

### Create File: `src/evaluate.py`

```python
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA Prognostics Scoring Function (Equation 16 in the paper).

    Asymmetric penalty where OVER-predictions are penalised more harshly
    than UNDER-predictions, reflecting the real-world cost of falsely
    assuring operators that an engine has more life than it does.

        s = sum( exp(-c/13) - 1 )  if c < 0  (early prediction)
        s = sum( exp( c/10) - 1 )  if c >= 0 (late prediction)

    where c = predicted_RUL - true_RUL

    Args:
        y_true : Ground-truth RUL values
        y_pred : Model-predicted RUL values

    Returns:
        Scalar score (lower is better; 0 = perfect)
    """
    c  = y_pred - y_true
    a1, a2 = 13.0, 10.0
    scores = np.where(
        c < 0,
        np.exp(-c / a1) - 1,
        np.exp( c / a2) - 1
    )
    return float(np.sum(scores))


def evaluate_model(model,
                    X_test: np.ndarray,
                    y_true: np.ndarray,
                    model_name: str = 'Model',
                    batch_size: int = 256) -> dict:
    """
    Run inference and compute all evaluation metrics for a given model.

    Args:
        model      : Any model with a .predict(X) method
        X_test     : Input windows of shape (n_engines, window_size, n_features)
        y_true     : Ground-truth RUL values, shape (n_engines,)
        model_name : Label for the results dict
        batch_size : Prediction batch size

    Returns:
        dict with keys: model, RMSE, NASA_Score, MAE, y_pred
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    return {
        'model':       model_name,
        'RMSE':        round(rmse(y_true, y_pred), 2),
        'MAE':         round(mae(y_true, y_pred), 2),
        'NASA_Score':  round(nasa_score(y_true, y_pred), 1),
        'y_pred':      y_pred
    }


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Combine a list of evaluate_model result dicts into a summary DataFrame.

    Args:
        results_list : List of dicts from evaluate_model()

    Returns:
        DataFrame sorted by RMSE ascending, without the y_pred column
    """
    df = pd.DataFrame(results_list)
    df = df.drop(columns='y_pred', errors='ignore')
    return df.sort_values('RMSE').reset_index(drop=True)
```

---

## 2.3 — SHAP Explainer Wrapper

### Create File: `src/explainer.py`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def build_shap_explainer(model,
                          X_background: np.ndarray,
                          n_background: int = 100,
                          random_state: int = 42):
    """
    Build a SHAP DeepExplainer for a Keras/TensorFlow model.

    DeepExplainer uses a sample of training data as the background
    distribution against which individual predictions are explained.

    Args:
        model        : Trained Keras model (single RUL output)
        X_background : Training data of shape (N, window_size, n_features)
        n_background : Number of background samples to use (100 is sufficient)

    Returns:
        shap.DeepExplainer instance
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_background), size=n_background, replace=False)
    bg  = X_background[idx].astype(np.float32)
    return shap.DeepExplainer(model, bg)


def compute_shap_values(explainer,
                         X_explain: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for a batch of windows.

    Args:
        explainer  : shap.DeepExplainer instance
        X_explain  : Windows of shape (n_samples, window_size, n_features)

    Returns:
        SHAP values of shape (n_samples, window_size, n_features)
        Positive values = pushes RUL higher (healthier signal)
        Negative values = pushes RUL lower (degradation signal)
    """
    raw = explainer.shap_values(X_explain.astype(np.float32))
    # DeepExplainer returns a list for multi-output; take first output
    if isinstance(raw, list):
        return raw[0]
    return raw


def aggregate_shap_by_feature(shap_values: np.ndarray,
                                feature_cols: list) -> pd.Series:
    """
    Collapse (n_samples, window_size, n_features) → per-feature importance.
    Aggregates by mean absolute SHAP across all samples and timesteps.

    Returns:
        pd.Series indexed by feature name, sorted descending
    """
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))   # (n_features,)
    return pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)


def build_explanation_text(feature_importance: pd.Series,
                             rul_prediction: float,
                             health_state: str,
                             top_k: int = 3) -> str:
    """
    Generate a human-readable explanation of the RUL prediction.

    Args:
        feature_importance : pd.Series of mean |SHAP| values per feature
        rul_prediction     : Predicted RUL in cycles
        health_state       : 'Healthy', 'Warning', or 'Critical'
        top_k              : Number of top drivers to mention

    Returns:
        Plain-language explanation string for factory operators
    """
    top_drivers = feature_importance.head(top_k)
    driver_names = ', '.join(top_drivers.index.tolist())

    if health_state == 'Critical':
        urgency = f"CRITICAL: Only {rul_prediction:.0f} cycles remaining."
    elif health_state == 'Warning':
        urgency = f"WARNING: Approximately {rul_prediction:.0f} cycles remaining."
    else:
        urgency = f"Healthy operation. Estimated {rul_prediction:.0f} cycles remaining."

    return (
        f"{urgency} "
        f"The primary sensors driving this prediction are: {driver_names}. "
        f"These sensors have shown the strongest deviation from their healthy baseline "
        f"over the recent observation window."
    )


def plot_feature_importance(feature_importance: pd.Series,
                              title: str = 'Feature Importance (Mean |SHAP|)',
                              ax=None) -> plt.Axes:
    """
    Horizontal bar chart of SHAP-based feature importances.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance) // 2)))

    sorted_imp = feature_importance.sort_values(ascending=True)
    sorted_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.axvline(feature_importance.mean(), color='red', linestyle='--',
               linewidth=1.2, label='Mean importance')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.legend()
    return ax


def plot_shap_heatmap(shap_matrix: np.ndarray,
                       feature_cols: list,
                       title: str = 'SHAP Importance Heatmap') -> plt.Figure:
    """
    Heatmap of |SHAP| values over (window_size × n_features) for one sample.

    Args:
        shap_matrix : np.ndarray of shape (window_size, n_features)
    """
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(
        np.abs(shap_matrix).T,
        ax=ax,
        cmap='YlOrRd',
        xticklabels=list(range(shap_matrix.shape[0])),
        yticklabels=feature_cols
    )
    ax.set_xlabel('Timestep in Window (0 = oldest, T-1 = most recent)')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    return fig
```
