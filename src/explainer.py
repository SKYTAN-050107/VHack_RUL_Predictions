import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


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
        SHAP explainer instance (prefers DeepExplainer; falls back to GradientExplainer)
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_background), size=n_background, replace=False)
    bg  = X_background[idx].astype(np.float32)
    try:
        return shap.DeepExplainer(model, bg)
    except Exception:
        return shap.GradientExplainer(model, bg)


def compute_shap_values(explainer,
                         X_explain: np.ndarray,
                         model=None,
                         background: np.ndarray | None = None) -> np.ndarray:
    """
    Compute SHAP values for a batch of windows.

    Args:
        explainer  : SHAP explainer instance
        X_explain  : Windows of shape (n_samples, window_size, n_features)
        model      : Optional Keras model for gradient-based fallback
        background : Optional background windows for baseline in fallback

    Returns:
        SHAP values of shape (n_samples, window_size, n_features)
        Positive values = pushes RUL higher (healthier signal)
        Negative values = pushes RUL lower (degradation signal)
    """
    X_float = X_explain.astype(np.float32)

    def _gradient_fallback_values(model_obj, x_values, bg_values=None):
        if model_obj is None:
            raise RuntimeError('No model provided for fallback attribution computation.')
        if bg_values is not None and len(bg_values) > 0:
            baseline = np.mean(bg_values.astype(np.float32), axis=0, keepdims=True)
        else:
            baseline = np.zeros((1, x_values.shape[1], x_values.shape[2]), dtype=np.float32)
        baseline_batch = np.repeat(baseline, len(x_values), axis=0)
        x_tensor = tf.convert_to_tensor(x_values)
        baseline_tensor = tf.convert_to_tensor(baseline_batch)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            preds = model_obj(x_tensor, training=False)
            preds = tf.reshape(preds, (-1,))
        grads = tape.gradient(preds, x_tensor)
        return (grads * (x_tensor - baseline_tensor)).numpy()

    try:
        raw = explainer.shap_values(X_float)
    except Exception as exc:
        message = str(exc)
        if ('shap_TensorListStack' in message) or ('shap_StridedSlice' in message):
            try:
                model_for_fallback = model
                if model_for_fallback is None:
                    model_for_fallback = getattr(explainer, 'model', None)
                raw = _gradient_fallback_values(model_for_fallback, X_float, background)
            except Exception:
                raise
        else:
            raise
    # DeepExplainer returns a list for multi-output; take first output
    if isinstance(raw, list):
        raw = raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 4 and raw.shape[-1] == 1:
        raw = raw[..., 0]
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
