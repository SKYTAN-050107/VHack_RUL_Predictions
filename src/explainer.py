import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


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
