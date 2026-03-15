# 08 — Notebooks 7 & 8: Interpretability and Model Export

> **IDE Agent Instructions:**
> - Create TWO Jupyter notebook files at the paths shown.
> - `# %% [markdown]` → Markdown cell. `# %%` → Code cell.

---

## Notebook 7 — Model Interpretability (SHAP)

### Create File: `notebooks/07_interpretability.ipynb`

```python
# %% [markdown]
# # Notebook 7 — Model Interpretability
#
# **Goal:** Provide actionable, human-readable explanations of RUL predictions
# using SHAP (SHapley Additive exPlanations).
#
# SHAP decomposes each prediction into per-feature contributions, answering:
# *"Why is the predicted RUL decreasing for this engine right now?"*
#
# Three levels of analysis:
# 1. **Fleet-level**: Which sensors matter most overall?
# 2. **Single-prediction**: What drove this specific RUL estimate?
# 3. **Lifecycle**: How does feature importance evolve as an engine ages?

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

from src.data_loader    import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor   import full_preprocess_pipeline
from src.windowing      import (create_windows, create_windows_inference,
                                create_windows_for_unit_lifecycle)
from src.models.lstm_baseline import build_lstm_baseline
from src.explainer      import (build_shap_explainer, compute_shap_values,
                                 aggregate_shap_by_feature, build_explanation_text,
                                 plot_feature_importance, plot_shap_heatmap)

WINDOW_SIZE = 30

datasets = load_all_datasets(data_dir='../data/raw')
for ds_id in ['FD001']:
    df_tr, df_te, scaler = full_preprocess_pipeline(
        df_train=datasets[ds_id]['train'],
        df_test=datasets[ds_id]['test'],
        feature_cols=FEATURE_COLS, sensor_cols=SENSOR_COLS,
        smooth=True, max_rul=125
    )
    datasets[ds_id]['train_norm'] = df_tr
    datasets[ds_id]['test_norm']  = df_te
    X, y, _ = create_windows(df_tr, FEATURE_COLS, WINDOW_SIZE)
    datasets[ds_id]['X_train'] = X
    datasets[ds_id]['y_train'] = y

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    datasets['FD001']['X_train'].astype(np.float32),
    datasets['FD001']['y_train'].astype(np.float32),
    test_size=0.1, random_state=42
)

# Load pre-trained baseline model
model = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
weights_path = '../models/saved/lstm_target_only_FD001.keras'
try:
    model.load_weights(weights_path)
    print("Pre-trained weights loaded.")
except:
    print("Training model from scratch (NB04 not run yet)...")
    from tensorflow.keras.callbacks import EarlyStopping
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=256,
              callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
              verbose=0)
    model.save_weights(weights_path)

print("Model ready.")

# %% [markdown]
# ## 7.1 Build SHAP Explainer

# %%
# Build DeepExplainer using 100 background training samples
explainer = build_shap_explainer(model, X_tr, n_background=100)
print("SHAP DeepExplainer built.")

# Explain 100 test samples
X_test, test_unit_ids = create_windows_inference(
    datasets['FD001']['test_norm'], FEATURE_COLS, WINDOW_SIZE
)
n_explain  = min(100, len(X_test))
X_explain  = X_test[:n_explain].astype(np.float32)
y_test_rul = datasets['FD001']['rul'][:n_explain]

shap_vals = compute_shap_values(explainer, X_explain)
print(f"SHAP values shape: {shap_vals.shape}  (n_samples, window_size, n_features)")

# %% [markdown]
# ## 7.2 Fleet-Level Feature Importance

# %%
feature_importance = aggregate_shap_by_feature(shap_vals, FEATURE_COLS)

fig, ax = plt.subplots(figsize=(10, 10))
plot_feature_importance(
    feature_importance,
    title='Mean |SHAP| — Feature Importance for RUL Prediction (FD001, 100 test engines)',
    ax=ax
)
plt.tight_layout()
plt.show()

print("\nTop 5 most important features:")
print(feature_importance.head(5).to_string())
print("\nTop 5 least important features:")
print(feature_importance.tail(5).to_string())

# %% [markdown]
# **Insight for factory operators:** The sensors appearing at the top of this
# chart are the primary "health indicators" for this machine type. In FD001
# (single operating condition, HPC fault mode), sensors 11, 14, and 4 typically
# dominate — corresponding to physical signals most correlated with High Pressure
# Compressor degradation. Sensors at the bottom (near-constant in this dataset)
# contribute minimally and can be deprioritised in monitoring dashboards.

# %%
# %% [markdown]
# ## 7.3 Single-Engine Prediction Explanation (Waterfall Chart)

# %%
# Choose a test sample — engine near end of life
sample_idx  = np.argmin(y_test_rul[:n_explain])  # engine with lowest true RUL
pred_rul    = float(model.predict(X_explain[sample_idx:sample_idx+1], verbose=0).flatten()[0])
true_rul    = y_test_rul[sample_idx]
health_state = 'Critical' if pred_rul < 20 else ('Warning' if pred_rul < 50 else 'Healthy')

print(f"Engine {test_unit_ids[sample_idx]}: True RUL = {true_rul:.0f}, "
      f"Predicted RUL = {pred_rul:.1f}, Health = {health_state}")

# Per-feature contribution for this sample (averaged over timesteps)
single_shap = shap_vals[sample_idx]                    # (30, 24)
single_imp  = np.abs(single_shap).mean(axis=0)          # (24,)
top_k = 12

top_features = pd.Series(single_imp, index=FEATURE_COLS).nlargest(top_k)
sorted_top   = top_features.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d73027' if v > top_features.mean() else '#1a9850'
          for v in sorted_top.values]
sorted_top.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.85)
ax.set_title(
    f'Top {top_k} Feature Contributions to RUL Prediction\n'
    f'Engine {test_unit_ids[sample_idx]} | Predicted RUL: {pred_rul:.1f} cycles '
    f'| State: {health_state}',
    fontsize=11
)
ax.set_xlabel('Mean |SHAP Value|')
ax.axvline(top_features.mean(), color='gray', linestyle='--',
           linewidth=1, label='Mean importance')
ax.legend()
plt.tight_layout()
plt.show()

# Generated explanation text
explanation = build_explanation_text(
    feature_importance=pd.Series(single_imp, index=FEATURE_COLS).sort_values(ascending=False),
    rul_prediction=pred_rul,
    health_state=health_state,
    top_k=3
)
print("\nGenerated operator explanation:")
print(f"  → {explanation}")

# %% [markdown]
# **For factory operators:** The generated explanation text above is the format
# used by the FastAPI endpoint's `explanation` field. It translates the model's
# internal SHAP analysis into a plain-language message that non-technical
# operators can act on without understanding machine learning.

# %%
# %% [markdown]
# ## 7.4 Temporal SHAP Heatmap: Which Timesteps Matter Most?

# %%
fig = plot_shap_heatmap(
    shap_vals[sample_idx],
    FEATURE_COLS,
    title=f'SHAP Importance Over Time — Engine {test_unit_ids[sample_idx]}\n'
          f'(Rows=Features, Columns=Timesteps in window, Colour=|SHAP|)'
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The heatmap reveals WHEN within the 30-cycle window each sensor
# is most influential. If the rightmost columns (most recent cycles) dominate,
# the model is responding to sudden recent changes — characteristic of
# accelerated degradation. If earlier columns dominate, the model relies on
# the longer trend, appropriate for gradual wear accumulation.

# %%
# %% [markdown]
# ## 7.5 Lifecycle Feature Importance: How Explanations Evolve Over Engine Life

# %%
# Pick an engine unit and track SHAP importance across its full lifecycle
df_fd001 = datasets['FD001']['train_norm']
LIFECYCLE_UNIT = 10

unit_data = df_fd001[df_fd001['unit_id'] == LIFECYCLE_UNIT].sort_values('cycle')
X_lc      = create_windows_for_unit_lifecycle(unit_data, FEATURE_COLS, WINDOW_SIZE)
shap_lc   = compute_shap_values(explainer, X_lc.astype(np.float32))

# Mean |SHAP| per feature per window → (n_windows, n_features)
feature_trace = np.abs(shap_lc).mean(axis=1)

# RUL over lifecycle
rul_lc = unit_data['RUL'].values[WINDOW_SIZE:]
cycles_lc = unit_data['cycle'].values[WINDOW_SIZE:]

# Identify top 5 features by total lifecycle importance
top5_idx = feature_trace.mean(axis=0).argsort()[-5:][::-1]

fig, axes = plt.subplots(3, 1, figsize=(16, 13))

# Panel 1: RUL timeline
axes[0].plot(cycles_lc, rul_lc, color='navy', linewidth=2.5)
axes[0].fill_between(cycles_lc, 0, rul_lc, alpha=0.1, color='navy')
axes[0].set_ylabel('RUL (cycles)')
axes[0].set_title(f'Engine Unit {LIFECYCLE_UNIT} — RUL Lifecycle')
axes[0].grid(alpha=0.3)

# Panel 2: Raw sensor values for top sensors
for idx in top5_idx:
    axes[1].plot(cycles_lc, X_lc[len(X_lc)-len(cycles_lc):, -1, idx],
                 label=FEATURE_COLS[idx], alpha=0.8)
axes[1].set_ylabel('Normalised Sensor Value')
axes[1].set_title('Top-5 Sensors by SHAP Importance')
axes[1].legend(ncol=2, fontsize=8)
axes[1].grid(alpha=0.3)

# Panel 3: SHAP importance traces
for idx in top5_idx:
    axes[2].plot(cycles_lc, feature_trace[:, idx],
                 label=FEATURE_COLS[idx], alpha=0.85, linewidth=1.8)
axes[2].set_xlabel('Cycle')
axes[2].set_ylabel('Mean |SHAP Value|')
axes[2].set_title('How Feature Importance Shifts Over Engine Lifetime')
axes[2].legend(ncol=2, fontsize=8)
axes[2].grid(alpha=0.3)

plt.suptitle(
    f'Engine {LIFECYCLE_UNIT} — Full Lifecycle Interpretability Analysis (FD001)',
    fontsize=13
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight for operators:** The bottom panel reveals a key pattern —
# feature importance is relatively uniform early in the lifecycle (model
# is monitoring many signals equally) but concentrates on 1-2 dominant sensors
# as failure approaches. This concentration is the signal that the engine has
# entered an accelerated degradation phase. A sudden spike in SHAP importance
# for a specific sensor is an early warning independent of the RUL value itself.
```

---

## Notebook 8 — Model Export & FastAPI Integration

### Create File: `notebooks/08_model_export_fastapi.ipynb`

```python
# %% [markdown]
# # Notebook 8 — Model Export & FastAPI Integration
#
# **Goal:** Package the full inference pipeline into a single `.joblib` file
# and verify it works end-to-end before deploying via FastAPI.
#
# **Pipeline contents:**
# 1. Per-dataset MinMaxScaler
# 2. Savitzky-Golay noise smoother
# 3. Linear imputer for missing values
# 4. Time-window constructor
# 5. Trained LSTM model (weights loaded at prediction time)
# 6. CUSUM health-state classifier

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import joblib
import os
import json
import requests
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin

from src.data_loader    import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor   import (apply_savgol_filter, impute_missing,
                                 full_preprocess_pipeline)
from src.windowing      import create_windows
from src.models.lstm_baseline import build_lstm_baseline
from src.changepoint    import cusum_detector, classify_health_state
from src.explainer      import (build_shap_explainer, compute_shap_values,
                                 aggregate_shap_by_feature, build_explanation_text)

WINDOW_SIZE = 30
MAX_RUL     = 125
os.makedirs('../models/saved', exist_ok=True)

# %% [markdown]
# ## 8.1 Define the Production Pipeline Wrapper

# %%
class PredictiveMaintenancePipeline(BaseEstimator, RegressorMixin):
    """
    Joblib-serialisable wrapper combining the full inference chain:
        raw sensor array
            → Savitzky-Golay smoothing
            → linear imputation of missing values
            → MinMaxScaler normalisation
            → time-window construction
            → LSTM RUL prediction
            → CUSUM health state classification

    Usage:
        pipeline = PredictiveMaintenancePipeline(...)
        result   = pipeline.predict(X_raw)

        X_raw : np.ndarray of shape (n_cycles, 24)
                columns = op_setting_1-3, sensor_1-21 (in order)

    The LSTM model weights are lazy-loaded on the first predict() call and
    cached in self._model (not serialised by joblib — weights path is stored).
    """

    def __init__(self,
                  scaler,
                  model_weights_path: str,
                  feature_cols: list,
                  window_size: int = 30,
                  max_rul: int = 125,
                  cusum_threshold: float = 5.0,
                  sg_window: int = 11,
                  sg_poly: int = 3):
        self.scaler             = scaler
        self.model_weights_path = model_weights_path
        self.feature_cols       = feature_cols
        self.window_size        = window_size
        self.max_rul            = max_rul
        self.cusum_threshold    = cusum_threshold
        self.sg_window          = sg_window
        self.sg_poly            = sg_poly
        self._model             = None   # not serialised

    def _load_model(self):
        """Lazy-load LSTM weights on first call."""
        if self._model is None:
            self._model = build_lstm_baseline(
                window_size=self.window_size,
                n_features=len(self.feature_cols)
            )
            self._model.load_weights(self.model_weights_path)

    def predict(self, X_raw: np.ndarray) -> dict:
        """
        Full inference pipeline.

        Args:
            X_raw : np.ndarray of shape (n_cycles, 24) — raw sensor readings
                    ordered by cycle (oldest → most recent)

        Returns:
            dict with:
                rul_prediction        : float (cycles remaining)
                health_state          : str ('Healthy' | 'Warning' | 'Critical')
                change_point_detected : bool
                change_point_step     : int or None
        """
        from scipy.signal import savgol_filter

        self._load_model()
        X = X_raw.astype(np.float64).copy()

        # 1. Smooth noise (per column, like per-unit smoothing in training)
        if len(X) >= self.sg_window:
            for j in range(X.shape[1]):
                X[:, j] = savgol_filter(X[:, j], self.sg_window, self.sg_poly)

        # 2. Impute missing values (linear fill within array)
        X = pd.DataFrame(X, columns=self.feature_cols)
        X = X.interpolate(method='linear', limit_direction='both').values

        # 3. Normalise
        X_norm = self.scaler.transform(X)

        # 4. Build last window
        T = len(X_norm)
        if T < self.window_size:
            pad    = np.zeros((self.window_size - T, X_norm.shape[1]))
            X_norm = np.vstack([pad, X_norm])
        window = X_norm[-self.window_size:][np.newaxis, :, :]  # (1, Tw, F)

        # 5. Predict RUL
        rul = float(self._model.predict(window, verbose=0).flatten()[0])
        rul = float(np.clip(rul, 0, self.max_rul))

        # 6. Change-point detection on last 50 cycles
        n_recent = min(50, len(X_norm))
        recent   = X_norm[-n_recent:]
        # Monitor the most informative sensor (sensor_11 at index 13)
        sensor_11_idx = self.feature_cols.index('sensor_11')
        cp = cusum_detector(recent[:, sensor_11_idx],
                             threshold=self.cusum_threshold)

        health = classify_health_state(
            rul_prediction=rul,
            change_point_detected=(cp is not None),
            critical_rul=20.0, warning_rul=50.0
        )

        return {
            'rul_prediction':        round(rul, 1),
            'health_state':          health,
            'change_point_detected': cp is not None,
            'change_point_step':     int(cp) if cp is not None else None
        }

    def __getstate__(self):
        """Exclude the loaded model from joblib serialisation."""
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        """Restore state; model will be lazy-loaded on next predict()."""
        self.__dict__.update(state)
        self._model = None

# %% [markdown]
# ## 8.2 Build and Save Pipelines for All Datasets

# %%
datasets = load_all_datasets(data_dir='../data/raw')

for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    print(f"\nBuilding pipeline for {ds_id}...")

    df_tr, df_te, scaler = full_preprocess_pipeline(
        df_train=datasets[ds_id]['train'],
        df_test=datasets[ds_id]['test'],
        feature_cols=FEATURE_COLS, sensor_cols=SENSOR_COLS,
        smooth=True, max_rul=MAX_RUL,
        scaler_save_path=f'../models/saved/scaler_{ds_id}.joblib'
    )
    datasets[ds_id]['train_norm'] = df_tr

    # Check if trained weights exist (run NB04 first if not)
    weights_path = f'../models/saved/lstm_target_only_{ds_id}.keras'
    if not os.path.exists(weights_path):
        print(f"  Weights not found — training baseline model for {ds_id}...")
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.model_selection import train_test_split

        X, y, _ = create_windows(df_tr, FEATURE_COLS, WINDOW_SIZE)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X.astype(np.float32), y.astype(np.float32),
            test_size=0.1, random_state=42
        )
        m = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
        m.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=100, batch_size=256,
               callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
               verbose=0)
        m.save_weights(weights_path)
        print(f"  Weights saved: {weights_path}")

    pipeline = PredictiveMaintenancePipeline(
        scaler=scaler,
        model_weights_path=weights_path,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        max_rul=MAX_RUL,
        cusum_threshold=5.0
    )

    pipeline_path = f'../models/saved/pm_pipeline_{ds_id.lower()}.joblib'
    joblib.dump(pipeline, pipeline_path)
    print(f"  Pipeline saved: {pipeline_path}")

print("\nAll pipelines exported.")

# %% [markdown]
# ## 8.3 Verify Pipeline End-to-End

# %%
# Load the FD001 pipeline and run a test prediction
pipeline = joblib.load('../models/saved/pm_pipeline_fd001.joblib')

# Use a real engine from the test set
df_test = datasets['FD001']['test']
unit_1_raw = df_test[df_test['unit_id'] == 1][FEATURE_COLS].values

result = pipeline.predict(unit_1_raw)
print("Pipeline test prediction:")
for k, v in result.items():
    print(f"  {k}: {v}")

# %%
# Verify prediction changes with engine age
df_train_fd001 = datasets['FD001']['train']
unit_max_cycle = df_train_fd001[df_train_fd001['unit_id'] == 1]['cycle'].max()

rul_trajectory = []
for pct in [0.2, 0.4, 0.6, 0.8, 1.0]:
    n_cycles = int(unit_max_cycle * pct)
    raw_data = df_train_fd001[
        (df_train_fd001['unit_id'] == 1) &
        (df_train_fd001['cycle'] <= n_cycles)
    ][FEATURE_COLS].values
    r = pipeline.predict(raw_data)
    rul_trajectory.append({
        'Pct Lifetime': f'{pct*100:.0f}%',
        'Cycles Used':  n_cycles,
        'Predicted RUL': r['rul_prediction'],
        'Health State':  r['health_state'],
        'Change Point':  r['change_point_detected']
    })

traj_df = pd.DataFrame(rul_trajectory)
print("\nPipeline predictions at different lifecycle stages:")
print(traj_df.to_string(index=False))

# %%
# Visualise predicted RUL over lifecycle stages
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(traj_df['Cycles Used'], traj_df['Predicted RUL'],
        marker='o', linewidth=2.5, color='steelblue', markersize=8)
ax.axhline(50, color='orange', linestyle='--', linewidth=1.5, label='Warning threshold (50)')
ax.axhline(20, color='red',    linestyle='--', linewidth=1.5, label='Critical threshold (20)')

for _, row in traj_df.iterrows():
    color = ('red' if row['Health State'] == 'Critical'
             else ('orange' if row['Health State'] == 'Warning' else 'green'))
    ax.annotate(row['Health State'],
                xy=(row['Cycles Used'], row['Predicted RUL']),
                xytext=(0, 12), textcoords='offset points',
                ha='center', color=color, fontsize=9)

ax.set_xlabel('Cycles of Data Available')
ax.set_ylabel('Predicted RUL')
ax.set_title('Pipeline Predictions at Different Lifecycle Stages (Engine 1, FD001)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The predicted RUL decreases monotonically as more cycles become
# available and the engine approaches failure. The health state transitions from
# Healthy → Warning → Critical, providing a natural three-stage alert system
# for factory operators.

# %%
# %% [markdown]
# ## 8.4 FastAPI Startup Check

# %%
# Verify API can start by checking import paths
print("FastAPI module check:")
try:
    from api.schemas   import PredictRequest, PredictResponse
    from api.predictor import run_prediction, list_available_models
    from api.main      import app
    print("  All API modules imported successfully.")
    models_available = list_available_models(models_dir='../models/saved')
    print(f"  Available models: {models_available}")
except ImportError as e:
    print(f"  Import error: {e}")
    print("  Ensure api/__init__.py exists and all api/ files are created.")

# %% [markdown]
# ## 8.5 API Startup & Test Instructions

# %%
startup_instructions = """
To launch the FastAPI server, run from the project root directory:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then open your browser to:
    http://localhost:8000/docs   ← Swagger UI (interactive API explorer)
    http://localhost:8000/redoc  ← ReDoc documentation

Available endpoints:
    GET  /health   → liveness check
    GET  /models   → list available dataset models
    POST /predict  → RUL prediction + health state + explanation

Example Python request (run after starting the server):

    import requests, json
    payload = {
        "unit_id":    "engine_001",
        "dataset_id": "FD001",
        "readings":   raw_sensor_data.tolist()   # shape: (n_cycles, 24)
    }
    r = requests.post("http://localhost:8000/predict", json=payload)
    print(json.dumps(r.json(), indent=2))
"""
print(startup_instructions)

# %%
# Live API test (run this cell AFTER starting the server in a separate terminal)
try:
    # Test /health
    r = requests.get("http://localhost:8000/health", timeout=3)
    print("Health check:", r.json())

    # Test /models
    r = requests.get("http://localhost:8000/models", timeout=3)
    print("Available models:", r.json())

    # Test /predict
    unit_1_readings = df_test[df_test['unit_id'] == 1][FEATURE_COLS].values
    payload = {
        "unit_id":    "engine_001",
        "dataset_id": "FD001",
        "readings":   unit_1_readings.tolist()
    }
    r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
    print("\nPrediction response:")
    print(json.dumps(r.json(), indent=2))

except requests.exceptions.ConnectionError:
    print("Server not running. Start it with: uvicorn api.main:app --port 8000")

# %% [markdown]
# ## 8.6 Final Export Summary

# %%
print("=" * 55)
print("EXPORTED FILES IN models/saved/")
print("=" * 55)
for f in sorted(os.listdir('../models/saved')):
    size_kb = os.path.getsize(f'../models/saved/{f}') / 1024
    print(f"  {f:<45} {size_kb:>8.1f} KB")

print("\n" + "=" * 55)
print("DEPLOYMENT CHECKLIST")
print("=" * 55)
checklist = [
    "pm_pipeline_fd001.joblib  — FD001 full inference pipeline",
    "pm_pipeline_fd002.joblib  — FD002 full inference pipeline",
    "pm_pipeline_fd003.joblib  — FD003 full inference pipeline",
    "pm_pipeline_fd004.joblib  — FD004 full inference pipeline",
    "lstm_target_only_*.keras  — LSTM weights per dataset",
    "scaler_*.joblib           — MinMaxScaler per dataset",
]
for item in checklist:
    path_key = item.split('—')[0].strip().replace('*', 'FD001')
    exists = os.path.exists(f'../models/saved/{path_key}')
    status = "✓" if exists else "✗ (MISSING)"
    print(f"  [{status}] {item}")

print("\nTo add a NEW machine type:")
print("  1. Collect run-to-failure sensor data in (n_cycles × 24) format")
print("  2. Run full_preprocess_pipeline() → save scaler as scaler_NEWTYPE.joblib")
print("  3. Train LSTM → save weights as lstm_target_only_NEWTYPE.keras")
print("  4. Create PredictiveMaintenancePipeline → save as pm_pipeline_newtype.joblib")
print("  5. Call POST /predict with dataset_id='NEWTYPE'")
```
