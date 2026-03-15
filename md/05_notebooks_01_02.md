# 05 — Notebooks 1 & 2: Data Exploration and Preprocessing

> **IDE Agent Instructions:**
> - Create TWO Jupyter notebook files at the paths shown.
> - Each section marked `### Create File:` is a separate notebook.
> - Each `# %% [markdown]` block becomes a **Markdown cell**.
> - Each `# %%` block (without `[markdown]`) becomes a **Code cell**.
> - Use the Jupyter/VS Code notebook format (.ipynb). You may also create them as `.py` files with `# %%` cell markers (compatible with VS Code interactive window).

---

## Notebook 1 — Data Exploration

### Create File: `notebooks/01_data_exploration.ipynb`

```python
# %% [markdown]
# # Notebook 1 — Data Exploration & EDA
#
# **Goal:** Understand the raw C-MAPSS dataset structure, sensor distributions,
# operating condition regimes, and how degradation manifests in sensor signals.
#
# **Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
# - 4 sub-datasets: FD001, FD002, FD003, FD004
# - 21 sensors + 3 operational settings per cycle
# - Run-to-failure trajectories for turbofan engines

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.data_loader import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor import add_piecewise_rul, find_constant_sensors

plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', 30)

datasets = load_all_datasets(data_dir='../data/raw')
print("Loaded datasets:", list(datasets.keys()))
datasets['FD001']['train'].head(3)

# %% [markdown]
# ## 1.1 Dataset Summary Statistics
#
# The four C-MAPSS datasets differ in number of engines, operating conditions,
# and fault modes. Understanding these differences is essential before designing
# a domain adaptation strategy.

# %%
summary_rows = []
for ds_id, splits in datasets.items():
    df = splits['train']
    summary_rows.append({
        'Dataset':           ds_id,
        'Train Engines':     df['unit_id'].nunique(),
        'Test Engines':      splits['test']['unit_id'].nunique(),
        'Total Train Rows':  len(df),
        'Avg Cycles/Engine': round(df.groupby('unit_id')['cycle'].max().mean(), 1),
        'Min Cycles':        df.groupby('unit_id')['cycle'].max().min(),
        'Max Cycles':        df.groupby('unit_id')['cycle'].max().max(),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# %% [markdown]
# **Insight:** FD002 and FD004 have substantially more engines (~260) because
# they simulate 6 operating conditions. FD001 and FD003 operate under a single
# sea-level condition with 100 engines each. Engine lifetimes vary between
# ~130 and ~370 cycles, with multi-condition datasets showing wider variance.

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
for ax, ds_id in zip(axes, ['FD001', 'FD002', 'FD003', 'FD004']):
    life_lengths = datasets[ds_id]['train'].groupby('unit_id')['cycle'].max()
    ax.hist(life_lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(f'{ds_id} Engine Lifetimes')
    ax.set_xlabel('Total Cycles')
    ax.set_ylabel('Count')
    ax.axvline(life_lengths.mean(), color='red', linestyle='--',
               label=f'Mean: {life_lengths.mean():.0f}')
    ax.legend(fontsize=8)
plt.suptitle('Distribution of Engine Lifetime Lengths Across Datasets', fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.2 Identifying Constant Sensors Per Dataset
#
# The paper notes that 7 sensors have constant readings in FD001/FD003.
# These are RETAINED for cross-domain consistency — if they vary in other
# datasets, removing them would break feature-space alignment.

# %%
const_sensors_map = {}
for ds_id, splits in datasets.items():
    const = find_constant_sensors(splits['train'], SENSOR_COLS, threshold=1e-4)
    const_sensors_map[ds_id] = const
    print(f"{ds_id}: {len(const)} constant sensors → {const}")

# %%
# Heatmap: which sensors are constant in which datasets
rows = []
for ds_id, const in const_sensors_map.items():
    row = {s: (1 if s in const else 0) for s in SENSOR_COLS}
    row['dataset'] = ds_id
    rows.append(row)
const_df = pd.DataFrame(rows).set_index('dataset')

fig, ax = plt.subplots(figsize=(18, 3))
sns.heatmap(const_df, cmap='RdYlGn_r', linewidths=0.5,
            annot=True, fmt='d', cbar=False, ax=ax)
ax.set_title('Constant Sensors per Dataset  (1 = constant, 0 = informative)')
ax.set_xlabel('Sensor')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Sensors 1, 5, 6, 10, 16, 18, and 19 are near-constant in
# FD001 and FD003. However, they DO vary in FD002 and FD004. Retaining all
# 24 features maintains a consistent 24-dimensional input space across all
# datasets — a prerequisite for the DANN architecture.

# %%
# %% [markdown]
# ## 1.3 Sensor Distribution Comparison Across Datasets
#
# The paper (Figure 3) shows normalised sensor distributions near failure.
# Here we compare raw distributions across all four datasets.

# %%
sensors_to_compare = ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_12',
                       'sensor_14', 'sensor_15']

fig, axes = plt.subplots(len(sensors_to_compare), 1, figsize=(16, 3 * len(sensors_to_compare)))
colors = {'FD001': '#1f77b4', 'FD002': '#ff7f0e',
          'FD003': '#2ca02c', 'FD004': '#d62728'}

for ax, sensor in zip(axes, sensors_to_compare):
    for ds_id, splits in datasets.items():
        vals = splits['train'][sensor].dropna()
        ax.hist(vals, bins=60, alpha=0.5, label=ds_id,
                color=colors[ds_id], density=True)
    ax.set_title(f'{sensor} Value Distribution Across Datasets')
    ax.set_xlabel('Raw Sensor Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('Raw Sensor Distributions — Cross-Dataset Comparison', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** FD001/FD003 (1 operating condition) show narrow, unimodal
# distributions. FD002/FD004 (6 operating conditions) show multimodal
# distributions reflecting the different operating regimes. This distribution
# shift is exactly what the DANN model must overcome.

# %%
# %% [markdown]
# ## 1.4 Average Sensor Trend vs. Remaining Useful Life (FD001)
#
# Align all engines at their end-of-life point to visualise how sensors
# behave as failure approaches.

# %%
df_with_rul = add_piecewise_rul(datasets['FD001']['train'], max_rul=125)

sensors_to_plot = ['sensor_2', 'sensor_7', 'sensor_11',
                    'sensor_12', 'sensor_14', 'sensor_15']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, sensor in zip(axes.flatten(), sensors_to_plot):
    trend = df_with_rul.groupby('RUL')[sensor].mean()
    std   = df_with_rul.groupby('RUL')[sensor].std()
    ax.plot(trend.index, trend.values, color='steelblue', linewidth=2)
    ax.fill_between(trend.index,
                     trend.values - std.values,
                     trend.values + std.values,
                     alpha=0.15, color='steelblue')
    ax.invert_xaxis()
    ax.set_xlabel('Cycles Before Failure (RUL)')
    ax.set_ylabel('Mean Sensor Value')
    ax.set_title(f'{sensor} vs. RUL (FD001)')
    ax.grid(alpha=0.3)

plt.suptitle('Average Sensor Trends as Engines Approach Failure (FD001)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Sensors 11, 12, and 14 show clear monotonic trends as RUL
# approaches zero — these are the primary degradation indicators for the
# HPC (High Pressure Compressor) fault mode in FD001/FD002.
# Sensor 7 and sensor 2 are noisier but still carry predictive signal.
# This guides SHAP feature importance analysis in Notebook 07.

# %%
# %% [markdown]
# ## 1.5 Operating Condition Clustering (FD002 / FD004)

# %%
OP_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
df_fd002 = datasets['FD002']['train'].copy()

scaler  = StandardScaler()
op_sc   = scaler.fit_transform(df_fd002[OP_COLS])
kmeans  = KMeans(n_clusters=6, random_state=42, n_init=10)
df_fd002['op_cluster'] = kmeans.fit_predict(op_sc)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
scatter = axes[0].scatter(df_fd002['op_setting_1'], df_fd002['op_setting_2'],
                           c=df_fd002['op_cluster'], cmap='tab10', alpha=0.3, s=5)
plt.colorbar(scatter, ax=axes[0], label='Cluster')
axes[0].set_xlabel('op_setting_1 (Altitude proxy)')
axes[0].set_ylabel('op_setting_2 (Throttle Angle proxy)')
axes[0].set_title('FD002 Operating Condition Clusters (K=6)')

scatter2 = axes[1].scatter(df_fd002['op_setting_1'], df_fd002['op_setting_3'],
                            c=df_fd002['op_cluster'], cmap='tab10', alpha=0.3, s=5)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')
axes[1].set_xlabel('op_setting_1')
axes[1].set_ylabel('op_setting_3 (Mach proxy)')
axes[1].set_title('FD002 — Setting 1 vs Setting 3')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The 6 operating conditions in FD002/FD004 form well-separated
# clusters in the operational settings space. A model trained on FD001 (single
# sea-level condition, bottom-left cluster only) will produce biased features
# for the other 5 operating regimes — precisely what DANN addresses.

# %%
# %% [markdown]
# ## 1.6 Per-Unit RUL Profile: Visualising Piecewise Linear Targets

# %%
df_rul_plot = add_piecewise_rul(datasets['FD001']['train'], max_rul=125)
sample_units = [1, 5, 10, 20, 30, 50, 75, 100]

fig, ax = plt.subplots(figsize=(14, 6))
for unit in sample_units:
    ud = df_rul_plot[df_rul_plot['unit_id'] == unit]
    ax.plot(ud['cycle'], ud['RUL'], alpha=0.7, label=f'Unit {unit}')
ax.axhline(125, color='gray', linestyle=':', linewidth=1, label='RUL cap = 125')
ax.set_xlabel('Cycle')
ax.set_ylabel('RUL (piecewise linear, capped at 125)')
ax.set_title('Piecewise Linear RUL Labels — FD001 (Selected Engines)')
ax.legend(ncol=4, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The piecewise linear target treats all engines as equally healthy
# during the flat phase (RUL = 125) then assigns linearly decreasing RUL.
# The flat region length varies per unit because some engines start degrading
# earlier than others, but all degrade monotonically once past the inflection point.
```

---

## Notebook 2 — Preprocessing & Noise Handling

### Create File: `notebooks/02_preprocessing_noise_handling.ipynb`

```python
# %% [markdown]
# # Notebook 2 — Preprocessing & Noise Handling
#
# **Goal:** Implement the full preprocessing pipeline:
# 1. Noise characterisation (SNR per sensor)
# 2. Savitzky-Golay smoothing per engine unit
# 3. Synthetic missing data injection and imputation
# 4. Per-dataset min-max normalisation (individual scalers)
# 5. Time-window feature extraction
# 6. Save processed arrays for use in downstream notebooks

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.data_loader    import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor   import (apply_savgol_filter, inject_missing_data,
                                 impute_missing, fit_normaliser, apply_normaliser,
                                 add_piecewise_rul, find_constant_sensors,
                                 full_preprocess_pipeline, snr_db)
from src.windowing      import create_windows, create_windows_inference

os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models/saved',   exist_ok=True)

datasets = load_all_datasets(data_dir='../data/raw')
print("Datasets loaded:", list(datasets.keys()))

# %% [markdown]
# ## 2.1 Sensor Noise Characterisation
#
# We measure the Signal-to-Noise Ratio (SNR) in dB for each sensor.
# SNR < 20 dB indicates sensors that may benefit from smoothing.

# %%
df_fd001 = datasets['FD001']['train']

snr_scores = {}
for sensor in SENSOR_COLS:
    vals = df_fd001[df_fd001['unit_id'] == 1][sensor].values
    snr_scores[sensor] = snr_db(vals)

snr_series = pd.Series(snr_scores).sort_values()

fig, ax = plt.subplots(figsize=(12, 6))
snr_series.plot(kind='barh', ax=ax, color='coral', edgecolor='black', alpha=0.85)
ax.axvline(20, color='navy', linestyle='--', linewidth=1.5, label='SNR = 20 dB threshold')
ax.set_xlabel('SNR (dB)')
ax.set_title('Signal-to-Noise Ratio per Sensor (FD001, Unit 1)')
ax.legend()
plt.tight_layout()
plt.show()

print("\nSensors below 20 dB (smoothing candidates):")
print(snr_series[snr_series < 20].to_string())

# %% [markdown]
# **Insight:** Sensors with SNR below 20 dB contain more noise relative to
# their signal. The Savitzky-Golay filter will be applied to all sensors but
# is most impactful for low-SNR signals. We use a conservative filter (window=11,
# polyorder=3) to smooth while preserving the degradation trend shape.

# %%
# %% [markdown]
# ## 2.2 Savitzky-Golay Filter — Visual Validation

# %%
unit_df   = df_fd001[df_fd001['unit_id'] == 1].copy()
smoothed  = apply_savgol_filter(unit_df, SENSOR_COLS, window_length=11, polyorder=3)

sensor_to_show = 'sensor_11'
residual = unit_df[sensor_to_show].values - smoothed[sensor_to_show].values

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(unit_df['cycle'], unit_df[sensor_to_show],
             alpha=0.6, label='Raw', color='steelblue')
axes[0].plot(smoothed['cycle'], smoothed[sensor_to_show],
             color='red', linewidth=2, label='SG Filtered')
axes[0].set_title(f'{sensor_to_show} — Unit 1 (Raw vs Filtered)')
axes[0].set_xlabel('Cycle')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(unit_df['cycle'], residual, color='gray', alpha=0.7)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].fill_between(unit_df['cycle'], residual, 0, alpha=0.2, color='orange')
axes[1].set_title('Residual (Removed Noise Component)')
axes[1].set_xlabel('Cycle')
axes[1].set_ylabel('Residual Value')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The residual (noise) oscillates near zero with no systematic
# trend, confirming the filter removes only high-frequency noise and leaves
# the low-frequency degradation trend intact. The degradation profile's
# monotonic shape is fully preserved in the filtered signal.

# %%
# %% [markdown]
# ## 2.3 Synthetic Missing Data — Injection & Imputation

# %%
unit_3_orig    = df_fd001[df_fd001['unit_id'] == 3].copy()
unit_3_missing = inject_missing_data(unit_3_orig, SENSOR_COLS, missing_rate=0.05)
unit_3_imputed = impute_missing(unit_3_missing, SENSOR_COLS, method='linear')

missing_counts = unit_3_missing[SENSOR_COLS].isna().sum()
print("Missing values injected per sensor:")
print(missing_counts[missing_counts > 0].to_string())

for sensor in ['sensor_4', 'sensor_11', 'sensor_14']:
    missing_mask = unit_3_missing[sensor].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        continue

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(unit_3_orig['cycle'], unit_3_orig[sensor],
            linewidth=2, label='Original', alpha=0.8)
    ax.scatter(unit_3_orig.loc[missing_mask, 'cycle'],
               unit_3_orig.loc[missing_mask, sensor],
               color='red', zorder=5, s=50,
               label=f'Injected NaN ({n_missing} pts)')
    ax.plot(unit_3_imputed['cycle'], unit_3_imputed[sensor],
            linestyle='--', color='green', linewidth=1.5, label='Imputed')
    ax.set_title(f'{sensor} — Missing Data Imputation (Unit 3, FD001)')
    ax.set_xlabel('Cycle')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Insight:** Linear interpolation accurately recovers missing sensor values
# within smooth degradation regions. At series edges, backward fill handles
# any remaining NaN values. This dual strategy is robust to both interior
# sensor glitches (majority case) and early-lifecycle gaps.

# %%
# %% [markdown]
# ## 2.4 Per-Dataset Normalisation (Independent Scalers)
#
# Each dataset is normalised with its OWN scaler, as specified in Equation 15
# of the paper. This deliberately preserves cross-dataset distribution shift,
# which is the input to the domain adversarial training process.

# %%
scalers = {}
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df_tr  = datasets[ds_id]['train']
    df_te  = datasets[ds_id]['test']

    df_tr_processed, df_te_processed, scaler = full_preprocess_pipeline(
        df_train     = df_tr,
        df_test      = df_te,
        feature_cols = FEATURE_COLS,
        sensor_cols  = SENSOR_COLS,
        smooth       = True,
        max_rul      = 125,
        scaler_save_path = f'../models/saved/scaler_{ds_id}.joblib'
    )

    scalers[ds_id] = scaler
    datasets[ds_id]['train_norm'] = df_tr_processed
    datasets[ds_id]['test_norm']  = df_te_processed

    print(f"{ds_id}: scaler saved, train shape = {df_tr_processed.shape}")

# %%
# Visualise: distributions before vs after normalisation for FD001
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
datasets['FD001']['train'][SENSOR_COLS].hist(
    bins=30, ax=axes[0], layout=(3, 7), figsize=(18, 6))
axes[0].set_title('Before Normalisation')

datasets['FD001']['train_norm'][SENSOR_COLS].hist(
    bins=30, ax=axes[1], layout=(3, 7), figsize=(18, 6))
axes[1].set_title('After Min-Max Normalisation [0, 1]')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** After normalisation, all sensor values are in [0, 1] within
# each dataset. Crucially, the SAME raw sensor reading (e.g., sensor_11 = 554)
# maps to DIFFERENT normalised values across datasets, preserving the
# inter-dataset distribution shift that DANN needs to adapt across.

# %%
# Cross-dataset distribution shift on sensor_11 (post-normalisation)
fig, ax = plt.subplots(figsize=(12, 5))
colors = {'FD001': '#1f77b4', 'FD002': '#ff7f0e',
          'FD003': '#2ca02c', 'FD004': '#d62728'}
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    vals = datasets[ds_id]['train_norm']['sensor_11']
    vals.hist(bins=60, ax=ax, alpha=0.5, label=ds_id,
              color=colors[ds_id], density=True)
ax.set_title('sensor_11 Normalised Distributions — Cross-Dataset Shift (Preserved)')
ax.set_xlabel('Normalised Value [0, 1]')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** The normalised distributions of sensor_11 differ noticeably
# across datasets. FD002/FD004 show multimodal distributions (6 operating
# conditions), while FD001/FD003 show unimodal distributions. This is the
# distribution shift the DANN architecture must learn to bridge.

# %%
# %% [markdown]
# ## 2.5 Time-Window Feature Extraction
#
# Implements function h_t from Section 3.2 of the paper.
# Each window = (T_w=30) consecutive cycles of sensor readings.

# %%
WINDOW_SIZE = 30

for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df = datasets[ds_id]['train_norm']
    X, y, unit_ids = create_windows(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    datasets[ds_id]['X_train']   = X
    datasets[ds_id]['y_train']   = y
    datasets[ds_id]['unit_ids']  = unit_ids
    print(f"{ds_id}: X_train={X.shape}, y_train={y.shape}, "
          f"windows per engine ≈ {len(X)//df['unit_id'].nunique():.0f}")

# %%
# Visualise one sample window
sample_window = datasets['FD001']['X_train'][500]   # shape: (30, 24)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for i, col in enumerate(FEATURE_COLS[:6]):
    axes[0].plot(sample_window[:, i], label=col, alpha=0.8)
axes[0].set_xlabel('Timestep within Window (0 = oldest)')
axes[0].set_ylabel('Normalised Value')
axes[0].set_title('Sample Time Window — First 6 Features')
axes[0].legend(ncol=3, fontsize=8)
axes[0].grid(alpha=0.3)

# Heatmap of the full window
im = axes[1].imshow(sample_window.T, aspect='auto', cmap='viridis',
                     interpolation='nearest')
axes[1].set_xlabel('Timestep within Window')
axes[1].set_ylabel('Feature Index')
axes[1].set_title('Full Window Heatmap (30 timesteps × 24 features)')
axes[1].set_yticks(range(len(FEATURE_COLS)))
axes[1].set_yticklabels(FEATURE_COLS, fontsize=6)
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Each window is a 30×24 matrix: 30 cycles of history, 24 features.
# The LSTM processes this left-to-right, building a hidden state that captures
# how sensor values have evolved over the past 30 cycles before making a RUL
# prediction. The heatmap shows how different sensors activate at different
# points in the window — key for understanding what temporal patterns the
# LSTM learns to detect.

# %%
# %% [markdown]
# ## 2.6 Save Processed Data

# %%
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    np.save(f'../data/processed/X_train_{ds_id}.npy', datasets[ds_id]['X_train'])
    np.save(f'../data/processed/y_train_{ds_id}.npy', datasets[ds_id]['y_train'])

print("All processed arrays saved to data/processed/")
print("Scalers saved to models/saved/")
```
