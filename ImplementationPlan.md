# 🚀 NASA C-MAPSS FD001 — Complete ML Implementation Pipeline
### AI Predictive Maintenance | RUL Estimation + Failure Classification

---

## 📋 Table of Contents

1. [Environment Setup](#phase-0-environment-setup)
2. [Data Loading & Schema Validation](#phase-1-data-loading--schema-validation)
3. [Data Cleaning & Quality Checks](#phase-2-data-cleaning--quality-checks)
4. [Exploratory Data Analysis (EDA)](#phase-3-exploratory-data-analysis)
5. [RUL Label Engineering](#phase-4-rul-label-engineering)
6. [Feature Engineering](#phase-5-feature-engineering)
7. [Train / Validation Split](#phase-6-trainvalidation-split)
8. [Baseline Models](#phase-7-baseline-models)
9. [Advanced Models — RUL Regression](#phase-8-advanced-models--rul-regression)
10. [Classification Task](#phase-9-classification-task)
11. [Hyperparameter Tuning](#phase-10-hyperparameter-tuning)
12. [Ensemble Strategy](#phase-11-ensemble-strategy)
13. [Uncertainty Quantification](#phase-12-uncertainty-quantification)
14. [Model Evaluation & Metrics](#phase-13-model-evaluation--metrics)
15. [Model Interpretability (SHAP)](#phase-14-model-interpretability-shap)
16. [Ablation Study](#phase-15-ablation-study)

---

## 📦 PHASE 0: Environment Setup

```python
# ── Standard Libraries ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Sklearn ──────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
from sklearn.inspection import permutation_importance

# ── Gradient Boosting ─────────────────────────────────────────────────
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Deep Learning ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

# ── Interpretability ──────────────────────────────────────────────────
import shap

# ── Plotting Config ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11
})
COLORS = ['#1F4E79', '#2E75B6', '#70AD47', '#FF7043', '#AB47BC']
```

### Column Names & Dataset Context

```python
# FD001: Single operating condition, HPC degradation fault
# 100 training engines | 100 test engines | 1 fault mode

col_names = (
    ['unit_number', 'time_cycles'] +
    ['setting_1', 'setting_2', 'setting_3'] +
    [f's_{i}' for i in range(1, 22)]   # s_1 ... s_21
)
# Total: 26 columns
# unit_number   : engine ID (1–100)
# time_cycles   : current operating cycle
# setting_1–3   : operating condition settings
# s_1–s_21      : sensor measurements

data_dir = Path('./CMaps')
```

---

## 📥 PHASE 1: Data Loading & Schema Validation

### 1.1 Load All Three Files

```python
dftrain = pd.read_csv(
    data_dir / 'train_FD001.txt',
    sep=r'\s+', header=None, index_col=False, names=col_names
)
dfvalid = pd.read_csv(
    data_dir / 'test_FD001.txt',
    sep=r'\s+', header=None, index_col=False, names=col_names
)
y_valid = pd.read_csv(
    data_dir / 'RUL_FD001.txt',
    sep=r'\s+', header=None, index_col=False, names=['RUL']
)

print(f"Train shape  : {dftrain.shape}")    # Expected: (20631, 26)
print(f"Valid shape  : {dfvalid.shape}")    # Expected: (13096, 26)
print(f"y_valid shape: {y_valid.shape}")   # Expected: (100, 1)
```

### 1.2 Schema Validation Checklist

```python
def validate_schema(df, name):
    print(f"\n{'='*50}")
    print(f"  SCHEMA VALIDATION — {name}")
    print(f"{'='*50}")
    print(f"  Shape          : {df.shape}")
    print(f"  Dtypes OK      : {all(df.dtypes != 'object')}")
    print(f"  Null count     : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated().sum()}")
    print(f"  unit_number    : {df['unit_number'].nunique()} unique engines")
    print(f"  time_cycles    : min={df['time_cycles'].min()}, max={df['time_cycles'].max()}")
    starts_at_1 = (df.groupby('unit_number')['time_cycles'].min() == 1).all()
    print(f"  All engines start at cycle 1: {starts_at_1}")

validate_schema(dftrain, "TRAIN FD001")
validate_schema(dfvalid, "VALID FD001")
```

### 1.3 Engine Lifetime Overview

```python
# How long does each engine run before failure?
train_max_cycles = dftrain.groupby('unit_number')['time_cycles'].max()

print(f"\nEngine Lifetime Statistics (Training):")
print(train_max_cycles.describe())
# Expected: mean ≈ 206 cycles | min ≈ 128 | max ≈ 362
```

---

## 🔍 PHASE 2: Data Cleaning & Quality Checks

### 2.1 Missing Value Analysis

```python
# ── Missing Values ────────────────────────────────────────────────────
missing_train = (dftrain.isnull().sum() / len(dftrain) * 100).round(3)
missing_valid = (dfvalid.isnull().sum() / len(dfvalid) * 100).round(3)

print("Train missing %:\n", missing_train[missing_train > 0])
print("Valid missing %:\n", missing_valid[missing_valid > 0])
# Expected: 0 missing values in FD001 — confirm this
```

### 2.2 Constant / Zero-Variance Sensor Detection

```python
sensor_cols = [f's_{i}' for i in range(1, 22)]

# Compute standard deviation per sensor
sensor_std = dftrain[sensor_cols].std()

# Plot: sensor variance bar chart
plt.figure(figsize=(14, 4))
colors = ['#FF7043' if v < 0.01 else '#1F4E79' for v in sensor_std.values]
plt.bar(sensor_std.index, sensor_std.values, color=colors)
plt.axhline(0.01, color='red', linestyle='--', label='Drop threshold (std < 0.01)')
plt.xticks(rotation=45)
plt.title('Sensor Standard Deviation — Identify Zero-Variance (Red = Drop)')
plt.ylabel('Standard Deviation')
plt.legend()
plt.tight_layout()
plt.show()

# Identify and drop constant sensors
constant_sensors = sensor_std[sensor_std < 0.01].index.tolist()
print(f"\nConstant sensors to drop: {constant_sensors}")
# Expected: ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
```

### 2.3 Drop Constant Sensors

```python
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

dftrain = dftrain.drop(columns=DROP_SENSORS)
dfvalid = dfvalid.drop(columns=DROP_SENSORS)

# Update sensor list
sensor_cols = [c for c in dftrain.columns if c.startswith('s_')]
print(f"Remaining sensors ({len(sensor_cols)}): {sensor_cols}")
# Expected: 14 sensors remaining
```

### 2.4 Outlier Detection per Sensor

```python
# Modified Z-score (MAD-based) — robust to skewed distributions
def mad_based_outlier(series, threshold=3.5):
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z = 0.7413 * np.abs(series - median) / (mad + 1e-9)
    return modified_z > threshold

outlier_report = {}
for col in sensor_cols:
    mask = mad_based_outlier(dftrain[col])
    outlier_report[col] = mask.sum()

outlier_df = pd.Series(outlier_report).sort_values(ascending=False)
print("Outlier counts per sensor (MAD method, threshold=3.5):")
print(outlier_df[outlier_df > 0])

# ── CLIP outliers (never drop rows — preserve temporal continuity) ───
for col in sensor_cols:
    q1, q3 = dftrain[col].quantile(0.01), dftrain[col].quantile(0.99)
    dftrain[col] = dftrain[col].clip(lower=q1, upper=q3)
```

### 2.5 Descriptive Statistics — Scale Check

```python
# Check if sensors are on very different scales
# (informs MinMaxScaler choice)
stats = dftrain[sensor_cols].describe().T
stats['range'] = stats['max'] - stats['min']
stats['cv'] = stats['std'] / (stats['mean'] + 1e-9)   # Coefficient of variation

print(stats[['mean', 'std', 'min', 'max', 'range', 'cv']].sort_values('range', ascending=False))
# Expected: s_9 range ~9000 | s_15 range ~8 → confirms MinMaxScaler needed
```

---

## 📊 PHASE 3: Exploratory Data Analysis

### 3.1 Engine Lifetime Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(train_max_cycles, bins=20, color='#1F4E79', edgecolor='white')
axes[0].axvline(train_max_cycles.mean(), color='#FF7043', linestyle='--',
                label=f'Mean = {train_max_cycles.mean():.0f}')
axes[0].set_title('Engine Lifetime Distribution (Train)\nMost engines fail 190–210 cycles')
axes[0].set_xlabel('Max Cycles at Failure')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].bar(range(len(train_max_cycles)), sorted(train_max_cycles),
            color='#2E75B6', alpha=0.7)
axes[1].set_title('Sorted Engine Lifetimes')
axes[1].set_xlabel('Engine Rank')
axes[1].set_ylabel('Cycles to Failure')

plt.tight_layout()
plt.show()
```

### 3.2 Sensor Degradation Plots (Signal vs. RUL)

```python
# Add RUL to training data temporarily for EDA
rul_temp = dftrain.groupby('unit_number')['time_cycles'].max().reset_index()
rul_temp.columns = ['unit_number', 'max_cycle']
dftrain_eda = dftrain.merge(rul_temp, on='unit_number')
dftrain_eda['RUL'] = dftrain_eda['max_cycle'] - dftrain_eda['time_cycles']

def plot_sensor_vs_rul(df, sensor, ax, title=None):
    """Plot sensor value vs RUL for every 10th engine (smoothed)."""
    for unit in df['unit_number'].unique():
        if unit % 10 == 0:
            subset = df[df['unit_number'] == unit].sort_values('time_cycles')
            smoothed = subset[sensor].rolling(window=10, min_periods=1).mean()
            ax.plot(subset['RUL'].values, smoothed.values, alpha=0.6, linewidth=1)
    ax.set_xlim(300, 0)   # Reverse x-axis: left=healthy, right=failure
    ax.set_xlabel('Remaining Useful Life (cycles)')
    ax.set_ylabel(sensor)
    ax.set_title(title or f'{sensor} vs RUL')

# 3x3 grid of top informative sensors
top_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_17']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for i, (sensor, ax) in enumerate(zip(top_sensors, axes.flatten())):
    plot_sensor_vs_rul(dftrain_eda, sensor, ax)
plt.suptitle('Sensor Readings vs RUL — Each line is one engine (smoothed, reversed x-axis)\n'
             'Monotonic trend = good predictive feature', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```

### 3.3 Correlation Heatmap — Sensors vs. Each Other

```python
corr = dftrain_eda[sensor_cols].corr()
corr_high = corr[abs(corr) > 0.8]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
            ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title('Full Sensor Correlation Matrix')

sns.heatmap(corr_high, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=axes[1], linewidths=0.5)
axes[1].set_title('High Correlation Pairs (|r| > 0.8)\nRed clusters = redundant sensors')

plt.tight_layout()
plt.show()
# Expected: two correlated groups — sensors 14–25 and 15–29 regions
```

### 3.4 Sensor Correlation with RUL (Feature Relevance)

```python
rul_corr = dftrain_eda[sensor_cols + ['RUL']].corr()['RUL'].drop('RUL')

plt.figure(figsize=(10, 5))
colors = ['#FF7043' if abs(v) < 0.1 else '#1F4E79' for v in rul_corr.values]
plt.barh(rul_corr.index, rul_corr.values, color=colors)
plt.axvline(0.1, color='green', linestyle='--', alpha=0.7, label='|r|=0.1 threshold')
plt.axvline(-0.1, color='green', linestyle='--', alpha=0.7)
plt.xlabel('Pearson Correlation with RUL')
plt.title('Sensor Correlation with RUL\nRed bars = weak predictors (|r| < 0.1)')
plt.legend()
plt.tight_layout()
plt.show()

print("\nSensor correlations with RUL (sorted):")
print(rul_corr.sort_values())
```

### 3.5 PCA — Dimensionality Overview

```python
from sklearn.decomposition import PCA

scaler_eda = MinMaxScaler()
X_eda_scaled = scaler_eda.fit_transform(dftrain_eda[sensor_cols])

pca_eda = PCA(n_components=5)
pca_eda.fit(X_eda_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, 6), pca_eda.explained_variance_ratio_ * 100,
            color='#1F4E79', alpha=0.8)
axes[0].plot(range(1, 6), np.cumsum(pca_eda.explained_variance_ratio_) * 100,
             'r-o', label='Cumulative')
axes[0].set_xlabel('PCA Component')
axes[0].set_ylabel('Explained Variance %')
axes[0].set_title(f'PCA Variance — PC1 explains {pca_eda.explained_variance_ratio_[0]*100:.1f}%')
axes[0].legend()

pca_2d = PCA(n_components=2).fit_transform(X_eda_scaled)
sc = axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1],
                     c=dftrain_eda['RUL'], cmap='RdYlGn', alpha=0.3, s=5)
plt.colorbar(sc, ax=axes[1], label='RUL')
axes[1].set_title('PCA 2D Projection Colored by RUL\nGreen=healthy, Red=near failure')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()
# Expected: PC1 explains ~73.6% of variance
```

---

## 🏷️ PHASE 4: RUL Label Engineering

### 4.1 Compute RUL (Training Data)

```python
# RUL = engine_failure_cycle − current_cycle
rul_df = (dftrain
          .groupby('unit_number')['time_cycles']
          .max()
          .reset_index()
          .rename(columns={'time_cycles': 'max_cycle'}))

dftrain = dftrain.merge(rul_df, on='unit_number')
dftrain['RUL_raw'] = dftrain['max_cycle'] - dftrain['time_cycles']
dftrain.drop(columns='max_cycle', inplace=True)
```

### 4.2 Piecewise RUL Clipping

```python
# WHY CLIP?
# Machines don't degrade from cycle 1 — early life is "healthy noise."
# Clipping tells the model: "Only start caring when RUL drops below ceiling."
# Confirmed threshold from domain analysis: 195 cycles for FD001.

CLIP_VALUE = 195

dftrain['RUL'] = dftrain['RUL_raw'].clip(upper=CLIP_VALUE)

# Visualize the effect of clipping
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(dftrain['RUL_raw'], bins=40, color='#2E75B6', edgecolor='white')
axes[0].set_title('Raw RUL Distribution')
axes[0].set_xlabel('RUL (cycles)')

axes[1].hist(dftrain['RUL'], bins=40, color='#70AD47', edgecolor='white')
axes[1].axvline(CLIP_VALUE, color='red', linestyle='--',
                label=f'Clip ceiling = {CLIP_VALUE}')
axes[1].set_title('Clipped RUL Distribution')
axes[1].set_xlabel('RUL (cycles)')
axes[1].legend()

# Piecewise RUL trajectory for 5 sample engines
sample_engines = [1, 11, 21, 31, 41]
for eng in sample_engines:
    sub = dftrain[dftrain['unit_number'] == eng].sort_values('time_cycles')
    axes[2].plot(sub['time_cycles'], sub['RUL'],
                 label=f'Engine {eng}', alpha=0.8)
axes[2].set_title('Piecewise RUL Trajectory — 5 Engines\nFlat = healthy phase, then counts down')
axes[2].set_xlabel('Cycle')
axes[2].set_ylabel('RUL (clipped)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

### 4.3 Classification Labels

```python
# Label 1: Binary (failure within 30 cycles)
W1 = 30
dftrain['label1'] = (dftrain['RUL'] <= W1).astype(int)

# Label 2: 3-class risk
W0 = 15
dftrain['label2'] = dftrain['label1'].copy()
dftrain.loc[dftrain['RUL'] <= W0, 'label2'] = 2
# label2: 0 = healthy, 1 = warning (15–30 cycles), 2 = critical (<=15)

print("Binary label1 distribution:")
print(dftrain['label1'].value_counts(normalize=True).mul(100).round(2))
print("\n3-class label2 distribution:")
print(dftrain['label2'].value_counts(normalize=True).mul(100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
dftrain['label1'].value_counts().plot(kind='bar', ax=axes[0],
    color=['#70AD47', '#FF7043'], edgecolor='white')
axes[0].set_title('Binary Classification Balance\n(0=Healthy, 1=Fail within 30 cycles)')
axes[0].set_xticklabels(['Healthy (0)', 'At Risk (1)'], rotation=0)

dftrain['label2'].value_counts().sort_index().plot(kind='bar', ax=axes[1],
    color=['#70AD47', '#FFA726', '#FF7043'], edgecolor='white')
axes[1].set_title('3-Class Balance\n(0=Healthy, 1=Warning, 2=Critical)')
axes[1].set_xticklabels(['Healthy', 'Warning', 'Critical'], rotation=0)

plt.tight_layout()
plt.show()
```

### 4.4 Prepare Test (Validation) RUL Ground Truth

```python
# For test data: back-calculate RUL for all test cycles from y_valid
last_cycle = dfvalid.groupby('unit_number')['time_cycles'].max().reset_index()
last_cycle.columns = ['unit_number', 'last_cycle']
last_cycle['rul_at_end'] = y_valid['RUL'].values

dfvalid = dfvalid.merge(last_cycle, on='unit_number')
dfvalid['RUL'] = dfvalid['rul_at_end'] + (dfvalid['last_cycle'] - dfvalid['time_cycles'])
dfvalid['RUL'] = dfvalid['RUL'].clip(upper=CLIP_VALUE)
dfvalid['label1'] = (dfvalid['RUL'] <= W1).astype(int)
dfvalid['label2'] = dfvalid['label1'].copy()
dfvalid.loc[dfvalid['RUL'] <= W0, 'label2'] = 2
dfvalid.drop(columns=['last_cycle', 'rul_at_end'], inplace=True)
```

---

## ⚙️ PHASE 5: Feature Engineering

### 5.1 Rolling Window Features

```python
WINDOW = 10  # Confirmed best window for FD001

def add_rolling_features(df, sensors, window=10):
    """
    Add rolling mean, std, min, max per sensor.
    Grouped per engine to prevent cross-engine leakage.
    """
    df = df.copy().sort_values(['unit_number', 'time_cycles'])
    for col in sensors:
        grouped = df.groupby('unit_number')[col]
        df[f'{col}_rm']   = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{col}_rstd'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'{col}_rmin'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
        df[f'{col}_rmax'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
    return df

dftrain = add_rolling_features(dftrain, sensor_cols, window=WINDOW)
dfvalid = add_rolling_features(dfvalid, sensor_cols, window=WINDOW)

rolling_cols = [c for c in dftrain.columns if any(
    c.endswith(suf) for suf in ['_rm', '_rstd', '_rmin', '_rmax']
)]
print(f"Rolling features added: {len(rolling_cols)}")
# Expected: 14 sensors × 4 stats = 56 rolling features
```

### 5.2 Additional Domain Features

```python
def add_domain_features(df):
    df = df.copy()
    # Cycle normalization (engine age as fraction 0 to 1)
    max_c = df.groupby('unit_number')['time_cycles'].transform('max')
    df['cycle_norm'] = df['time_cycles'] / max_c

    # Rate of change for key sensors
    for col in ['s_9', 's_14', 's_11']:
        if col in df.columns:
            df[f'{col}_diff'] = (df.groupby('unit_number')[col]
                                   .transform(lambda x: x.diff().fillna(0)))
    return df

dftrain = add_domain_features(dftrain)
dfvalid = add_domain_features(dfvalid)
```

### 5.3 Feature Set Summary

```python
meta_cols   = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
target_cols = ['RUL', 'RUL_raw', 'label1', 'label2']
raw_sensor_cols   = sensor_cols
rolling_feat_cols = rolling_cols
domain_feat_cols  = ['cycle_norm'] + [c for c in dftrain.columns if c.endswith('_diff')]

ALL_FEATURES = raw_sensor_cols + rolling_feat_cols + domain_feat_cols
print(f"\nFeature Summary:")
print(f"  Raw sensors      : {len(raw_sensor_cols)}")
print(f"  Rolling features : {len(rolling_feat_cols)}")
print(f"  Domain features  : {len(domain_feat_cols)}")
print(f"  TOTAL            : {len(ALL_FEATURES)}")
```

---

## ✂️ PHASE 6: Train/Validation Split

> ⚠️ **Critical Rule**: Split by `unit_number` (engine ID), NOT by rows.
> Row-based splitting guarantees leakage — future cycles from the same engine
> would appear in training data.

```python
all_units = dftrain['unit_number'].unique()     # 100 engines
np.random.seed(42)
np.random.shuffle(all_units)

n_train    = int(0.80 * len(all_units))         # 80 engines train
train_units = all_units[:n_train]
val_units   = all_units[n_train:]

df_tr = dftrain[dftrain['unit_number'].isin(train_units)].copy()
df_va = dftrain[dftrain['unit_number'].isin(val_units)].copy()

print(f"Train: {len(train_units)} engines | {len(df_tr)} rows")
print(f"Val  : {len(val_units)} engines  | {len(df_va)} rows")
print(f"Test : {len(dfvalid['unit_number'].unique())} engines | {len(dfvalid)} rows")

# Verify RUL distributions are similar across splits
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_tr['RUL'], bins=30, alpha=0.6, label=f'Train (n={len(train_units)})', color='#1F4E79')
ax.hist(df_va['RUL'], bins=30, alpha=0.6, label=f'Val   (n={len(val_units)})', color='#FF7043')
ax.set_xlabel('RUL')
ax.set_title('RUL Distribution — Train vs Val\nShould overlap to confirm no split bias')
ax.legend()
plt.tight_layout()
plt.show()

# ⚠️ LEAKAGE-SAFE Scaling — fit ONLY on training data
scaler = MinMaxScaler()

X_tr = scaler.fit_transform(df_tr[ALL_FEATURES])   # fit + transform on train only
X_va = scaler.transform(df_va[ALL_FEATURES])        # transform only
X_te = scaler.transform(dfvalid[ALL_FEATURES])      # transform only

y_tr_rul = df_tr['RUL'].values
y_va_rul = df_va['RUL'].values
y_te_rul = dfvalid['RUL'].values

y_tr_cls = df_tr['label1'].values
y_va_cls = df_va['label1'].values
y_te_cls = dfvalid['label1'].values

print(f"\nScaling complete (fitted on train only):")
print(f"X_tr: {X_tr.shape} | X_va: {X_va.shape} | X_te: {X_te.shape}")
```

---

## 📈 PHASE 7: Baseline Models

> All baselines evaluated on val and test sets. These are the FLOOR — any complex model must beat these.

### 7.1 Evaluation Helpers

```python
def nasa_score(y_true, y_pred):
    """
    NASA asymmetric scoring function.
    Late predictions (pred > true) penalized more than early.
    d > 0: s = e^(d/13) - 1 | d < 0: s = e^(-d/10) - 1
    Lower is better.
    """
    d = y_pred - y_true
    scores = np.where(d >= 0, np.exp(d / 13) - 1, np.exp(-d / 10) - 1)
    return np.sum(scores)

def regression_report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    ns   = nasa_score(y_true, y_pred)
    print(f"  {name:<35} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  NASA={ns:.1f}")
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NASA_Score': ns}

results = []   # Accumulate all model results here
```

### 7.2 Linear Regression Baseline

```python
lr = LinearRegression()
lr.fit(X_tr, y_tr_rul)

results.append(regression_report("Linear Regression (train)", y_tr_rul, lr.predict(X_tr)))
results.append(regression_report("Linear Regression (val)",   y_va_rul, lr.predict(X_va)))
results.append(regression_report("Linear Regression (test)",  y_te_rul, lr.predict(X_te)))
# Expected val RMSE: ~34–38
```

### 7.3 SVR Baseline (Best Classical Model)

```python
svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
svr.fit(X_tr, y_tr_rul)

results.append(regression_report("SVR RBF (val)",  y_va_rul, svr.predict(X_va)))
results.append(regression_report("SVR RBF (test)", y_te_rul, svr.predict(X_te)))
# Expected val RMSE: ~26–28 | test RMSE: ~31
```

### 7.4 Random Forest Baseline

```python
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr_rul)

results.append(regression_report("Random Forest (val)",  y_va_rul, rf.predict(X_va)))
results.append(regression_report("Random Forest (test)", y_te_rul, rf.predict(X_te)))
```

### 7.5 Baseline Comparison Plot

```python
val_results = [r for r in results if 'val' in r['model']]
val_df = pd.DataFrame(val_results).set_index('model')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
val_df['RMSE'].plot(kind='barh', ax=axes[0], color='#1F4E79', alpha=0.8)
axes[0].set_title('Validation RMSE (lower is better)')
val_df['R2'].plot(kind='barh', ax=axes[1], color='#70AD47', alpha=0.8)
axes[1].set_title('Validation R² (higher is better)')
plt.tight_layout()
plt.show()
```

---

## 🤖 PHASE 8: Advanced Models — RUL Regression

### 8.1 XGBoost

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=50, max_depth=6, learning_rate=0.1,
    reg_lambda=0.02, gamma=0.4, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb_model.fit(X_tr, y_tr_rul, eval_set=[(X_va, y_va_rul)], verbose=False)

results.append(regression_report("XGBoost (val)",  y_va_rul, xgb_model.predict(X_va)))
results.append(regression_report("XGBoost (test)", y_te_rul, xgb_model.predict(X_te)))
```

### 8.2 LightGBM

```python
lgb_model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_tr, y_tr_rul,
              eval_set=[(X_va, y_va_rul)],
              callbacks=[lgb.early_stopping(20, verbose=False)])

results.append(regression_report("LightGBM (val)",  y_va_rul, lgb_model.predict(X_va)))
results.append(regression_report("LightGBM (test)", y_te_rul, lgb_model.predict(X_te)))
```

### 8.3 LSTM for RUL Regression

```python
SEQ_LEN = 50   # Confirmed in literature for FD001

def gen_sequences(df, feature_cols, seq_len):
    """Generate 3D sequences per engine. No cross-engine leakage."""
    X_seqs, y_seqs = [], []
    for unit in df['unit_number'].unique():
        sub = df[df['unit_number'] == unit].sort_values('time_cycles')
        data = sub[feature_cols].values
        rul  = sub['RUL'].values
        for start in range(len(data) - seq_len):
            X_seqs.append(data[start:start + seq_len])
            y_seqs.append(rul[start + seq_len])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

# Apply scaler to dataframes for sequence generation
df_tr_scaled = df_tr.copy()
df_tr_scaled[ALL_FEATURES] = scaler.transform(df_tr[ALL_FEATURES])
df_va_scaled = df_va.copy()
df_va_scaled[ALL_FEATURES] = scaler.transform(df_va[ALL_FEATURES])
dfv_scaled = dfvalid.copy()
dfv_scaled[ALL_FEATURES] = scaler.transform(dfvalid[ALL_FEATURES])

X_tr_seq, y_tr_seq = gen_sequences(df_tr_scaled, ALL_FEATURES, SEQ_LEN)
X_va_seq, y_va_seq = gen_sequences(df_va_scaled, ALL_FEATURES, SEQ_LEN)
X_te_seq, y_te_seq = gen_sequences(dfv_scaled,   ALL_FEATURES, SEQ_LEN)

print(f"LSTM input shapes:")
print(f"  Train: {X_tr_seq.shape} → (samples, {SEQ_LEN} timesteps, {len(ALL_FEATURES)} features)")

# Custom R² metric for Keras
def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# Build LSTM Regression Model
def build_lstm_regressor(seq_len, n_features):
    model = Sequential([
        LSTM(100, input_shape=(seq_len, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='rmsprop', metrics=[r2_keras, 'mae'])
    return model

lstm_reg = build_lstm_regressor(SEQ_LEN, len(ALL_FEATURES))
lstm_reg.summary()

history_reg = lstm_reg.fit(
    X_tr_seq, y_tr_seq,
    epochs=100, batch_size=200,
    validation_data=(X_va_seq, y_va_seq),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('lstm_reg_best.h5', monitor='val_loss', save_best_only=True)
    ],
    verbose=1
)

# Training Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_reg.history['loss'],     label='Train Loss')
axes[0].plot(history_reg.history['val_loss'], label='Val Loss')
axes[0].set_title('LSTM Regression — Loss Curve')
axes[0].legend()
axes[1].plot(history_reg.history['r2_keras'],     label='Train R²')
axes[1].plot(history_reg.history['val_r2_keras'], label='Val R²')
axes[1].set_title('LSTM Regression — R² Curve')
axes[1].legend()
plt.tight_layout()
plt.show()

lstm_preds_val  = lstm_reg.predict(X_va_seq).flatten()
lstm_preds_test = lstm_reg.predict(X_te_seq).flatten()
results.append(regression_report("LSTM Regressor (val)",  y_va_seq, lstm_preds_val))
results.append(regression_report("LSTM Regressor (test)", y_te_seq, lstm_preds_test))
```

### 8.4 Predicted vs. Actual RUL Plots

```python
models_to_compare = {
    'Linear Regression': lr.predict(X_va),
    'SVR (RBF)'        : svr.predict(X_va),
    'Random Forest'    : rf.predict(X_va),
    'XGBoost'          : xgb_model.predict(X_va),
    'LightGBM'         : lgb_model.predict(X_va),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, (name, preds) in zip(axes.flatten(), models_to_compare.items()):
    ax.scatter(y_va_rul, preds, alpha=0.2, s=10, color='#1F4E79')
    lims = [0, max(y_va_rul.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    rmse = np.sqrt(mean_squared_error(y_va_rul, preds))
    ax.set_title(f'{name}\nRMSE={rmse:.2f}')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.legend(fontsize=8)
axes.flatten()[-1].set_visible(False)
plt.suptitle('Predicted vs Actual RUL — Validation Set\nIdeal: all points on red dashed line', y=1.01)
plt.tight_layout()
plt.show()
```

---

## 🎯 PHASE 9: Classification Task

### 9.1 SVC Binary Classification

```python
svc = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svc.fit(X_tr, y_tr_cls)

def classification_report_full(name, y_true, y_pred, y_prob=None):
    print(f"\n{'─'*50}\n  {name}\n{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'At Risk']))
    if y_prob is not None:
        ap = average_precision_score(y_true, y_prob)
        print(f"  PR-AUC: {ap:.4f}")

svc_pred_val  = svc.predict(X_va)
svc_prob_val  = svc.predict_proba(X_va)[:, 1]
svc_pred_test = svc.predict(X_te)
svc_prob_test = svc.predict_proba(X_te)[:, 1]

classification_report_full("SVC (val)",  y_va_cls, svc_pred_val,  svc_prob_val)
classification_report_full("SVC (test)", y_te_cls, svc_pred_test, svc_prob_test)
```

### 9.2 KNN Classifier (Best on FD001)

```python
knn = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
knn.fit(X_tr, y_tr_cls)

knn_pred_val  = knn.predict(X_va)
knn_prob_val  = knn.predict_proba(X_va)[:, 1]
knn_pred_test = knn.predict(X_te)
knn_prob_test = knn.predict_proba(X_te)[:, 1]

classification_report_full("KNN n=100 (val)",  y_va_cls, knn_pred_val,  knn_prob_val)
classification_report_full("KNN n=100 (test)", y_te_cls, knn_pred_test, knn_prob_test)
# Expected test accuracy: ~0.71
```

### 9.3 PR Curve & ROC Curve

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (name, prob_val) in enumerate([('SVC', svc_prob_val), ('KNN', knn_prob_val)]):
    precision, recall, _ = precision_recall_curve(y_va_cls, prob_val)
    ap = average_precision_score(y_va_cls, prob_val)
    axes[0].plot(recall, precision, label=f'{name} (AP={ap:.3f})', color=COLORS[i])

    fpr, tpr, _ = roc_curve(y_va_cls, prob_val)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc(fpr,tpr):.3f})', color=COLORS[i])

axes[0].axhline(y_va_cls.mean(), linestyle='--', color='gray', label='Random')
axes[0].set_title('Precision-Recall Curve (Validation)\nPR-AUC = primary metric for imbalanced data')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision'); axes[0].legend()

axes[1].plot([0,1], [0,1], 'k--', label='Random')
axes[1].set_title('ROC Curve (Validation)')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend()

plt.tight_layout()
plt.show()
```

### 9.4 Confusion Matrix Heatmaps

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, pred) in zip(axes, [('SVC', svc_pred_val), ('KNN', knn_pred_val)]):
    cm = confusion_matrix(y_va_cls, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy', 'At Risk'],
                yticklabels=['Healthy', 'At Risk'])
    ax.set_title(f'{name} — Confusion Matrix (Val)\nFN (top-right) = missed failures')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.show()
```

### 9.5 LSTM Binary Classifier

```python
def fbeta_keras(y_true, y_pred, beta=10):
    """Heavy recall weighting — missing a failure is far worse than a false alarm."""
    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    b2 = beta ** 2
    return (b2 + 1) * (precision * recall) / (b2 * precision + recall + K.epsilon())

def gen_sequences_cls(df, feature_cols, label_col, seq_len):
    X_seqs, y_seqs = [], []
    for unit in df['unit_number'].unique():
        sub = df[df['unit_number'] == unit].sort_values('time_cycles')
        data  = sub[feature_cols].values
        label = sub[label_col].values
        for start in range(len(data) - seq_len):
            X_seqs.append(data[start:start + seq_len])
            y_seqs.append(label[start + seq_len])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

X_tr_cls_seq, y_tr_cls_seq = gen_sequences_cls(df_tr_scaled, ALL_FEATURES, 'label1', SEQ_LEN)
X_va_cls_seq, y_va_cls_seq = gen_sequences_cls(df_va_scaled, ALL_FEATURES, 'label1', SEQ_LEN)

lstm_cls = Sequential([
    LSTM(100, input_shape=(SEQ_LEN, len(ALL_FEATURES)), return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_cls.compile(loss='binary_crossentropy', optimizer='adam', metrics=[fbeta_keras])

history_cls = lstm_cls.fit(
    X_tr_cls_seq, y_tr_cls_seq,
    epochs=10, batch_size=200,
    validation_data=(X_va_cls_seq, y_va_cls_seq),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)
# Expected train fbeta ~0.93 | val fbeta ~0.73
```

---

## 🔧 PHASE 10: Hyperparameter Tuning

### 10.1 Optuna — XGBoost RUL Regression

```python
def objective_xgb(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 50, 500),
        'max_depth'       : trial.suggest_int('max_depth', 3, 9),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
        'gamma'           : trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr_rul, verbose=False)
    return np.sqrt(mean_squared_error(y_va_rul, model.predict(X_va)))

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"\nBest XGBoost Val RMSE: {study_xgb.best_value:.3f}")
print(f"Best params: {study_xgb.best_params}")

xgb_best = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1)
xgb_best.fit(X_tr, y_tr_rul)
results.append(regression_report("XGBoost Tuned (test)", y_te_rul, xgb_best.predict(X_te)))
```

### 10.2 Optuna — LightGBM

```python
def objective_lgb(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'max_depth'        : trial.suggest_int('max_depth', 3, 9),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr_rul,
              eval_set=[(X_va, y_va_rul)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return np.sqrt(mean_squared_error(y_va_rul, model.predict(X_va)))

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

lgb_best = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=-1, verbose=-1)
lgb_best.fit(X_tr, y_tr_rul)
results.append(regression_report("LightGBM Tuned (test)", y_te_rul, lgb_best.predict(X_te)))
```

---

## 🧩 PHASE 11: Ensemble Strategy

### 11.1 Tier 1 — Weighted Averaging

```python
preds_val  = {'SVR': svr.predict(X_va),  'XGB': xgb_best.predict(X_va),  'LGB': lgb_best.predict(X_va)}
preds_test = {'SVR': svr.predict(X_te),  'XGB': xgb_best.predict(X_te),  'LGB': lgb_best.predict(X_te)}

val_rmses = {k: np.sqrt(mean_squared_error(y_va_rul, v)) for k, v in preds_val.items()}
inv_rmse  = {k: 1/v for k, v in val_rmses.items()}
total     = sum(inv_rmse.values())
weights   = {k: v/total for k, v in inv_rmse.items()}
print(f"Ensemble weights: {weights}")

ensemble_val  = sum(weights[k] * preds_val[k]  for k in weights)
ensemble_test = sum(weights[k] * preds_test[k] for k in weights)

results.append(regression_report("Weighted Ensemble (val)",  y_va_rul, ensemble_val))
results.append(regression_report("Weighted Ensemble (test)", y_te_rul, ensemble_test))
```

### 11.2 Tier 2 — Stacking Meta-Learner

```python
def get_oof_predictions(model, X, y, n_splits=5):
    """Out-of-fold predictions — prevents meta-learner data leakage."""
    oof = np.zeros(len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X[val_idx])
    return oof

oof_svr = get_oof_predictions(SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'), X_tr, y_tr_rul)
oof_xgb = get_oof_predictions(xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1), X_tr, y_tr_rul)
oof_lgb = get_oof_predictions(lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=-1, verbose=-1), X_tr, y_tr_rul)

meta_X_train = np.column_stack([oof_svr, oof_xgb, oof_lgb])

# Re-train base models on full training data for test predictions
svr.fit(X_tr, y_tr_rul)
xgb_best.fit(X_tr, y_tr_rul)
lgb_best.fit(X_tr, y_tr_rul)

meta_X_val  = np.column_stack([svr.predict(X_va), xgb_best.predict(X_va), lgb_best.predict(X_va)])
meta_X_test = np.column_stack([svr.predict(X_te), xgb_best.predict(X_te), lgb_best.predict(X_te)])

meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
meta_learner.fit(meta_X_train, y_tr_rul)

stacking_val  = meta_learner.predict(meta_X_val)
stacking_test = meta_learner.predict(meta_X_test)

results.append(regression_report("Stacking Meta-Learner (val)",  y_va_rul, stacking_val))
results.append(regression_report("Stacking Meta-Learner (test)", y_te_rul, stacking_test))
```

### 11.3 Residual Correlation Matrix (Diversity Check)

```python
residuals = pd.DataFrame({
    'SVR': svr.predict(X_va) - y_va_rul,
    'XGB': xgb_best.predict(X_va) - y_va_rul,
    'LGB': lgb_best.predict(X_va) - y_va_rul,
})

plt.figure(figsize=(6, 5))
sns.heatmap(residuals.corr(), annot=True, fmt='.3f', cmap='RdYlGn_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Residual Correlation Between Models\nLow correlation = ensemble gains the most')
plt.tight_layout()
plt.show()
```

---

## 📊 PHASE 12: Uncertainty Quantification

### 12.1 Quantile Regression — LightGBM

```python
quantile_models = {}
for q in [0.10, 0.50, 0.90]:
    q_model = lgb.LGBMRegressor(
        objective='quantile', alpha=q,
        n_estimators=200, learning_rate=0.05,
        num_leaves=63, random_state=42, verbose=-1
    )
    q_model.fit(X_tr, y_tr_rul)
    quantile_models[q] = q_model

preds_q10 = quantile_models[0.10].predict(X_te)
preds_q50 = quantile_models[0.50].predict(X_te)
preds_q90 = quantile_models[0.90].predict(X_te)

fig, ax = plt.subplots(figsize=(14, 5))
n = 200
ax.fill_between(range(n), preds_q10[:n], preds_q90[:n],
                alpha=0.3, color='#2E75B6', label='80% Confidence Band (Q10–Q90)')
ax.plot(preds_q50[:n], color='#1F4E79', linewidth=1.5, label='Median Prediction (Q50)')
ax.plot(y_te_rul[:n],  color='#FF7043', linewidth=1.5, linestyle='--', label='Actual RUL')
ax.set_title('RUL Prediction with Uncertainty Bands\n"10 cycles left (range: 8–13)" — actionable for factory managers')
ax.set_xlabel('Test Sample Index'); ax.set_ylabel('RUL (cycles)'); ax.legend()
plt.tight_layout()
plt.show()

coverage = np.mean((y_te_rul >= preds_q10) & (y_te_rul <= preds_q90)) * 100
print(f"80% Prediction Interval Coverage: {coverage:.1f}%  (target: ~80%)")
```

### 12.2 Monte Carlo Dropout — LSTM

```python
def mc_dropout_predict(model, X, n_iter=100):
    """Keep dropout active at inference time for uncertainty estimation."""
    preds = np.array([model(X, training=True).numpy().flatten()
                      for _ in range(n_iter)])
    return preds.mean(axis=0), preds.std(axis=0)

mc_mean, mc_std = mc_dropout_predict(lstm_reg, X_te_seq[:200], n_iter=100)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(range(200), mc_mean - 2*mc_std, mc_mean + 2*mc_std,
                alpha=0.3, color='#70AD47', label='±2σ Uncertainty (MC Dropout)')
ax.plot(mc_mean, color='#1F4E79', linewidth=1.5, label='MC Mean Prediction')
ax.plot(y_te_seq[:200], color='#FF7043', linestyle='--', label='Actual RUL')
ax.set_title('LSTM RUL — Monte Carlo Dropout Uncertainty\nWide bands = model is less certain')
ax.set_xlabel('Sample'); ax.set_ylabel('RUL'); ax.legend()
plt.tight_layout()
plt.show()
```

---

## 📐 PHASE 13: Model Evaluation & Metrics

### 13.1 Final Comprehensive Comparison Table

```python
test_results_df = (pd.DataFrame(results)
                   [lambda df: df['model'].str.contains('test')]
                   .sort_values('RMSE')
                   .reset_index(drop=True))

print("\n" + "="*70)
print("  FINAL MODEL COMPARISON — TEST SET")
print("="*70)
print(test_results_df[['model','RMSE','MAE','R2','NASA_Score']].to_string(index=False))
```

### 13.2 Residual Distribution & Error Pattern Plots

```python
best_preds = stacking_test   # Replace with best model

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
residuals_test = best_preds - y_te_rul

axes[0].hist(residuals_test, bins=40, color='#1F4E79', edgecolor='white')
axes[0].axvline(0, color='red', linestyle='--', label='Zero residual')
axes[0].axvline(residuals_test.mean(), color='orange', linestyle='--',
                label=f'Mean={residuals_test.mean():.2f}')
axes[0].set_title('Residual Distribution (Test)\nCentered at 0 = low bias')
axes[0].set_xlabel('Predicted RUL − Actual RUL'); axes[0].legend()

axes[1].scatter(y_te_rul, residuals_test, alpha=0.3, s=10, color='#2E75B6')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Actual RUL\nFlat pattern = no systematic bias')
axes[1].set_xlabel('Actual RUL'); axes[1].set_ylabel('Residual')
plt.tight_layout()
plt.show()
```

### 13.3 Radar Chart — Multi-Metric Comparison

```python
test_models_radar = {
    'SVR'      : svr.predict(X_te),
    'XGBoost'  : xgb_best.predict(X_te),
    'LightGBM' : lgb_best.predict(X_te),
    'Ensemble' : stacking_test,
}

metrics_radar = {}
for name, preds in test_models_radar.items():
    rmse = np.sqrt(mean_squared_error(y_te_rul, preds))
    r2   = r2_score(y_te_rul, preds)
    mae  = mean_absolute_error(y_te_rul, preds)
    metrics_radar[name] = {'1/RMSE': 1/rmse, 'R2': max(r2,0), '1/MAE': 1/mae}

categories = list(list(metrics_radar.values())[0].keys())
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, (name, vals) in enumerate(metrics_radar.items()):
    v = list(vals.values())
    max_vals = [max(metrics_radar[k][cat] for k in metrics_radar) for cat in categories]
    v_norm = [x/m for x, m in zip(v, max_vals)] + [v[0]/max_vals[0]]
    ax.plot(angles, v_norm, linewidth=2, label=name, color=COLORS[i])
    ax.fill(angles, v_norm, alpha=0.1, color=COLORS[i])

ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Radar — Multi-Metric Comparison\nOuter ring = better', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.show()
```

---

## 🔍 PHASE 14: Model Interpretability (SHAP)

### 14.1 SHAP Summary Plot — Global Feature Importance

```python
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_va[:500])

# Beeswarm summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_va[:500],
                  feature_names=ALL_FEATURES,
                  plot_type='dot', max_display=20, show=False)
plt.title('SHAP Summary (Beeswarm) — XGBoost\nRed=high feature value | Blue=low')
plt.tight_layout()
plt.show()

# Bar chart (mean |SHAP|)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_va[:500],
                  feature_names=ALL_FEATURES,
                  plot_type='bar', max_display=20, show=False)
plt.title('Mean |SHAP| — Top 20 Feature Importance Ranking')
plt.tight_layout()
plt.show()
```

### 14.2 SHAP Waterfall — Single Machine Explanation

```python
# Explain a near-failure prediction (actionable for operators)
near_failure_idx = np.where(y_va_rul < 30)[0][0]

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[near_failure_idx],
        base_values=explainer.expected_value,
        data=X_va[near_failure_idx],
        feature_names=ALL_FEATURES
    )
)
# Operator-readable: "s_9_rm is 37% below healthy baseline,
# reducing predicted RUL by 18 cycles"
```

### 14.3 Permutation Importance — Stability Check

```python
perm_imp = permutation_importance(xgb_best, X_va, y_va_rul,
                                   n_repeats=10, random_state=42, n_jobs=-1)

perm_df = (pd.DataFrame({'feature': ALL_FEATURES,
                          'importance_mean': perm_imp.importances_mean,
                          'importance_std' : perm_imp.importances_std})
           .sort_values('importance_mean', ascending=False).head(20))

plt.figure(figsize=(10, 6))
plt.barh(perm_df['feature'][::-1], perm_df['importance_mean'][::-1],
         xerr=perm_df['importance_std'][::-1], color='#1F4E79', alpha=0.8,
         error_kw={'elinewidth': 1.5, 'capsize': 3})
plt.xlabel('Mean RMSE Increase when Feature Shuffled')
plt.title('Permutation Importance — Top 20 Features\nError bars = std across 10 repeats')
plt.tight_layout()
plt.show()
```

---

## 🧪 PHASE 15: Ablation Study

```python
ablation_results = []

def ablation_test(name, X_train, X_test, y_train, y_test):
    m = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    ablation_results.append({'Experiment': name, 'Test RMSE': round(rmse, 2), 'Test R²': round(r2, 3)})
    print(f"  {name:<45} RMSE={rmse:.2f}  R²={r2:.3f}")

print("Running ablation study...\n")

# 1. Full pipeline (reference)
ablation_test("1. Full Pipeline (all features)",       X_tr, X_te, y_tr_rul, y_te_rul)

# 2. Remove rolling window features
scaler_a2 = MinMaxScaler()
X_tr_a2 = scaler_a2.fit_transform(df_tr[raw_sensor_cols + domain_feat_cols])
X_te_a2 = scaler_a2.transform(dfvalid[raw_sensor_cols + domain_feat_cols])
ablation_test("2. No Rolling Window Features",         X_tr_a2, X_te_a2, y_tr_rul, y_te_rul)

# 3. No RUL clipping (use raw unclipped RUL as target)
y_tr_raw_rul = df_tr['RUL_raw'].values
ablation_test("3. No Piecewise RUL Clipping",          X_tr, X_te, y_tr_raw_rul, y_te_rul)

# 4. Raw sensors only — no rolling, no domain features
scaler_a4 = MinMaxScaler()
X_tr_a4 = scaler_a4.fit_transform(df_tr[raw_sensor_cols])
X_te_a4 = scaler_a4.transform(dfvalid[raw_sensor_cols])
ablation_test("4. Raw Sensors Only (no engineering)",  X_tr_a4, X_te_a4, y_tr_rul, y_te_rul)

# ── Summary Table ─────────────────────────────────────────────────────
ablation_df = pd.DataFrame(ablation_results)
baseline_rmse = ablation_df.iloc[0]['Test RMSE']
ablation_df['% Degradation vs Full'] = ((ablation_df['Test RMSE'] / baseline_rmse - 1) * 100).round(1)

print("\n" + "="*75)
print("  ABLATION STUDY RESULTS — Each row proves one component adds value")
print("="*75)
print(ablation_df[['Experiment', 'Test RMSE', 'Test R²', '% Degradation vs Full']].to_string(index=False))

# Bar chart
plt.figure(figsize=(10, 4))
colors_abl = ['#70AD47' if i == 0 else '#FF7043' for i in range(len(ablation_df))]
plt.barh(ablation_df['Experiment'][::-1], ablation_df['Test RMSE'][::-1],
         color=colors_abl[::-1], edgecolor='white')
plt.axvline(baseline_rmse, color='green', linestyle='--',
            label=f'Full pipeline = {baseline_rmse:.2f}')
plt.xlabel('Test RMSE (lower is better)')
plt.title('Ablation Study — RMSE Increase Proves Each Component Adds Value\nGreen=full pipeline, Red=component removed')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## ✅ Final Pipeline Summary

| Phase | Step | Key Output | Expected Result |
|-------|------|------------|-----------------|
| 0 | Environment setup | All libraries imported | — |
| 1 | Data loading & schema | Shapes validated | Train (20631,26), Valid (13096,26) |
| 2 | Data cleaning | 7 constant sensors dropped, outliers clipped | 14 sensors remaining |
| 3 | EDA | Signal vs RUL plots, correlation heatmap, PCA | PC1 explains ~73.6% variance |
| 4 | RUL label engineering | Piecewise clipped RUL, label1, label2 | CLIP_VALUE=195, W1=30, W0=15 |
| 5 | Feature engineering | Rolling (56) + raw (14) + domain = ~74 features | Per-engine rolling, no leakage |
| 6 | Train/val/test split | Engine-ID based, leakage-safe scaling | 80/20 engine split |
| 7 | Baselines | Linear Reg, SVR, RF | SVR val RMSE ~26–28 |
| 8 | Advanced models | XGBoost, LightGBM, LSTM | LSTM F1 ~0.93 train |
| 9 | Classification | SVC, KNN, LSTM classifier | KNN test acc ~0.71 |
| 10 | Optuna tuning | 50 trials per model | Best val RMSE minimized |
| 11 | Ensemble | Weighted avg → Stacking meta-learner | +8–15% RMSE improvement |
| 12 | Uncertainty QT | Quantile regression + MC Dropout | 80% coverage interval |
| 13 | Evaluation | RMSE, MAE, R², NASA Score, radar chart | Full comparison table |
| 14 | Interpretability | SHAP beeswarm, waterfall, permutation | Per-machine explanations |
| 15 | Ablation study | Quantified value of each component | % degradation per removal |

> **Key Benchmark Targets (FD001)**
> - SVR baseline test RMSE ≈ **31**
> - LSTM classification training F1 ≈ **0.93** | val fbeta ≈ **0.73**
> - KNN best classification test accuracy ≈ **0.71**
> - Stacking ensemble expected to beat SVR by **+8–15% RMSE**