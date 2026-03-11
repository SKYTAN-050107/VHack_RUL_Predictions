# RUL Prediction – Data Preparation Experiment Plan

---

## 1. Objective

This document defines the data preparation pipelines for factorial experiments designed to evaluate the impact of:

- Outlier handling methods
- Feature engineering strategies
- Optional dimensionality reduction (PCA)

The goal is to identify the best data preparation combination before model training. Model training and hyperparameter tuning will be conducted separately.

---

## 2. Starting Dataset

The experiments start from a cleaned dataset with the following preprocessing already completed:

- Removed zero-variance features
- Removed features with no correlation to RUL
- Dataset contains only useful sensor and operational features

**Typical columns include:**

| Column | Description |
|--------|-------------|
| `unit_number` | Engine identifier |
| `time_cycles` | Cycle counter |
| `sensor_1 ... sensor_n` | Sensor readings |
| `RUL` | Remaining Useful Life (target) |

> **Important:** Time-series ordering must be preserved.

---

## 3. Experiment Factors

The experiment follows a factorial design based on three factors.

### Factor A — Outlier Handling

Eight outlier handling strategies will be tested.

| ID | Method | Description |
|----|--------|-------------|
| A1 | Raw | No outlier handling |
| A2 | Clipping 1–99 | Clip values outside the 1st and 99th percentile |
| A3 | Fixed Capping | Cap sensor values to fixed thresholds |
| A4 | Winsorizing | Replace extreme values with nearest percentile |
| A5 | Robust Scaling | Median and IQR scaling |
| A6 | MinMax Scaling | Scale features to 0–1 |
| A7 | Capping + MinMax | Cap values then apply MinMax scaling |
| A8 | Clipping + MinMax | Clip values then apply MinMax scaling |

---

### Factor B — Feature Engineering

Five feature engineering configurations will be evaluated.

| ID | Feature Type | Description |
|----|-------------|-------------|
| B1 | None | Raw sensor features only |
| B2 | Lag | Lag features (t-1 to t-5) |
| B3 | Lag + Rolling | Lag features + rolling mean/std |
| B4 | Lag + Rolling + Diff | Add derivative features |
| B5 | Lag + Rolling + Diff + EWMA | Add exponentially weighted moving average |

#### Feature Engineering Definitions

**Lag Features**
Capture sensor history from previous cycles.

Example features: `s_3_lag1`, `s_3_lag2`, `s_3_lag5`

```python
df.groupby("unit_number")[sensor].shift(k)
```

**Rolling Window Features**
Capture short-term trends using rolling statistics (mean, std).

Example window sizes: 3, 5, 10 cycles

Example features: `s_3_roll_mean_5`, `s_3_roll_std_5`

**Difference Features**
Measure rate of change between cycles.

```
s_3_diff = s_3(t) − s_3(t−1)
```

Purpose: Detect rapid degradation or sudden changes.

**EWMA Features**
Exponentially weighted moving average that emphasizes recent values.

Example feature: `s_3_ewma_5`

Purpose: Detect recent sensor trend changes faster than rolling mean.

---

### Factor C — Dimensionality Reduction

Three dimensionality reduction options will be tested.

| ID | Method | Description |
|----|--------|-------------|
| C1 | None | No dimensionality reduction |
| C2 | PCA 95% | PCA retaining 95% variance |
| C3 | PCA 90% | PCA retaining 90% variance |

> **Note:** PCA is applied after scaling and feature engineering.

---

## 4. Total Experiment Design

Factorial experiment size:

| Factor | Count |
|--------|-------|
| Outlier methods | 8 |
| Feature engineering methods | 5 |
| Dimensionality reduction methods | 3 |

**Total experiments: 8 × 5 × 3 = 120**

---

## 5. Data Preparation Pipeline

Each experiment follows the same pipeline structure:

```
Cleaned Dataset
        ↓
Outlier Handling
        ↓
Feature Engineering
        ↓
Scaling (if required)
        ↓
Dimensionality Reduction (optional)
        ↓
Final Training Dataset
```

---

## 6. Detailed Data Processing Steps

### Step 1 — Load Cleaned Dataset
Ensure data is sorted:
- Sort by `unit_number`
- Sort by `time_cycles`

### Step 2 — Apply Outlier Handling
Apply one of the Factor A methods.

**Example — Percentile clipping:**
```
clip lower = 1st percentile
clip upper = 99th percentile
```

**Example — Winsorizing:**
```
Replace extreme values with nearest percentile
```

### Step 3 — Apply Feature Engineering
Apply one Factor B configuration.

**Important rules:**
- All features must respect engine boundaries
- Use `groupby(unit_number)`
- Avoid future leakage

**Example implementations:**

```python
df.groupby("unit_number")[sensor].shift()
df.groupby("unit_number")[sensor].rolling()
```

After feature creation, drop rows with NaN created by lag/rolling operations.

### Step 4 — Feature Scaling
Scaling is applied after feature engineering, when required.

**Available scalers:**
- `MinMaxScaler`
- `RobustScaler`
- `StandardScaler`

### Step 5 — Dimensionality Reduction (Optional)
Apply PCA if the experiment requires it.

```python
PCA(n_components=0.95)
```

Steps:
1. Fit PCA on training features
2. Transform dataset

---

## 7. Experiment Dataset Naming Convention

Each prepared dataset follows this naming format:

```
EXP_[OutlierID]_[FeatureID]_[PCAID]
```

**Examples:**

| Dataset Name | Meaning |
|-------------|---------|
| `EXP_A1_B1_C1` | Raw · No features · No PCA |
| `EXP_A3_B2_C1` | Fixed Capping · Lag · No PCA |
| `EXP_A7_B4_C2` | Capping + MinMax · Lag+Rolling+Diff · PCA 95% |

---

## 8. Experiment Matrix Example

| Experiment | Outlier | Feature Engineering | PCA |
|-----------|---------|---------------------|-----|
| EXP_A1_B1_C1 | Raw | None | None |
| EXP_A1_B2_C1 | Raw | Lag | None |
| EXP_A2_B3_C1 | Clipping | Lag + Rolling | None |
| EXP_A4_B4_C2 | Winsorizing | Lag + Rolling + Diff | PCA 95% |
| EXP_A8_B5_C3 | Clip + MinMax | Full features | PCA 90% |

**Total combinations: 120 experiment datasets**

---

## 9. Important Implementation Rules

### Prevent Data Leakage
- All time-based features must use past values only
- Never use future cycles

### Group by Engine
Feature engineering must respect engine boundaries. Always use:

```python
groupby(unit_number)
```

### Remove Generated NaN
Lag and rolling operations create missing values. These rows must be removed before modeling.

### Preserve Target
The `RUL` column must remain unchanged throughout all processing steps.

---

## 10. Recommended Experiment Order

To reduce computational cost, experiments can be run in stages.

**Stage 1 — Outlier Comparison**
Test A1–A8 with B1 to identify the best outlier strategy baseline.

**Stage 2 — Feature Engineering**
Apply feature engineering to the best outlier methods.
Example: Top 3 outlier methods × B2–B5

**Stage 3 — Dimensionality Reduction**
Apply PCA only to the best pipelines.
Example: Best 10 pipelines × PCA variants

---

## 11. Expected Feature Growth

| Stage | Approx. Features |
|-------|-----------------|
| Raw sensors | ~20 |
| Lag features | ~100 |
| Lag + Rolling | ~200 |
| Full features | 300+ |

> PCA may reduce this to **20–40 components**.

---

## 12. Output of Each Experiment

Each experiment produces a dataset containing:

- `X_features` — all engineered features
- `y_target` — RUL (unchanged)

The dataset will be passed directly to the model training pipeline.

---

## 13. Folder Structure

Recommended project structure:

```
data/
├── raw/
└── cleaned/

experiments/
├── exp_A1_B1_C1/
├── exp_A1_B2_C1/
└── exp_A3_B4_C2/

scripts/
├── outlier_methods.py
├── feature_engineering.py
└── pca_pipeline.py
```

---

## 14. Reproducibility

To ensure reproducibility:

- Fix random seed
- Store experiment parameters
- Log preprocessing configuration
