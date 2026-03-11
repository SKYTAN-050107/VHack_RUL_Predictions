# Notebook Structure for Experiment Pipeline

## Project Notebook Overview

This project uses three main notebooks:

- **EDA Notebook**
- **Feature Engineering Notebook**
- **Model Evaluation Notebook**

Each notebook has a specific responsibility to keep the pipeline organized and reproducible.

---

## 1. EDA Notebook

**Notebook name:** `01_EDA.ipynb`

### Purpose
The EDA notebook is used only for understanding the dataset and identifying preprocessing strategies. No experiment datasets should be generated here.

### Contents

#### 1. Dataset Overview
Load the dataset and inspect:

```python
df.head()
df.info()
df.describe()
```

**Tasks:**
- Understand sensor distributions
- Check dataset size
- Confirm engine lifecycle patterns

#### 2. Engine Lifecycle Analysis
Visualize degradation trends.

**Example plots:**
- Sensor values vs `time_cycles`
- Sensor values vs `RUL`

#### 3. Outlier Analysis
Analyze sensor distributions.

**Tasks:**
- Identify extreme values
- Check skewness
- Observe percentile ranges

**Example tools:**
- Boxplots
- Histograms
- Percentile analysis

> Output from this step informs which outlier methods will be tested.

#### 4. Feature Relevance Analysis
Remove useless features.

**Tasks:**
- Remove zero variance features
- Remove features with no correlation with RUL

**Example methods:**
- Variance threshold
- Correlation analysis

**Output:** `cleaned_dataset.csv`

This cleaned dataset becomes the starting point for experiments.

---

## 2. Feature Engineering Notebook

**Notebook name:** `02_feature_engineering_experiments.ipynb`

This is the main notebook for experiment dataset generation.

### Structure

#### 1. Load Cleaned Dataset
Load the dataset produced by EDA:

```python
df = pd.read_csv("data/cleaned_dataset.csv")
```

Ensure correct ordering:

```python
df = df.sort_values(["unit_number", "time_cycles"])
```

#### 2. Define Experiment Methods
Define lists of methods for factorial experiments.

**Outlier methods:**

```python
outlier_methods = [
    "raw",
    "clip_1_99",
    "fixed_cap",
    "winsorize",
    "robust_scaling",
    "minmax",
    "cap_minmax",
    "clip_minmax"
]
```

**Feature engineering methods:**

```python
feature_methods = [
    "none",
    "lag",
    "lag_rolling",
    "lag_rolling_diff",
    "lag_rolling_diff_ewma"
]
```

**Dimensionality reduction:**

```python
pca_methods = [
    "none",
    "pca95",
    "pca90"
]
```

#### 3. Outlier Handling Functions
Implement outlier handling functions here.

**Examples:**
- `apply_clipping()`
- `apply_winsorizing()`
- `apply_capping()`
- `apply_scaling()`

Each function should:
- **Input:** dataframe
- **Output:** processed dataframe

#### 4. Feature Engineering Functions
Define functions for:

- **Lag Features** — `create_lag_features(df)`
- **Rolling Features** — `create_rolling_features(df)`
- **Difference Features** — `create_diff_features(df)`
- **EWMA Features** — `create_ewma_features(df)`

#### 5. PCA Function
Define PCA pipeline:

```python
apply_pca(X, variance=0.95)
```

> **Important:** PCA must be applied after feature engineering and scaling.

#### 6. Experiment Dataset Generation Loop
This section generates all experiment datasets automatically:

```python
for outlier in outlier_methods:

    df_outlier = apply_outlier(df, outlier)

    for feature_method in feature_methods:

        df_features = apply_feature_engineering(df_outlier, feature_method)

        for pca_method in pca_methods:

            dataset = apply_pca_if_needed(df_features, pca_method)

            save_dataset(dataset)
```

#### 7. Dataset Saving
Each dataset is saved using a standard naming convention:

```
EXP_A1_B1_C1.csv
EXP_A3_B2_C1.csv
EXP_A8_B5_C2.csv
```

```python
dataset.to_csv("experiments/EXP_A1_B1_C1.csv")
```

### Output of Feature Engineering Notebook

```
experiments/
├── EXP_A1_B1_C1.csv
├── EXP_A1_B2_C1.csv
├── EXP_A1_B3_C1.csv
├── ...
└── EXP_A8_B5_C3.csv
```

**Total possible datasets: 120**

These datasets will be used in the model evaluation notebook.

---

## 3. Model Evaluation Notebook

**Notebook name:** `03_model_evaluation.ipynb`

This notebook only loads experiment datasets and trains models.

### Structure

#### 1. Load Experiment Dataset

```python
df = pd.read_csv("experiments/EXP_A3_B4_C1.csv")

X = df.drop(columns=["RUL"])
y = df["RUL"]
```

#### 2. Train Models
Train the prepared models:

- Random Forest
- LightGBM
- XGBoost
- SVM
- Neural Networks

```python
model.fit(X_train, y_train)
```

#### 3. Model Evaluation
Evaluate performance using the following metrics:

- **RMSE**
- **MAE**
- **R²**

Store results in a table:

```python
results.append({
    "experiment": exp_name,
    "model": model_name,
    "rmse": rmse
})
```

#### 4. Experiment Comparison
Compare all experiments using a summary results table:

| Experiment   | Outlier    | Features            | PCA   | Model    | RMSE |
|-------------|------------|---------------------|-------|----------|------|
| EXP_A1_B1_C1 | Raw        | None                | No    | LightGBM | 22   |
| EXP_A3_B4_C1 | Clip       | Lag+Rolling+Diff    | No    | LightGBM | 16   |
| EXP_A7_B5_C2 | Cap+MinMax | Full Features       | PCA95 | SVM      | 18   |

This identifies the best preprocessing pipeline.

---

## Recommended Notebook Workflow

```
01_EDA.ipynb
      ↓
cleaned_dataset.csv
      ↓
02_feature_engineering_experiments.ipynb
      ↓
120 prepared datasets
      ↓
03_model_evaluation.ipynb
      ↓
best preprocessing + best model
```
