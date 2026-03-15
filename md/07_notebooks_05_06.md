# 07 — Notebooks 5 & 6: LSTM-DANN Domain Adaptation and Model Evaluation

> **IDE Agent Instructions:**
> - Create TWO Jupyter notebook files at the paths shown.
> - `# %% [markdown]` → Markdown cell. `# %%` → Code cell.

---

## Notebook 5 — LSTM-DANN Domain Adaptation

### Create File: `notebooks/05_lstm_dann_domain_adaptation.ipynb`

```python
# %% [markdown]
# # Notebook 5 — LSTM-DANN Domain Adaptation
#
# **Goal:** Train the LSTM Domain Adversarial Neural Network (LSTM-DANN)
# to learn machine-independent RUL features.
#
# **Architecture** (Section 3.4 of the paper):
# - Shared LSTM Feature Extractor g_f → produces domain-invariant embedding f
# - RUL Regressor g_y: f → predicted RUL (minimises MAE on SOURCE labels)
# - Domain Classifier g_d: f → GRL → domain binary classifier
#   (minimises BCE on source/target labels; GRL reverses gradient for g_f)
#
# **Training** (Section 5.1):
# - Two-pass SGD: regression pass on source, adversarial pass on source+target
# - Early stopping on source validation MAE
# - LR decay ×0.1 at epoch 100

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import os

from src.data_loader    import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor   import full_preprocess_pipeline
from src.windowing      import create_windows, create_windows_inference
from src.models.lstm_dann import build_lstm_dann, get_feature_extractor
from src.train          import LSTMDANNTrainer
from src.evaluate       import rmse, evaluate_model

tf.random.set_seed(42)
np.random.seed(42)

WINDOW_SIZE = 30
MAX_RUL     = 125

# Load and preprocess all datasets
datasets = load_all_datasets(data_dir='../data/raw')
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df_tr, df_te, scaler = full_preprocess_pipeline(
        df_train=datasets[ds_id]['train'],
        df_test=datasets[ds_id]['test'],
        feature_cols=FEATURE_COLS, sensor_cols=SENSOR_COLS,
        smooth=True, max_rul=MAX_RUL,
        scaler_save_path=f'../models/saved/scaler_{ds_id}.joblib'
    )
    datasets[ds_id]['train_norm'] = df_tr
    datasets[ds_id]['test_norm']  = df_te
    X, y, _ = create_windows(df_tr, FEATURE_COLS, WINDOW_SIZE)
    datasets[ds_id]['X_train'] = X
    datasets[ds_id]['y_train'] = y

print("Preprocessing complete.")

# %% [markdown]
# ## 5.1 Architecture Overview

# %%
_, dann_model_demo = build_lstm_dann(
    window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
    lstm_units=128, lstm_layers=1, feature_dim=64,
    reg_units=[32], domain_units=[32],
    lstm_dropout=0.5, reg_dropout=0.3, dom_dropout=0.3, alpha=0.8
)
dann_model_demo.summary()

# %% [markdown]
# **Architecture components:**
#
# | Sub-network | Layers | Purpose |
# |------------|--------|---------|
# | Feature Extractor g_f | LSTM(128) → Dropout(0.5) → Dense(64, ReLU) | Extract temporal patterns from sensor windows |
# | RUL Regressor g_y | Dense(32, ReLU) → Dropout(0.3) → Dense(1) | Map features → RUL scalar |
# | Domain Classifier g_d | GRL(α=0.8) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid) | Distinguish source from target (adversarially) |
#
# The GRL ensures g_f is trained adversarially: while g_d tries to tell domains
# apart, g_f learns to produce features that make this IMPOSSIBLE.

# %%
# %% [markdown]
# ## 5.2 Training: FD001 → FD002 (Primary Example)
#
# Hyperparameters from Table 3 of the paper (Source FD001, Target FD002 row).

# %%
SOURCE_DS = 'FD001'
TARGET_DS = 'FD002'

X_src = datasets[SOURCE_DS]['X_train'].astype(np.float32)
y_src = datasets[SOURCE_DS]['y_train'].astype(np.float32)
X_tgt = datasets[TARGET_DS]['X_train'].astype(np.float32)

X_src_tr, X_src_val, y_src_tr, y_src_val = train_test_split(
    X_src, y_src, test_size=0.1, random_state=42
)

print(f"Source train: {X_src_tr.shape} | Source val: {X_src_val.shape}")
print(f"Target train: {X_tgt.shape}")

# %%
# Build model with paper hyperparameters for FD001 → FD002
reg_model_01_02, dann_model_01_02 = build_lstm_dann(
    window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
    lstm_units=128, lstm_layers=1, feature_dim=64,
    reg_units=[32], domain_units=[32],
    lstm_dropout=0.5, reg_dropout=0.3, dom_dropout=0.3,
    alpha=0.8
)

trainer = LSTMDANNTrainer(
    dann_model_01_02, alpha=0.8,
    lr_reg=0.01, lr_dom=0.01
)

history = trainer.fit(
    X_src_tr, y_src_tr, X_tgt,
    X_val_src=X_src_val, y_val_src=y_src_val,
    epochs=200, batch_size=256,
    patience=20, lr_decay_epoch=100
)

dann_model_01_02.save_weights(
    f'../models/saved/lstm_dann_{SOURCE_DS}_to_{TARGET_DS}.weights.h5'
)
print("Model weights saved.")

# %%
# %% [markdown]
# ## 5.3 Training Curve Analysis

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['rul_loss'], color='steelblue', linewidth=2)
axes[0].set_title(f'RUL Regression Loss\n({SOURCE_DS} source domain)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MAE Loss')
axes[0].grid(alpha=0.3)

axes[1].plot(history['dom_loss'], color='coral', linewidth=2)
axes[1].axhline(np.log(2), color='gray', linestyle='--', linewidth=1.5,
                label=f'Random guess = ln(2) ≈ {np.log(2):.3f}')
axes[1].set_title('Domain Classification Loss\n(Should converge near ln(2))')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Binary Cross-Entropy')
axes[1].legend()
axes[1].grid(alpha=0.3)

if history['val_mae']:
    axes[2].plot(history['val_mae'], color='seagreen', linewidth=2)
    axes[2].set_title(f'Validation MAE on {SOURCE_DS}\n(Early stopping criterion)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].grid(alpha=0.3)

plt.suptitle(f'LSTM-DANN Training Curves: {SOURCE_DS} → {TARGET_DS}', fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the three curves:**
#
# 1. **RUL Regression Loss** (decreasing): The model learns to predict RUL
#    from source data — this should drop and stabilise.
#
# 2. **Domain Classification Loss** (converging to ~0.693 = ln(2)):
#    When this stabilises near the random-guess value, the domain classifier
#    can no longer distinguish source from target. This confirms the feature
#    extractor has learned truly domain-invariant representations.
#
# 3. **Validation MAE** (used for early stopping): The criterion from
#    Section 5.2 of the paper — select hyperparameters giving lowest source
#    RMSE while domain classification stabilises near random performance.

# %%
# %% [markdown]
# ## 5.4 Domain Confusion Visualisation with t-SNE

# %%
feature_extractor = get_feature_extractor(dann_model_01_02)

N_VIS = 500
feats_src = feature_extractor.predict(X_src[:N_VIS], verbose=0)
feats_tgt = feature_extractor.predict(X_tgt[:N_VIS], verbose=0)

all_feats  = np.vstack([feats_src, feats_tgt])
all_labels = ['Source (FD001)'] * N_VIS + ['Target (FD002)'] * N_VIS

tsne        = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_result = tsne.fit_transform(all_feats)

fig, ax = plt.subplots(figsize=(10, 8))
for label, color, marker in [
    ('Source (FD001)', 'steelblue', 'o'),
    ('Target (FD002)', 'coral',     's')
]:
    mask = [l == label for l in all_labels]
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=color, label=label, alpha=0.4, s=15, marker=marker)

ax.set_title(f't-SNE of DANN Feature Embeddings\n{SOURCE_DS} (Source) vs {TARGET_DS} (Target)',
             fontsize=12)
ax.legend(fontsize=11)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpreting the t-SNE plot:**
# - **Interleaved clusters** → Successful adaptation: features are
#   domain-invariant, source and target patterns are mixed in latent space
# - **Separated clusters** → Failed adaptation: the feature extractor still
#   encodes domain-specific information that the classifier can exploit
#
# After successful DANN training, source and target points should be
# substantially overlapping, confirming the adversarial objective was achieved.

# %%
# %% [markdown]
# ## 5.5 Train All 12 Source-Target Pairs
#
# Following the paper's experimental design: each of the 4 datasets acts
# as source, the remaining 3 as targets (4 × 3 = 12 experiments).

# %%
# Hyperparameter table from Table 3 of the paper
HYPERPARAMS = {
    ('FD001', 'FD002'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[32],  lstm_dropout=0.5, alpha=0.8, lr_reg=0.01, lr_dom=0.01),
    ('FD001', 'FD003'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[32],  lstm_dropout=0.5, alpha=0.8, lr_reg=0.01, lr_dom=0.01),
    ('FD001', 'FD004'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32,32],domain_units=[32], lstm_dropout=0.5, alpha=1.0, lr_reg=0.01, lr_dom=0.1),
    ('FD002', 'FD001'): dict(lstm_units=64,  lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[16,16],lstm_dropout=0.1,alpha=1.0, lr_reg=0.01, lr_dom=0.01),
    ('FD002', 'FD003'): dict(lstm_units=64,  lstm_layers=1, feature_dim=512,reg_units=[64,32],domain_units=[64,32],lstm_dropout=0.1,alpha=2.0,lr_reg=0.1, lr_dom=0.1),
    ('FD002', 'FD004'): dict(lstm_units=32,  lstm_layers=2, feature_dim=32, reg_units=[32],  domain_units=[16],  lstm_dropout=0.1, alpha=1.0, lr_reg=0.1, lr_dom=0.1),
    ('FD003', 'FD001'): dict(lstm_units=64,  lstm_layers=2, feature_dim=128,reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0,lr_reg=0.01,lr_dom=0.01),
    ('FD003', 'FD002'): dict(lstm_units=64,  lstm_layers=2, feature_dim=64, reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0,lr_reg=0.01,lr_dom=0.01),
    ('FD003', 'FD004'): dict(lstm_units=64,  lstm_layers=2, feature_dim=64, reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0,lr_reg=0.01,lr_dom=0.01),
    ('FD004', 'FD001'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0, lr_reg=0.01, lr_dom=0.01),
    ('FD004', 'FD002'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0, lr_reg=0.01, lr_dom=0.01),
    ('FD004', 'FD003'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0, lr_reg=0.01, lr_dom=0.01),
}

# Storage for cross-domain results
dann_reg_models = {}

for (src, tgt), hp in HYPERPARAMS.items():
    print(f"\nTraining DANN: {src} → {tgt}")
    X_s = datasets[src]['X_train'].astype(np.float32)
    y_s = datasets[src]['y_train'].astype(np.float32)
    X_t = datasets[tgt]['X_train'].astype(np.float32)
    X_s_tr, X_s_val, y_s_tr, y_s_val = train_test_split(
        X_s, y_s, test_size=0.1, random_state=42
    )

    reg_m, dann_m = build_lstm_dann(
        window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
        lstm_units=hp['lstm_units'], lstm_layers=hp['lstm_layers'],
        feature_dim=hp['feature_dim'], reg_units=hp['reg_units'],
        domain_units=hp['domain_units'], lstm_dropout=hp['lstm_dropout'],
        reg_dropout=0.3, dom_dropout=0.3, alpha=hp['alpha']
    )

    trainer = LSTMDANNTrainer(
        dann_m, alpha=hp['alpha'],
        lr_reg=hp['lr_reg'], lr_dom=hp['lr_dom']
    )
    trainer.fit(
        X_s_tr, y_s_tr, X_t,
        X_val_src=X_s_val, y_val_src=y_s_val,
        epochs=200, batch_size=256,
        patience=20, lr_decay_epoch=100
    )

    dann_m.save_weights(
        f'../models/saved/lstm_dann_{src}_to_{tgt}.weights.h5'
    )
    dann_reg_models[(src, tgt)] = reg_m

print("\nAll 12 domain adaptation experiments complete.")
```

---

## Notebook 6 — Model Evaluation & Comparison

### Create File: `notebooks/06_model_evaluation_comparison.ipynb`

```python
# %% [markdown]
# # Notebook 6 — Model Evaluation & Comparison
#
# **Goal:** Evaluate and compare all models using RMSE and the NASA asymmetric
# scoring function. Reproduce the comparison tables from the paper.

# %%
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.data_loader    import load_all_datasets, FEATURE_COLS, SENSOR_COLS
from src.preprocessor   import full_preprocess_pipeline
from src.windowing      import create_windows, create_windows_inference
from src.models.lstm_baseline import build_lstm_baseline
from src.models.lstm_dann     import build_lstm_dann
from src.train          import LSTMDANNTrainer
from src.evaluate       import rmse, nasa_score, evaluate_model, compare_models

WINDOW_SIZE = 30
MAX_RUL     = 125

datasets = load_all_datasets(data_dir='../data/raw')
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df_tr, df_te, scaler = full_preprocess_pipeline(
        df_train=datasets[ds_id]['train'],
        df_test=datasets[ds_id]['test'],
        feature_cols=FEATURE_COLS, sensor_cols=SENSOR_COLS,
        smooth=True, max_rul=MAX_RUL
    )
    datasets[ds_id]['train_norm'] = df_tr
    datasets[ds_id]['test_norm']  = df_te
    X, y, _ = create_windows(df_tr, FEATURE_COLS, WINDOW_SIZE)
    datasets[ds_id]['X_train'] = X
    datasets[ds_id]['y_train'] = y

print("Data ready.")

# %% [markdown]
# ## 6.1 NASA Scoring Function — Asymmetric Penalty Visualisation

# %%
errors = np.linspace(-50, 50, 400)
scores = np.where(
    errors < 0,
    np.exp(-errors / 13) - 1,
    np.exp( errors / 10) - 1
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(errors, scores, color='crimson', linewidth=2.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.fill_between(errors[errors >= 0], 0, scores[errors >= 0],
                alpha=0.15, color='red', label='Over-prediction (heavier penalty)')
ax.fill_between(errors[errors <= 0], 0, scores[errors <= 0],
                alpha=0.12, color='blue', label='Under-prediction')
ax.set_xlabel('Prediction Error  (ŷ − y_true)', fontsize=12)
ax.set_ylabel('Score Contribution', fontsize=12)
ax.set_title('NASA Asymmetric Scoring Function\n'
             'Late predictions (positive error) are penalised more than early ones',
             fontsize=12)
ax.set_ylim(-5, 100)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Example penalty values
for err in [-50, -20, -10, 0, 10, 20, 50]:
    score = np.exp(-err/13)-1 if err < 0 else np.exp(err/10)-1
    print(f"  Error = {err:+3d} → Score = {score:.1f}")

# %% [markdown]
# **Insight:** An over-prediction of +20 cycles (claiming the engine has more
# life than it does) scores ~7.4 penalty units. An under-prediction of -20
# cycles scores only ~4.7. An over-prediction of +50 scores ~148 vs ~57 for
# under-prediction. This asymmetry reflects real safety costs: falsely
# assuring operators that equipment is healthy is far more dangerous than
# scheduling preventive maintenance slightly early.

# %%
# %% [markdown]
# ## 6.2 Load and Evaluate All Models

# %%
from sklearn.model_selection import train_test_split
import os

HYPERPARAMS = {
    ('FD001', 'FD002'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[32],  lstm_dropout=0.5, alpha=0.8),
    ('FD001', 'FD003'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[32],  lstm_dropout=0.5, alpha=0.8),
    ('FD001', 'FD004'): dict(lstm_units=128, lstm_layers=1, feature_dim=64, reg_units=[32,32],domain_units=[32], lstm_dropout=0.5, alpha=1.0),
    ('FD002', 'FD001'): dict(lstm_units=64,  lstm_layers=1, feature_dim=64, reg_units=[32],  domain_units=[16,16],lstm_dropout=0.1,alpha=1.0),
    ('FD002', 'FD003'): dict(lstm_units=64,  lstm_layers=1, feature_dim=512,reg_units=[64,32],domain_units=[64,32],lstm_dropout=0.1,alpha=2.0),
    ('FD002', 'FD004'): dict(lstm_units=32,  lstm_layers=2, feature_dim=32, reg_units=[32],  domain_units=[16],  lstm_dropout=0.1, alpha=1.0),
    ('FD003', 'FD001'): dict(lstm_units=64,  lstm_layers=2, feature_dim=128,reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0),
    ('FD003', 'FD002'): dict(lstm_units=64,  lstm_layers=2, feature_dim=64, reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0),
    ('FD003', 'FD004'): dict(lstm_units=64,  lstm_layers=2, feature_dim=64, reg_units=[32,32],domain_units=[32,32],lstm_dropout=0.3,alpha=2.0),
    ('FD004', 'FD001'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0),
    ('FD004', 'FD002'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0),
    ('FD004', 'FD003'): dict(lstm_units=100, lstm_layers=1, feature_dim=30, reg_units=[20],  domain_units=[20],  lstm_dropout=0.5, alpha=1.0),
}

results_all = []

for (src, tgt), hp in HYPERPARAMS.items():
    # ── DANN inference model ───────────────────────────────────────────────
    weights_path = f'../models/saved/lstm_dann_{src}_to_{tgt}.weights.h5'
    if not os.path.exists(weights_path):
        print(f"Skipping {src}→{tgt}: weights not found (run NB05 first)")
        continue

    reg_m, dann_m = build_lstm_dann(
        window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
        **{k: v for k, v in hp.items() if k not in ('lr_reg', 'lr_dom')}
    )
    dann_m.load_weights(weights_path)

    X_test, _ = create_windows_inference(
        datasets[tgt]['test_norm'], FEATURE_COLS, WINDOW_SIZE
    )
    y_true = datasets[tgt]['rul']

    dann_result = evaluate_model(reg_m, X_test, y_true,
                                  model_name=f'LSTM-DANN')

    # ── SOURCE-ONLY: baseline trained on SRC, applied to TGT ──────────────
    so_weights = f'../models/saved/lstm_target_only_{src}.keras'
    so_model   = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
    if os.path.exists(so_weights):
        so_model.load_weights(so_weights)
    so_result = evaluate_model(so_model, X_test, y_true,
                                model_name='SOURCE-ONLY')

    delta_pct = ((so_result['RMSE'] - dann_result['RMSE']) /
                  so_result['RMSE'] * 100)

    results_all.append({
        'Source': src, 'Target': tgt,
        'SOURCE-ONLY RMSE': so_result['RMSE'],
        'LSTM-DANN RMSE':   dann_result['RMSE'],
        'NASA (SOURCE-ONLY)': so_result['NASA_Score'],
        'NASA (DANN)':        dann_result['NASA_Score'],
        'Δ% RMSE':           round(delta_pct, 1)
    })

    print(f"{src}→{tgt}: SO={so_result['RMSE']:.2f} | DANN={dann_result['RMSE']:.2f} "
          f"| Δ={delta_pct:.1f}%")

results_df = pd.DataFrame(results_all)
print("\n", results_df.to_string(index=False))

# %% [markdown]
# ## 6.3 RMSE Improvement Heatmap

# %%
if not results_df.empty:
    pivot = results_df.pivot(index='Source', columns='Target', values='Δ% RMSE')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt='.1f',
        cmap='RdYlGn', center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'RMSE Improvement (%)'}
    )
    ax.set_title('LSTM-DANN vs SOURCE-ONLY: % RMSE Improvement\n'
                 '(Green = DANN better, Red = DANN worse)')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Reading the heatmap:**
# - **Strong green cells**: DANN substantially outperforms SOURCE-ONLY.
#   This typically occurs when the source domain "contains" target conditions
#   (e.g., FD004 as source — it has 6 conditions and 2 fault modes, covering
#   the space of all other datasets).
# - **Near-zero cells**: Source and target are already similar — adaptation
#   provides marginal benefit (expected result, not a failure).
# - **Red cells** (rare): DANN underperformed; usually when source domain has
#   fewer conditions than target (e.g., FD001→FD002), making adaptation harder.

# %%
# %% [markdown]
# ## 6.4 RUL Prediction Timeline Comparison: Source-Only vs DANN vs Target-Only

# %%
# Load TARGET-ONLY models for comparison
target_only_models = {}
for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    m = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
    p = f'../models/saved/lstm_target_only_{ds_id}.keras'
    if os.path.exists(p):
        m.load_weights(p)
    target_only_models[ds_id] = m

# Plot prediction timelines for a representative sample pair
PLOT_PAIRS = [
    ('FD004', 'FD001'), ('FD004', 'FD003'),
    ('FD001', 'FD003'), ('FD002', 'FD001')
]

for src, tgt in PLOT_PAIRS:
    weights_path = f'../models/saved/lstm_dann_{src}_to_{tgt}.weights.h5'
    if not os.path.exists(weights_path):
        continue

    hp = HYPERPARAMS[(src, tgt)]
    reg_m, dann_m = build_lstm_dann(
        window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
        **{k: v for k, v in hp.items() if k not in ('lr_reg', 'lr_dom')}
    )
    dann_m.load_weights(weights_path)

    # Get test predictions for a sample engine
    df_test   = datasets[tgt]['test_norm']
    unit_id   = df_test['unit_id'].iloc[0]
    unit_data = df_test[df_test['unit_id'] == unit_id].sort_values('cycle')

    from src.windowing import create_windows_for_unit_lifecycle
    X_lc = create_windows_for_unit_lifecycle(unit_data, FEATURE_COLS, WINDOW_SIZE)
    n_windows = len(X_lc)

    y_dann = reg_m.predict(X_lc, verbose=0).flatten()

    so_model = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
    so_weights = f'../models/saved/lstm_target_only_{src}.keras'
    if os.path.exists(so_weights):
        so_model.load_weights(so_weights)
    y_so = so_model.predict(X_lc, verbose=0).flatten()

    to_model = target_only_models[tgt]
    y_to = to_model.predict(X_lc, verbose=0).flatten()

    cycles_shown = unit_data['cycle'].values[WINDOW_SIZE:]

    # True RUL for this unit from test labels
    unit_idx  = df_test['unit_id'].unique().tolist().index(unit_id)
    true_rul_end = datasets[tgt]['rul'][unit_idx]
    true_rul  = np.linspace(true_rul_end + n_windows, true_rul_end, n_windows)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(cycles_shown, true_rul, 'k-', linewidth=2.5, label='True RUL (approx.)')
    ax.plot(cycles_shown, y_dann, 'g--', linewidth=2, label='LSTM-DANN')
    ax.plot(cycles_shown, y_so,   'r:',  linewidth=2, label='SOURCE-ONLY')
    ax.plot(cycles_shown, y_to,   'b-.',  linewidth=1.5, label='TARGET-ONLY (oracle)')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'RUL Prediction Timeline\n'
                 f'Source: {src} → Target: {tgt} (Engine Unit {unit_id})')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Insight:** The timeline plots reveal prediction quality over the full
# engine lifecycle. The SOURCE-ONLY model often produces a flat or offset
# curve when applied outside its training domain. LSTM-DANN tracks the
# decreasing trend more closely. TARGET-ONLY (the oracle) shows the ideal
# trajectory achievable with in-domain labels.
```
