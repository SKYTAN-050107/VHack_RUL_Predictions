import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
import json

sys.path.append(os.path.dirname(__file__))

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_IDS  = ['FD001', 'FD002', 'FD003', 'FD004']
FEATURE_COLS = (
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}'     for i in range(1, 22)]
)
SENSOR_COLS  = [f'sensor_{i}' for i in range(1, 22)]
WINDOW_SIZE  = 30
MAX_RUL      = 125

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📂 1. Load Data",
        "🔧 2. Preprocess & Noise",
        "🚨 3. Change-Point Detection",
        "🧠 4. Train Baseline LSTM",
        "🔀 5. Domain Adaptation (DANN)",
        "📊 6. Evaluate & Compare",
        "💡 7. Interpretability (SHAP)",
        "📦 8. Export & API Test",
        "🔁 9. Transfer Learning",
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Run: `streamlit run app.py`")

# ── Status helpers ─────────────────────────────────────────────────────────────
def check_data_dir():
    return all(os.path.exists(f"data/raw/train_{ds}.txt") for ds in DATASET_IDS)

def check_processed():
    return os.path.exists("data/processed/X_train_FD001.npy")

def check_models():
    return os.path.exists("models/saved/lstm_target_only_FD001.keras")

def check_pipelines():
    return any(
        os.path.exists(f"models/saved/pm_pipeline_{ds.lower()}.joblib")
        for ds in DATASET_IDS
    )

def status_badge(ok, label):
    st.sidebar.write(f"{'✅' if ok else '❌'} {label}")

st.sidebar.markdown("**Project Status**")
status_badge(check_data_dir(),   "Raw data present")
status_badge(check_processed(),  "Data preprocessed")
status_badge(check_models(),     "Models trained")
status_badge(check_pipelines(),  "Pipelines exported")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("⚙️ Predictive Maintenance — Modular RUL System")
    st.markdown("""
    This app walks you through every step of the predictive maintenance pipeline,
    from raw data loading to deploying a FastAPI model endpoint.

    ---
    ### What this system does
    - **Predicts Remaining Useful Life (RUL)** of industrial machinery from sensor data
    - **Detects health state transitions** (Healthy → Warning → Critical)
    - **Generalises across machine types** using Domain Adversarial Neural Networks (DANN)
    - **Explains predictions** using SHAP values for factory operators
    - **Fine-tunes on new machines** via two-phase transfer learning
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets",    "4 (FD001–FD004)")
    col2.metric("Sensors",     "21 per engine")
    col3.metric("Window Size", "30 cycles")

    col4, col5, col6 = st.columns(3)
    col4.metric("RUL Cap",     "125 cycles")
    col5.metric("DANN Pairs",  "12 experiments")
    col6.metric("Output",      ".joblib → FastAPI")

    st.markdown("---")
    st.markdown("""
    ### Follow these steps in order:
    | Step | Page | Purpose |
    |------|------|---------|
    | 1 | Load Data | Load C-MAPSS datasets, view statistics |
    | 2 | Preprocess & Noise | Smooth sensors, impute missing, normalise, window |
    | 3 | Change-Point Detection | Find Healthy→Impaired transition per engine |
    | 4 | Train Baseline LSTM | Single-domain RUL model |
    | 5 | Domain Adaptation | LSTM-DANN cross-machine generalisation |
    | 6 | Evaluate & Compare | RMSE + NASA Score across all model pairs |
    | 7 | Interpretability | SHAP feature importance explanations |
    | 8 | Export & API Test | Save .joblib pipeline, test FastAPI endpoint |
    | 9 | Transfer Learning | Fine-tune on your own machine's sensor data |
    """)
    st.info("👈 Use the sidebar to navigate between steps.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 1. Load Data":
    st.title("📂 Step 1 — Load C-MAPSS Data")
    st.markdown("""
    Download the dataset from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
    Place all `.txt` files in `data/raw/`.
    """)

    if not check_data_dir():
        st.error("❌ Raw data not found in `data/raw/`. Download the C-MAPSS dataset first.")
        st.code("\n".join(
            [f"data/raw/train_{ds}.txt" for ds in DATASET_IDS] +
            [f"data/raw/test_{ds}.txt"  for ds in DATASET_IDS] +
            [f"data/raw/RUL_{ds}.txt"   for ds in DATASET_IDS]
        ))
        st.stop()

    st.success("✅ Raw data found.")

    try:
        from src.data_loader import load_all_datasets
        with st.spinner("Loading all four datasets..."):
            datasets = load_all_datasets(data_dir='data/raw')
        st.session_state['datasets'] = datasets
        st.success("All four datasets loaded.")
    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()

    st.subheader("Dataset Summary")
    rows = []
    for ds_id, splits in datasets.items():
        df = splits['train']
        rows.append({
            'Dataset': ds_id,
            'Train Engines': df['unit_id'].nunique(),
            'Test Engines':  splits['test']['unit_id'].nunique(),
            'Total Train Rows': len(df),
            'Avg Cycles/Engine': round(df.groupby('unit_id')['cycle'].max().mean(), 1),
            'Min Cycles': int(df.groupby('unit_id')['cycle'].max().min()),
            'Max Cycles': int(df.groupby('unit_id')['cycle'].max().max()),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Engine Lifetime Distributions")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for ax, (ds_id, splits), color in zip(axes, datasets.items(), colors):
        ll = splits['train'].groupby('unit_id')['cycle'].max()
        ax.hist(ll, bins=20, color=color, edgecolor='black', alpha=0.8)
        ax.axvline(ll.mean(), color='black', linestyle='--', linewidth=1.2)
        ax.set_title(f'{ds_id}\nMean: {ll.mean():.0f} cycles')
        ax.set_xlabel('Total Cycles'); ax.set_ylabel('Count')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Raw Data Preview")
    ds_sel = st.selectbox("Select dataset", DATASET_IDS)
    st.dataframe(datasets[ds_sel]['train'].head(10), use_container_width=True)

    st.subheader("Sensor Distribution Comparison")
    sensor_sel = st.selectbox("Select sensor", SENSOR_COLS, index=10)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for ds_id, color in zip(DATASET_IDS, colors):
        datasets[ds_id]['train'][sensor_sel].hist(
            bins=60, alpha=0.5, label=ds_id, color=color, density=True, ax=ax2
        )
    ax2.set_title(f'{sensor_sel} Distribution Across Datasets')
    ax2.set_xlabel('Sensor Value'); ax2.set_ylabel('Density'); ax2.legend()
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREPROCESS & NOISE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 2. Preprocess & Noise":
    st.title("🔧 Step 2 — Preprocessing & Noise Handling")

    if 'datasets' not in st.session_state:
        st.warning("Run Step 1 first to load the data.")
        st.stop()

    datasets = st.session_state['datasets']

    col1, col2 = st.columns(2)
    with col1:
        apply_smooth = st.checkbox("Apply Savitzky-Golay smoothing", value=True)
        sg_window    = st.slider("SG window length (odd)", 5, 21, 11, step=2)
        sg_poly      = st.slider("SG polynomial order", 2, 5, 3)
    with col2:
        missing_rate = st.slider("Inject synthetic missing data (%)", 0, 10, 3)
        max_rul_val  = st.number_input("RUL cap", 50, 200, 125, step=5)

    st.subheader("Noise Filter Preview")
    demo_ds     = st.selectbox("Dataset", DATASET_IDS, key='noise_ds')
    demo_sensor = st.selectbox("Sensor",  SENSOR_COLS, index=10, key='noise_sensor')
    demo_unit   = st.number_input("Engine Unit ID", 1, 20, 1, key='noise_unit')

    from scipy.signal import savgol_filter
    unit_data = datasets[demo_ds]['train']
    unit_data = unit_data[unit_data['unit_id'] == demo_unit].sort_values('cycle')

    if len(unit_data) > 0:
        raw_signal    = unit_data[demo_sensor].values
        smooth_signal = savgol_filter(raw_signal, sg_window, sg_poly) if len(raw_signal) >= sg_window else raw_signal
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(unit_data['cycle'], raw_signal, alpha=0.6, label='Raw', color='steelblue')
        axes[0].plot(unit_data['cycle'], smooth_signal, color='red', linewidth=2, label='Filtered')
        axes[0].set_title(f'{demo_sensor} — Unit {demo_unit}'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(unit_data['cycle'], raw_signal - smooth_signal, color='gray', alpha=0.7)
        axes[1].axhline(0, color='black', linewidth=0.8); axes[1].set_title('Residual (Removed Noise)')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Run Full Preprocessing Pipeline")
    if st.button("▶ Run Preprocessing (all 4 datasets)", type="primary"):
        try:
            from src.preprocessor import full_preprocess_pipeline
            from src.windowing    import create_windows
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('models/saved',   exist_ok=True)
            progress = st.progress(0)
            status   = st.empty()
            for i, ds_id in enumerate(DATASET_IDS):
                status.info(f"Processing {ds_id}...")
                df_tr, df_te, scaler = full_preprocess_pipeline(
                    df_train=datasets[ds_id]['train'],
                    df_test=datasets[ds_id]['test'],
                    feature_cols=FEATURE_COLS, sensor_cols=SENSOR_COLS,
                    smooth=apply_smooth, max_rul=int(max_rul_val),
                    scaler_save_path=f'models/saved/scaler_{ds_id}.joblib'
                )
                datasets[ds_id]['train_norm'] = df_tr
                datasets[ds_id]['test_norm']  = df_te
                X, y, _ = create_windows(df_tr, FEATURE_COLS, WINDOW_SIZE)
                datasets[ds_id]['X_train'] = X
                datasets[ds_id]['y_train'] = y
                np.save(f'data/processed/X_train_{ds_id}.npy', X)
                np.save(f'data/processed/y_train_{ds_id}.npy', y)
                progress.progress((i + 1) / len(DATASET_IDS))
            st.session_state['datasets'] = datasets
            status.success("✅ All datasets preprocessed and saved.")
            rows = [{'Dataset': ds_id, 'Windows': datasets[ds_id]['X_train'].shape[0],
                      'Shape': str(datasets[ds_id]['X_train'].shape[1:])} for ds_id in DATASET_IDS]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}"); st.exception(e)

    if 'train_norm' in datasets.get('FD001', {}):
        st.subheader("Normalised Sensor Distributions")
        norm_sensor = st.selectbox("Sensor to compare", SENSOR_COLS, index=10, key='norm_cmp')
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        for ds_id, color in zip(DATASET_IDS, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
            if 'train_norm' in datasets[ds_id]:
                datasets[ds_id]['train_norm'][norm_sensor].hist(
                    bins=60, alpha=0.5, label=ds_id, color=color, density=True, ax=ax3
                )
        ax3.set_title(f'{norm_sensor} — Normalised (per-dataset shift preserved)')
        ax3.set_xlabel('Normalised [0,1]'); ax3.legend()
        plt.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CHANGE-POINT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 3. Change-Point Detection":
    st.title("🚨 Step 3 — Anomaly & Change-Point Detection")

    datasets = st.session_state.get('datasets', {})
    if not datasets or 'train_norm' not in datasets.get('FD001', {}):
        st.warning("Run Step 2 (Preprocessing) first.")
        st.stop()

    from src.changepoint import cusum_detector, detect_health_transitions

    col1, col2 = st.columns(2)
    with col1:
        cp_ds     = st.selectbox("Dataset", DATASET_IDS)
        cp_unit   = st.number_input("Engine Unit ID", 1, 50, 1)
        threshold = st.slider("CUSUM Threshold", 1.0, 10.0, 5.0, 0.5)
    with col2:
        cp_sensor = st.selectbox("Sensor to monitor", SENSOR_COLS, index=10)

    df_norm = datasets[cp_ds]['train_norm']
    unit_df = df_norm[df_norm['unit_id'] == cp_unit].sort_values('cycle')

    if len(unit_df) == 0:
        st.warning(f"Unit {cp_unit} not found.")
    else:
        cp_idx   = cusum_detector(unit_df[cp_sensor].values, threshold=threshold)
        cp_cycle = unit_df['cycle'].iloc[cp_idx] if cp_idx is not None else None
        fig, ax  = plt.subplots(figsize=(14, 5))
        ax.plot(unit_df['cycle'], unit_df[cp_sensor], color='steelblue', linewidth=1.8, label=cp_sensor)
        if cp_cycle:
            ax.axvline(cp_cycle, color='red', linestyle='--', linewidth=2.2,
                       label=f'CUSUM trigger @ cycle {cp_cycle}')
            ax.axvspan(cp_cycle, unit_df['cycle'].max(), alpha=0.07, color='red')
            ax.text(cp_cycle + 1, unit_df[cp_sensor].quantile(0.9), '⚠ IMPAIRED', color='red', fontsize=10)
        ax.axvspan(unit_df['cycle'].min(),
                   cp_cycle if cp_cycle else unit_df['cycle'].max(),
                   alpha=0.04, color='green')
        ax.set_title(f'CUSUM Detection — Unit {cp_unit}, {cp_ds}')
        ax.set_xlabel('Cycle'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if cp_cycle:
            st.success(f"🔴 Impairment detected at cycle **{cp_cycle}** — "
                       f"**{unit_df['cycle'].max() - cp_cycle} cycles** warning provided.")
        else:
            st.info("No change-point detected for this combination.")

    st.subheader(f"Fleet-Wide Transition Statistics ({cp_ds})")
    if st.button("Run fleet-wide CUSUM detection"):
        with st.spinner("Analysing all engines..."):
            transitions = detect_health_transitions(
                datasets[cp_ds]['train_norm'], SENSOR_COLS, threshold=threshold
            )
        st.dataframe(transitions.describe().round(1).T, use_container_width=True)
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(transitions['health_transition_cycle'], bins=20, color='coral', edgecolor='black', alpha=0.8)
        axes[0].axvline(transitions['health_transition_cycle'].median(), color='red', linestyle='--')
        axes[0].set_title('Impairment Onset Cycle'); axes[0].set_xlabel('Cycle')
        axes[1].hist(transitions['rul_at_transition'], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
        axes[1].axvline(transitions['rul_at_transition'].median(), color='navy', linestyle='--',
                        label=f"Median: {transitions['rul_at_transition'].median():.0f} cycles")
        axes[1].set_title('Warning Lead Time (RUL at Detection)'); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: TRAIN BASELINE LSTM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 4. Train Baseline LSTM":
    st.title("🧠 Step 4 — Baseline LSTM (SOURCE/TARGET-ONLY)")

    datasets = st.session_state.get('datasets', {})
    if not datasets or 'X_train' not in datasets.get('FD001', {}):
        st.warning("Run Step 2 (Preprocessing) first.")
        st.stop()

    from src.models.lstm_baseline import build_lstm_baseline
    from src.windowing            import create_windows_inference
    from src.evaluate             import rmse, nasa_score

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ds   = st.selectbox("Dataset to train on", DATASET_IDS)
        lstm_units = st.number_input("LSTM units", 32, 256, 100, step=16)
    with col2:
        epochs     = st.number_input("Max epochs", 10, 200, 50, step=10)
        batch_size = st.selectbox("Batch size", [128, 256, 512], index=1)
    with col3:
        dropout    = st.slider("LSTM dropout", 0.0, 0.8, 0.5, 0.1)
        patience   = st.number_input("Early stopping patience", 5, 50, 20)

    if st.button(f"▶ Train on {train_ds}", type="primary"):
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        import tensorflow as tf

        X_all = datasets[train_ds]['X_train'].astype('float32')
        y_all = datasets[train_ds]['y_train'].astype('float32')
        X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)

        model = build_lstm_baseline(
            window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
            lstm_units=int(lstm_units), dropout_rate=float(dropout)
        )
        os.makedirs('models/saved', exist_ok=True)
        cbs = [
            EarlyStopping(monitor='val_loss', patience=int(patience), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        ]

        history_loss, history_val = [], []
        progress   = st.progress(0)
        loss_chart = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                history_loss.append(logs.get('loss', 0))
                history_val.append(logs.get('val_loss', 0))
                progress.progress(min((epoch + 1) / int(epochs), 1.0))
                if len(history_loss) > 1:
                    fig_loss, ax_l = plt.subplots(figsize=(8, 3))
                    ax_l.plot(history_loss, label='Train Loss')
                    ax_l.plot(history_val,  label='Val Loss')
                    ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('MSE'); ax_l.legend(); ax_l.grid(alpha=0.3)
                    plt.tight_layout(); loss_chart.pyplot(fig_loss); plt.close()

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                   epochs=int(epochs), batch_size=int(batch_size),
                   callbacks=cbs + [StreamlitCallback()], verbose=0)

        weights_path = f'models/saved/lstm_target_only_{train_ds}.keras'
        model.save_weights(weights_path)
        st.success(f"✅ Saved: `{weights_path}`")

        X_test, _ = create_windows_inference(datasets[train_ds]['test_norm'], FEATURE_COLS, WINDOW_SIZE)
        y_test    = datasets[train_ds]['rul']
        y_pred    = model.predict(X_test, verbose=0).flatten()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("RMSE",       f"{rmse(y_test, y_pred):.2f}")
        col_b.metric("MAE",        f"{np.mean(np.abs(y_pred - y_test)):.2f}")
        col_c.metric("NASA Score", f"{nasa_score(y_test, y_pred):.0f}")

        fig_sc, ax_sc = plt.subplots(figsize=(7, 6))
        ax_sc.scatter(y_test, y_pred, alpha=0.4, s=12, color='steelblue')
        lim = max(y_test.max(), y_pred.max()) + 5
        ax_sc.plot([0, lim], [0, lim], 'r--', linewidth=1.5)
        ax_sc.set_xlabel('True RUL'); ax_sc.set_ylabel('Predicted RUL')
        ax_sc.set_title(f'{train_ds} Test — Predicted vs True RUL'); ax_sc.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_sc); plt.close()

    st.subheader("Existing Trained Models")
    model_rows = [{'Dataset': ds_id,
                   'Status': '✅ Trained' if os.path.exists(f'models/saved/lstm_target_only_{ds_id}.keras') else '❌ Not trained'}
                  for ds_id in DATASET_IDS]
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: DANN DOMAIN ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔀 5. Domain Adaptation (DANN)":
    st.title("🔀 Step 5 — LSTM-DANN Domain Adaptation")

    datasets = st.session_state.get('datasets', {})
    if not datasets or 'X_train' not in datasets.get('FD001', {}):
        st.warning("Run Step 2 (Preprocessing) first.")
        st.stop()

    from src.models.lstm_dann import build_lstm_dann
    from src.train            import LSTMDANNTrainer
    from sklearn.model_selection import train_test_split

    col1, col2 = st.columns(2)
    with col1:
        src_ds = st.selectbox("Source Domain (labelled)",   DATASET_IDS, index=0)
        tgt_ds = st.selectbox("Target Domain (unlabelled)", DATASET_IDS, index=1)
    with col2:
        if src_ds == tgt_ds:
            st.warning("Source and target must be different.")

    col3, col4, col5 = st.columns(3)
    with col3:
        lstm_units  = st.number_input("LSTM units",        32,  256, 128, step=16)
        lstm_layers = st.selectbox("LSTM layers",          [1, 2], index=0)
        feature_dim = st.number_input("Feature dim",       16,  256,  64, step=16)
    with col4:
        alpha       = st.slider("GRL alpha",    0.1, 3.0, 0.8, 0.1)
        lstm_drop   = st.slider("LSTM dropout", 0.0, 0.8, 0.5, 0.1)
        dann_epochs = st.number_input("Max epochs", 10, 200, 50, step=10)
    with col5:
        lr_reg      = st.select_slider("LR regression",   [0.001, 0.01, 0.1], value=0.01)
        lr_dom      = st.select_slider("LR domain",       [0.001, 0.01, 0.1], value=0.01)
        dann_batch  = st.selectbox("Batch size",          [128, 256, 512], index=1, key='dann_batch')

    if src_ds != tgt_ds and st.button(f"▶ Train DANN: {src_ds} → {tgt_ds}", type="primary"):
        import tensorflow as tf
        X_s = datasets[src_ds]['X_train'].astype('float32')
        y_s = datasets[src_ds]['y_train'].astype('float32')
        X_t = datasets[tgt_ds]['X_train'].astype('float32')
        X_s_tr, X_s_val, y_s_tr, y_s_val = train_test_split(X_s, y_s, test_size=0.1, random_state=42)

        reg_m, dann_m = build_lstm_dann(
            window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS),
            lstm_units=int(lstm_units), lstm_layers=int(lstm_layers),
            feature_dim=int(feature_dim), reg_units=[32], domain_units=[32],
            lstm_dropout=float(lstm_drop), reg_dropout=0.3, dom_dropout=0.3, alpha=float(alpha)
        )
        trainer = LSTMDANNTrainer(dann_m, alpha=float(alpha), lr_reg=float(lr_reg), lr_dom=float(lr_dom))

        rul_losses, dom_losses, val_maes = [], [], []
        progress   = st.progress(0)
        chart_area = st.empty()

        n_src = len(X_s_tr)
        X_t_use = X_t.copy()
        if len(X_t_use) < n_src:
            repeat  = int(np.ceil(n_src / len(X_t_use)))
            X_t_use = np.tile(X_t_use, (repeat, 1, 1))[:n_src]

        n_batches   = int(np.ceil(n_src / int(dann_batch)))
        best_val    = np.inf
        no_imp      = 0
        best_w      = None
        patience_d  = 20

        for epoch in range(int(dann_epochs)):
            if epoch == 100:
                trainer.reg_opt.learning_rate.assign(float(lr_reg) * 0.1)
                trainer.dom_opt.learning_rate.assign(float(lr_dom) * 0.1)

            src_idx = np.random.permutation(n_src)
            tgt_idx = np.random.permutation(len(X_t_use))
            ep_rul, ep_dom = [], []

            for b in range(n_batches):
                sl = src_idx[b*int(dann_batch):(b+1)*int(dann_batch)]
                tl = tgt_idx[b*int(dann_batch):(b+1)*int(dann_batch)]
                rl, dl = trainer.train_step(X_s_tr[sl], y_s_tr[sl], X_t_use[tl])
                ep_rul.append(float(rl)); ep_dom.append(float(dl))

            rul_losses.append(np.mean(ep_rul)); dom_losses.append(np.mean(ep_dom))
            vp, _ = dann_m(X_s_val, training=False)
            vm = float(tf.reduce_mean(tf.abs(y_s_val[:, np.newaxis] - vp)))
            val_maes.append(vm)

            if vm < best_val: best_val = vm; best_w = dann_m.get_weights(); no_imp = 0
            else: no_imp += 1

            progress.progress((epoch + 1) / int(dann_epochs))
            if epoch % 5 == 0 and len(rul_losses) > 1:
                fig_t, ax_t = plt.subplots(1, 3, figsize=(15, 3))
                ax_t[0].plot(rul_losses, color='steelblue'); ax_t[0].set_title('RUL Loss')
                ax_t[1].plot(dom_losses, color='coral')
                ax_t[1].axhline(np.log(2), color='gray', linestyle='--'); ax_t[1].set_title('Domain Loss')
                ax_t[2].plot(val_maes, color='seagreen'); ax_t[2].set_title('Val MAE')
                for a in ax_t: a.grid(alpha=0.3)
                plt.tight_layout(); chart_area.pyplot(fig_t); plt.close()

            if no_imp >= patience_d: break

        if best_w: dann_m.set_weights(best_w)
        save_path = f'models/saved/lstm_dann_{src_ds}_to_{tgt_ds}.weights.h5'
        dann_m.save_weights(save_path)
        st.success(f"✅ Saved: `{save_path}`")
        st.metric("Best Val MAE (source)", f"{best_val:.4f}")

    st.subheader("Trained DANN Pairs")
    pairs = []
    for s in DATASET_IDS:
        for t in DATASET_IDS:
            if s != t:
                p = f'models/saved/lstm_dann_{s}_to_{t}.weights.h5'
                pairs.append({'Source': s, 'Target': t, 'Status': '✅' if os.path.exists(p) else '—'})
    pairs_df = pd.DataFrame(pairs).pivot(index='Source', columns='Target', values='Status').fillna('—')
    st.dataframe(pairs_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: EVALUATE & COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 6. Evaluate & Compare":
    st.title("📊 Step 6 — Model Evaluation & Comparison")

    datasets = st.session_state.get('datasets', {})
    if not datasets or 'X_train' not in datasets.get('FD001', {}):
        st.warning("Run Step 2 first.")
        st.stop()

    from src.models.lstm_baseline import build_lstm_baseline
    from src.models.lstm_dann     import build_lstm_dann
    from src.windowing            import create_windows_inference
    from src.evaluate             import rmse, nasa_score

    st.subheader("NASA Scoring Function — Asymmetric Penalty")
    errors = np.linspace(-60, 60, 500)
    scores = np.where(errors < 0, np.exp(-errors/13)-1, np.exp(errors/10)-1)
    fig_n, ax_n = plt.subplots(figsize=(10, 4))
    ax_n.plot(errors, scores, color='crimson', linewidth=2.5)
    ax_n.fill_between(errors[errors >= 0], 0, scores[errors >= 0], alpha=0.15, color='red',  label='Over-prediction (heavier penalty)')
    ax_n.fill_between(errors[errors <= 0], 0, scores[errors <= 0], alpha=0.1,  color='blue', label='Under-prediction')
    ax_n.axvline(0, color='gray', linestyle='--'); ax_n.set_ylim(-5, 80)
    ax_n.set_xlabel('Error (ŷ − y)'); ax_n.set_ylabel('Score'); ax_n.legend(); ax_n.grid(alpha=0.25)
    plt.tight_layout(); st.pyplot(fig_n); plt.close()

    st.subheader("Head-to-Head Comparison")
    col1, col2 = st.columns(2)
    eval_src = col1.selectbox("Source / Training Dataset", DATASET_IDS, key='eval_src')
    eval_tgt = col2.selectbox("Target / Test Dataset",     DATASET_IDS, key='eval_tgt', index=1)

    if st.button("▶ Run Comparison", type="primary"):
        X_test, _ = create_windows_inference(datasets[eval_tgt]['test_norm'], FEATURE_COLS, WINDOW_SIZE)
        y_true    = datasets[eval_tgt]['rul']
        results   = []

        for label, weights_path, is_dann in [
            (f'SOURCE-ONLY ({eval_src})', f'models/saved/lstm_target_only_{eval_src}.keras', False),
            (f'TARGET-ONLY ({eval_tgt})', f'models/saved/lstm_target_only_{eval_tgt}.keras', False),
            (f'LSTM-DANN ({eval_src}→{eval_tgt})', f'models/saved/lstm_dann_{eval_src}_to_{eval_tgt}.weights.h5', True),
        ]:
            if not os.path.exists(weights_path): continue
            if is_dann and eval_src == eval_tgt: continue
            if is_dann:
                reg_m, dann_m = build_lstm_dann(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS))
                dann_m.load_weights(weights_path)
                yp = reg_m.predict(X_test, verbose=0).flatten()
            else:
                m = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
                m.load_weights(weights_path)
                yp = m.predict(X_test, verbose=0).flatten()
            results.append({'Model': label,
                             'RMSE': round(rmse(y_true, yp), 2),
                             'NASA Score': round(nasa_score(y_true, yp), 0),
                             'MAE': round(float(np.mean(np.abs(yp - y_true))), 2),
                             'y_pred': yp})

        if results:
            st.dataframe(pd.DataFrame(results)[['Model','RMSE','MAE','NASA Score']], use_container_width=True)
            fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
            x = range(len(results)); labels = [r['Model'] for r in results]
            clrs = ['#d62728','#2ca02c','#1f77b4'][:len(results)]
            axes_b[0].bar(x, [r['RMSE'] for r in results], color=clrs, edgecolor='black', alpha=0.8)
            axes_b[0].set_xticks(x); axes_b[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
            axes_b[0].set_title('RMSE (lower = better)'); axes_b[0].grid(axis='y', alpha=0.3)
            axes_b[1].bar(x, [r['NASA Score'] for r in results], color=clrs, edgecolor='black', alpha=0.8)
            axes_b[1].set_xticks(x); axes_b[1].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
            axes_b[1].set_title('NASA Score (lower = better)'); axes_b[1].grid(axis='y', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_b); plt.close()
        else:
            st.warning("No trained models found for this pair. Train models in Steps 4 and 5 first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 7. Interpretability (SHAP)":
    st.title("💡 Step 7 — SHAP Interpretability")

    datasets = st.session_state.get('datasets', {})
    if not datasets or 'X_train' not in datasets.get('FD001', {}):
        st.warning("Run Step 2 first.")
        st.stop()

    from src.models.lstm_baseline import build_lstm_baseline
    from src.windowing            import create_windows_inference
    from src.explainer            import (build_shap_explainer, compute_shap_values,
                                           aggregate_shap_by_feature, build_explanation_text,
                                           plot_feature_importance, plot_shap_heatmap)

    shap_ds      = st.selectbox("Dataset", DATASET_IDS)
    weights_path = f'models/saved/lstm_target_only_{shap_ds}.keras'

    if not os.path.exists(weights_path):
        st.error(f"No trained model for {shap_ds}. Run Step 4 first.")
        st.stop()

    model = build_lstm_baseline(WINDOW_SIZE, len(FEATURE_COLS))
    model.load_weights(weights_path)

    n_bg      = st.slider("Background samples", 50, 200, 100, 25)
    n_explain = st.slider("Samples to explain", 20, 200, 50, 10)

    if st.button("▶ Compute SHAP Values", type="primary"):
        with st.spinner("Building explainer..."):
            X_tr      = datasets[shap_ds]['X_train'].astype('float32')
            explainer = build_shap_explainer(model, X_tr, n_background=int(n_bg))
        X_test, test_units = create_windows_inference(datasets[shap_ds]['test_norm'], FEATURE_COLS, WINDOW_SIZE)
        X_exp = X_test[:int(n_explain)].astype('float32')
        with st.spinner("Computing SHAP values..."):
            shap_vals = compute_shap_values(explainer, X_exp)
        st.session_state.update({
            'shap_vals': shap_vals, 'shap_X_exp': X_exp,
            'shap_units': test_units[:int(n_explain)],
            'shap_y_test': datasets[shap_ds]['rul'][:int(n_explain)],
            'shap_model': model
        })
        st.success("✅ SHAP values computed.")

    if 'shap_vals' in st.session_state:
        shap_vals = st.session_state['shap_vals']
        X_exp     = st.session_state['shap_X_exp']
        y_test    = st.session_state['shap_y_test']
        model     = st.session_state['shap_model']

        st.subheader("Fleet-Level Feature Importance")
        feat_imp = aggregate_shap_by_feature(shap_vals, FEATURE_COLS)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 9))
        plot_feature_importance(feat_imp, ax=ax_fi)
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()
        st.info(f"**Top 3 drivers:** {', '.join(feat_imp.head(3).index.tolist())}")

        st.subheader("Single Engine Explanation")
        sample_idx  = st.slider("Test engine index", 0, len(X_exp)-1, 0)
        pred_rul    = float(model.predict(X_exp[sample_idx:sample_idx+1], verbose=0).flatten()[0])
        true_rul    = float(y_test[sample_idx])
        health      = 'Critical' if pred_rul < 20 else ('Warning' if pred_rul < 50 else 'Healthy')

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted RUL", f"{pred_rul:.1f}"); col_b.metric("True RUL", f"{true_rul:.0f}"); col_c.metric("Health", health)

        single_imp  = pd.Series(np.abs(shap_vals[sample_idx]).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
        explanation = build_explanation_text(single_imp, pred_rul, health, top_k=3)
        {'Healthy': st.success, 'Warning': st.warning, 'Critical': st.error}.get(health, st.info)(f"**Operator Message:** {explanation}")

        top_k_feat = single_imp.nlargest(10).sort_values(ascending=True)
        fig_wf, ax_wf = plt.subplots(figsize=(9, 5))
        colors_wf = ['#d73027' if v > single_imp.mean() else '#1a9850' for v in top_k_feat.values]
        top_k_feat.plot(kind='barh', ax=ax_wf, color=colors_wf, edgecolor='black', alpha=0.85)
        ax_wf.set_title(f'Feature Contributions — Engine {st.session_state["shap_units"][sample_idx]}')
        plt.tight_layout(); st.pyplot(fig_wf); plt.close()

        st.subheader("Temporal SHAP Heatmap")
        fig_hm = plot_shap_heatmap(shap_vals[sample_idx], FEATURE_COLS)
        plt.tight_layout(); st.pyplot(fig_hm); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: EXPORT & API TEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 8. Export & API Test":
    st.title("📦 Step 8 — Export Pipeline & API Test")

    datasets = st.session_state.get('datasets', {})

    st.subheader("Export .joblib Pipelines")
    export_ds    = st.selectbox("Export pipeline for", DATASET_IDS)
    weights_path = f'models/saved/lstm_target_only_{export_ds}.keras'
    scaler_path  = f'models/saved/scaler_{export_ds}.joblib'

    col1, col2 = st.columns(2)
    col1.write(f"Weights: {'✅' if os.path.exists(weights_path) else '❌'} `{weights_path}`")
    col2.write(f"Scaler:  {'✅' if os.path.exists(scaler_path)  else '❌'} `{scaler_path}`")

    if st.button(f"▶ Export {export_ds} Pipeline", type="primary"):
        if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
            st.error("Train the model (Step 4) and preprocess (Step 2) first.")
        else:
            try:
                from src.preprocessor           import full_preprocess_pipeline
                from src.models.lstm_baseline    import build_lstm_baseline
                from src.changepoint             import cusum_detector, classify_health_state
                from scipy.signal                import savgol_filter as sgf

                scaler = joblib.load(scaler_path)

                class PredictiveMaintenancePipeline:
                    def __init__(self, scaler, model_weights_path, feature_cols,
                                  window_size=30, max_rul=125, cusum_threshold=5.0):
                        self.scaler             = scaler
                        self.model_weights_path = model_weights_path
                        self.feature_cols       = feature_cols
                        self.window_size        = window_size
                        self.max_rul            = max_rul
                        self.cusum_threshold    = cusum_threshold
                        self._model             = None

                    def _load_model(self):
                        if self._model is None:
                            self._model = build_lstm_baseline(self.window_size, len(self.feature_cols))
                            self._model.load_weights(self.model_weights_path)

                    def predict(self, X_raw):
                        self._load_model()
                        X = X_raw.astype(np.float64).copy()
                        if len(X) >= 11:
                            for j in range(X.shape[1]):
                                X[:, j] = sgf(X[:, j], 11, 3)
                        X = pd.DataFrame(X, columns=self.feature_cols)
                        X = X.interpolate(method='linear', limit_direction='both').values
                        X_norm = self.scaler.transform(X)
                        T = len(X_norm)
                        if T < self.window_size:
                            X_norm = np.vstack([np.zeros((self.window_size - T, X_norm.shape[1])), X_norm])
                        window = X_norm[-self.window_size:][np.newaxis]
                        rul    = float(np.clip(self._model.predict(window, verbose=0).flatten()[0], 0, self.max_rul))
                        n_r    = min(50, len(X_norm))
                        idx_11 = self.feature_cols.index('sensor_11')
                        cp     = cusum_detector(X_norm[-n_r:, idx_11], threshold=self.cusum_threshold)
                        health = classify_health_state(rul, cp is not None)
                        return {'rul_prediction': round(rul, 1), 'health_state': health,
                                'change_point_detected': cp is not None,
                                'change_point_step': int(cp) if cp is not None else None}

                    def __getstate__(self):
                        state = self.__dict__.copy(); state['_model'] = None; return state
                    def __setstate__(self, state):
                        self.__dict__.update(state); self._model = None

                pipeline      = PredictiveMaintenancePipeline(
                    scaler=scaler,
                    model_weights_path=os.path.abspath(weights_path),
                    feature_cols=FEATURE_COLS, window_size=WINDOW_SIZE, max_rul=MAX_RUL
                )
                pipeline_path = f'models/saved/pm_pipeline_{export_ds.lower()}.joblib'
                joblib.dump(pipeline, pipeline_path)
                st.success(f"✅ Saved: `{pipeline_path}`")
            except Exception as e:
                st.error(f"Export failed: {e}"); st.exception(e)

    st.subheader("Test Exported Pipeline")
    test_ds   = st.selectbox("Test pipeline for", DATASET_IDS, key='test_pipe_ds')
    pipe_path = f'models/saved/pm_pipeline_{test_ds.lower()}.joblib'

    if os.path.exists(pipe_path):
        st.success(f"Pipeline found: `{pipe_path}`")
        if st.button("▶ Run Test Prediction", key='run_pipe_test'):
            try:
                pipeline = joblib.load(pipe_path)
                if datasets and 'test' in datasets.get(test_ds, {}):
                    sample = datasets[test_ds]['test'][datasets[test_ds]['test']['unit_id'] == 1][FEATURE_COLS].values
                else:
                    sample = np.random.rand(50, 24).astype('float32')
                result = pipeline.predict(sample)
                st.json(result)
                state_color = {'Healthy': '🟢', 'Warning': '🟡', 'Critical': '🔴'}
                st.markdown(f"### {state_color.get(result['health_state'], '⚪')} {result['health_state']}")
                st.metric("Predicted RUL", f"{result['rul_prediction']} cycles")
            except Exception as e:
                st.error(f"Test failed: {e}"); st.exception(e)
    else:
        st.warning(f"No pipeline for {test_ds}. Export it above first.")

    st.subheader("Launch FastAPI Server")
    st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")

    st.subheader("Live API Test")
    api_url = st.text_input("FastAPI URL", "http://localhost:8000")
    if st.button("Check API Health"):
        try:
            import requests as req
            r = req.get(f"{api_url}/health", timeout=3)
            st.success(f"✅ API running: {r.json()}") if r.status_code == 200 else st.error(f"Status {r.status_code}")
        except Exception:
            st.error("Cannot connect. Start it with: `uvicorn api.main:app --port 8000`")

    st.subheader("Saved Files")
    if os.path.exists('models/saved'):
        file_rows = [{'File': f, 'Size (KB)': round(os.path.getsize(f'models/saved/{f}') / 1024, 1)}
                     for f in sorted(os.listdir('models/saved'))]
        if file_rows:
            st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9: TRANSFER LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔁 9. Transfer Learning":
    st.title("🔁 Step 9 — Transfer Learning for New Machines")

    st.markdown("""
    Adapt a pretrained LSTM to **your specific machine** using a small
    labelled dataset — as few as 35 cycles of sensor data with any number
    of sensor columns.

    | Phase | What happens | When to use |
    |-------|-------------|-------------|
    | **Feature Alignment** | Maps your sensors (any count) to the model's 24-dim input | Always — automatic |
    | **Phase 1 (Head-only)** | Trains only output layers, LSTM frozen | All cases |
    | **Phase 2 (Full fine-tune)** | Unlocks LSTM, trains end-to-end at low LR | When you have ≥ 20 labelled cycles |
    """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📤 Upload & Fine-Tune",
        "🔮 Predict (Adapted Model)",
        "📋 Machine Registry"
    ])

    with tab1:
        st.subheader("Upload Your Machine Data & Fine-Tune")

        col1, col2 = st.columns(2)
        with col1:
            machine_id = st.text_input("Machine ID", value="my_machine_001")
            base_ds    = st.selectbox("Base pretrained model", DATASET_IDS)
        with col2:
            p1_epochs = st.number_input("Phase 1 epochs (head-only)",    5,  100, 30)
            p2_epochs = st.number_input("Phase 2 epochs (full finetune)", 0,  100, 20)
            p1_lr     = st.select_slider("Phase 1 LR", [0.01, 0.001, 0.0001], value=0.001)
            p2_lr     = st.select_slider("Phase 2 LR", [0.001, 0.0001, 0.00001], value=0.0001)

        st.markdown("#### Upload Sensor Data CSV")
        st.markdown("""
        Your CSV must have:
        - One row per cycle (oldest → newest)
        - Any number of sensor columns
        - A column named **`RUL`** with remaining useful life per cycle
        ```
        sensor_temp, sensor_pressure, sensor_vibration, RUL
        0.45, 0.62, 0.33, 120
        0.46, 0.63, 0.34, 119
        ```
        """)

        with st.expander("📥 Generate a demo CSV to test with"):
            n_demo = st.slider("Demo cycles", 50, 300, 150)
            if st.button("Generate Demo CSV"):
                np.random.seed(42)
                cycles    = np.arange(1, n_demo + 1)
                demo_data = {}
                for i in range(1, 9):
                    trend = np.linspace(0.3, 0.9, n_demo) if i % 3 == 0 else np.linspace(0.7, 0.2, n_demo)
                    demo_data[f'sensor_{i}'] = np.clip(trend + np.random.normal(0, 0.02, n_demo), 0, 1)
                demo_data['RUL'] = np.maximum(0, n_demo - cycles).astype(float)
                csv_bytes = pd.DataFrame(demo_data).to_csv(index=False).encode()
                st.download_button("⬇ Download demo_machine_data.csv", data=csv_bytes,
                                    file_name="demo_machine_data.csv", mime="text/csv")

        uploaded_file = st.file_uploader("Upload labelled sensor CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"Loaded: {df_upload.shape[0]} cycles × {df_upload.shape[1]} columns")

                if 'RUL' not in df_upload.columns:
                    st.error("CSV must contain a column named 'RUL'."); st.stop()

                sensor_cols_upload = [c for c in df_upload.columns if c != 'RUL']
                rul_values         = df_upload['RUL'].values
                X_upload           = df_upload[sensor_cols_upload].values

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Cycles",         df_upload.shape[0])
                col_b.metric("Sensor Columns", len(sensor_cols_upload))
                col_c.metric("RUL Range",      f"{rul_values.min():.0f}–{rul_values.max():.0f}")

                preview_sensor = st.selectbox("Preview sensor", sensor_cols_upload)
                fig_up, axes_up = plt.subplots(1, 2, figsize=(14, 3))
                axes_up[0].plot(df_upload[preview_sensor].values, color='steelblue', linewidth=1.5)
                axes_up[0].set_title(f'{preview_sensor} over time'); axes_up[0].grid(alpha=0.3)
                axes_up[1].plot(rul_values, color='coral', linewidth=2)
                axes_up[1].set_title('RUL Labels'); axes_up[1].grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig_up); plt.close()

                if len(X_upload) < WINDOW_SIZE + 5:
                    st.error(f"Need at least {WINDOW_SIZE + 5} cycles. Got {len(X_upload)}."); st.stop()

                if st.button(f"▶ Fine-Tune for '{machine_id}'", type="primary"):
                    base_weights = f'models/saved/lstm_target_only_{base_ds}.keras'
                    if not os.path.exists(base_weights):
                        st.error(f"Pretrained weights not found for {base_ds}. Run Step 4 first."); st.stop()

                    try:
                        from src.feature_aligner       import FeatureAligner
                        from src.fine_tuner            import FewShotFineTuner
                        from src.models.lstm_baseline  import build_lstm_baseline
                        from src.models.adaptive_lstm  import AdaptivePipeline
                        from src.windowing             import create_windows
                        import tensorflow as tf
                        from tensorflow.keras.callbacks import EarlyStopping

                        TARGET_DIM = 24

                        with st.spinner("Fitting feature aligner..."):
                            aligner = FeatureAligner(target_dim=TARGET_DIM)
                            aligner.fit(X_upload, feature_names=sensor_cols_upload)
                            summary = aligner.summary()

                        st.info(f"Feature Alignment: {summary['input_dim']} sensors → "
                                f"{summary['target_dim']} dims via **{summary['method']}**")

                        X_aligned        = aligner.transform(X_upload)
                        feature_cols_tmp = [f'f_{i}' for i in range(TARGET_DIM)]
                        df_tmp = pd.DataFrame(X_aligned, columns=feature_cols_tmp)
                        df_tmp['unit_id'] = 1
                        df_tmp['cycle']   = np.arange(1, len(df_tmp) + 1)
                        df_tmp['RUL']     = rul_values
                        X_win, y_win, _   = create_windows(df_tmp, feature_cols_tmp, window_size=WINDOW_SIZE)

                        split  = max(1, int(len(X_win) * 0.8))
                        X_tr   = X_win[:split].astype(np.float32)
                        y_tr   = y_win[:split].astype(np.float32)
                        X_vl   = X_win[split:].astype(np.float32)
                        y_vl   = y_win[split:].astype(np.float32)

                        st.info(f"Training windows: {len(X_tr)} | Validation: {len(X_vl)}")

                        model = build_lstm_baseline(WINDOW_SIZE, TARGET_DIM)
                        model.load_weights(base_weights)
                        tuner = FewShotFineTuner(model, freeze_layers=['lstm'])

                        progress   = st.progress(0)
                        chart_spot = st.empty()
                        p1_losses, p2_losses = [], []

                        class FTCallback(tf.keras.callbacks.Callback):
                            def __init__(self, store, total, offset=0):
                                self.store = store; self.total = total; self.offset = offset
                            def on_epoch_end(self, epoch, logs=None):
                                self.store.append(logs.get('loss', 0))
                                progress.progress(min((self.offset + epoch + 1) / self.total, 1.0))

                        total_ep = int(p1_epochs) + int(p2_epochs)
                        val_data = (X_vl, y_vl) if len(X_vl) > 0 else None

                        tuner._freeze(freeze=True)
                        model.compile(optimizer=tf.keras.optimizers.Adam(float(p1_lr)), loss='mse', metrics=['mae'])
                        model.fit(X_tr, y_tr, validation_data=val_data,
                                   epochs=int(p1_epochs), batch_size=max(4, len(X_tr) // 4),
                                   callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                              FTCallback(p1_losses, total_ep, 0)], verbose=0)

                        p2_final = None
                        if int(p2_epochs) > 0:
                            tuner._freeze(freeze=False)
                            model.compile(optimizer=tf.keras.optimizers.Adam(float(p2_lr)), loss='mse', metrics=['mae'])
                            h2 = model.fit(X_tr, y_tr, validation_data=val_data,
                                            epochs=int(p2_epochs), batch_size=max(4, len(X_tr) // 4),
                                            callbacks=[EarlyStopping(patience=8, restore_best_weights=True),
                                                       FTCallback(p2_losses, total_ep, int(p1_epochs))], verbose=0)
                            p2_final = h2.history['loss'][-1]

                        fig_ft, ax_ft = plt.subplots(figsize=(12, 4))
                        if p1_losses:
                            ax_ft.plot(p1_losses, color='steelblue', label='Phase 1 (head-only)')
                        if p2_losses:
                            offset = len(p1_losses)
                            ax_ft.plot(range(offset, offset + len(p2_losses)), p2_losses,
                                       color='coral', label='Phase 2 (full fine-tune)')
                            ax_ft.axvline(offset, color='gray', linestyle='--')
                        ax_ft.set_xlabel('Epoch'); ax_ft.set_ylabel('MSE Loss')
                        ax_ft.set_title(f'Fine-Tuning Loss — {machine_id}')
                        ax_ft.legend(); ax_ft.grid(alpha=0.3)
                        plt.tight_layout(); chart_spot.pyplot(fig_ft); plt.close()

                        os.makedirs('models/saved', exist_ok=True)
                        adapted_weights = f'models/saved/lstm_adapted_{machine_id}.keras'
                        tuner._freeze(freeze=False)
                        model.save_weights(adapted_weights)

                        pipeline = AdaptivePipeline(
                            aligner=aligner,
                            model_weights_path=os.path.abspath(adapted_weights),
                            window_size=WINDOW_SIZE, max_rul=MAX_RUL
                        )
                        pipe_path = f'models/saved/pm_pipeline_{machine_id}.joblib'
                        pipeline.save(pipe_path)

                        st.success(f"✅ Saved: `{pipe_path}`")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Phase 1 Loss", f"{p1_losses[-1]:.4f}" if p1_losses else "—")
                        col_r2.metric("Phase 2 Loss", f"{p2_final:.4f}" if p2_final else "Skipped")
                        col_r3.metric("Alignment",    summary['method'])

                    except Exception as e:
                        st.error(f"Fine-tuning failed: {e}"); st.exception(e)

            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    with tab2:
        st.subheader("Predict RUL with Your Fine-Tuned Model")
        adapted_machines = []
        if os.path.exists('models/saved'):
            adapted_machines = [
                f.replace('pm_pipeline_', '').replace('.joblib', '')
                for f in os.listdir('models/saved')
                if f.startswith('pm_pipeline_') and
                f.replace('pm_pipeline_', '').replace('.joblib', '').upper() not in DATASET_IDS
            ]

        if not adapted_machines:
            st.info("No adapted machines found. Fine-tune a model in Tab 1 first.")
        else:
            pred_machine = st.selectbox("Select adapted machine", adapted_machines)
            pred_file    = st.file_uploader("Upload new sensor readings CSV", type=['csv'], key='pred_upload')

            if pred_file is not None:
                try:
                    df_pred          = pd.read_csv(pred_file)
                    pred_sensor_cols = [c for c in df_pred.columns if c.lower() != 'rul']
                    X_pred_raw       = df_pred[pred_sensor_cols].values
                    st.info(f"Loaded {len(X_pred_raw)} cycles × {len(pred_sensor_cols)} sensors")

                    if st.button("▶ Predict RUL", type="primary", key='run_adapted_pred'):
                        from src.models.adaptive_lstm import AdaptivePipeline
                        pipeline = AdaptivePipeline.load(f'models/saved/pm_pipeline_{pred_machine}.joblib')
                        result   = pipeline.predict(X_pred_raw)

                        state_icon = {'Healthy': '🟢', 'Warning': '🟡', 'Critical': '🔴'}
                        st.markdown(f"## {state_icon.get(result['health_state'], '⚪')} {result['health_state']}")

                        col_p1, col_p2, col_p3 = st.columns(3)
                        col_p1.metric("Predicted RUL",    f"{result['rul_prediction']} cycles")
                        col_p2.metric("Change Point",     "Detected ⚠" if result['change_point_detected'] else "None ✓")
                        col_p3.metric("Alignment Method", result['alignment_method'])

                        if   result['health_state'] == 'Critical': st.error("⛔ CRITICAL: Schedule maintenance immediately.")
                        elif result['health_state'] == 'Warning':  st.warning("⚠️ WARNING: Plan maintenance soon.")
                        else:                                       st.success("✅ Machine operating normally.")
                        st.json(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}"); st.exception(e)

    with tab3:
        st.subheader("Fine-Tuned Machine Registry")
        if not os.path.exists('models/saved'):
            st.info("No models directory found.")
        else:
            registry = []
            for f in sorted(os.listdir('models/saved')):
                if not f.startswith('pm_pipeline_'): continue
                name    = f.replace('pm_pipeline_', '').replace('.joblib', '')
                is_base = name.upper() in DATASET_IDS
                size_kb = os.path.getsize(f'models/saved/{f}') / 1024
                registry.append({
                    'Machine':   name,
                    'Type':      'Base (C-MAPSS)' if is_base else '🔁 Adapted',
                    'Size (KB)': round(size_kb, 1),
                    'Aligner':   '✅' if (is_base or os.path.exists(f'models/saved/aligner_{name}.joblib')) else '❌',
                    'Weights':   '✅' if (is_base or os.path.exists(f'models/saved/lstm_adapted_{name}.keras')) else '❌',
                })
            if registry:
                st.dataframe(pd.DataFrame(registry), use_container_width=True)
            else:
                st.info("No pipelines saved yet.")

            adapted_list = [r['Machine'] for r in registry if r['Type'] == '🔁 Adapted']
            if adapted_list:
                st.markdown("---")
                del_machine = st.selectbox("Remove an adapted model", adapted_list)
                if st.button(f"🗑 Delete '{del_machine}'", type="secondary"):
                    for fp in [f'models/saved/pm_pipeline_{del_machine}.joblib',
                               f'models/saved/lstm_adapted_{del_machine}.keras',
                               f'models/saved/aligner_{del_machine}.joblib']:
                        if os.path.exists(fp): os.remove(fp)
                    st.success(f"Removed files for '{del_machine}'.")
                    st.rerun()
