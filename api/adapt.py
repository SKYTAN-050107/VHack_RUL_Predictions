import numpy as np
import os
import joblib


def _normalize_dataset_id(dataset_id: str) -> str:
    return str(dataset_id).strip().upper()


def _normalize_machine_id(machine_id: str) -> str:
    machine_id = str(machine_id).strip()
    if not machine_id:
        raise ValueError("machine_id must be a non-empty string.")
    return machine_id


def run_adaptation(
    machine_id:      str,
    base_dataset_id: str,
    sensor_names:    list,
    readings:        list,
    rul_labels:      list,
    phase1_epochs:   int   = 30,
    phase2_epochs:   int   = 20,
    phase1_lr:       float = 1e-3,
    phase2_lr:       float = 1e-4,
    models_dir:      str   = 'models/saved',
    window_size:     int   = 30
) -> dict:
    """
    Full transfer learning pipeline triggered by POST /adapt:
      1. Validate inputs
      2. Load pretrained LSTM from base_dataset_id
      3. Fit FeatureAligner on the new sensor layout
      4. Build sliding windows from new machine data
      5. Fine-tune with FewShotFineTuner (phase 1 + phase 2)
      6. Save AdaptivePipeline as pm_pipeline_{machine_id}.joblib
      7. Return training summary dict

    Args:
        machine_id      : Your identifier for the new machine
        base_dataset_id : Which pretrained C-MAPSS model to start from
        sensor_names    : Column names for the new machine's sensors
                          (must match order of columns in readings)
        readings        : Raw sensor data, shape (n_cycles, n_sensors)
        rul_labels      : RUL value per cycle, shape (n_cycles,)
        phase1_epochs   : Head-only training epochs
        phase2_epochs   : Full fine-tune epochs (0 to skip)
        phase1_lr       : Learning rate Phase 1
        phase2_lr       : Learning rate Phase 2
        models_dir      : Directory to save adapted pipeline
        window_size     : Must match LSTM training window size (30)

    Returns:
        dict matching AdaptResponse schema
    """
    from src.feature_aligner       import FeatureAligner
    from src.fine_tuner            import FewShotFineTuner
    from src.models.lstm_baseline  import build_lstm_baseline
    from src.models.adaptive_lstm  import AdaptivePipeline
    from src.windowing             import create_windows
    import pandas as pd

    TARGET_DIM = 24  # fixed LSTM input size matching C-MAPSS training
    machine_id = _normalize_machine_id(machine_id)
    base_dataset_id = _normalize_dataset_id(base_dataset_id)

    # ── 1. Validate ───────────────────────────────────────────────────────────
    X_raw = np.array(readings,   dtype=np.float32)
    y_raw = np.array(rul_labels, dtype=np.float32)

    if len(X_raw) != len(y_raw):
        raise ValueError(
            f"readings has {len(X_raw)} rows but rul_labels has {len(y_raw)}. "
            "They must be the same length."
        )
    if X_raw.shape[1] != len(sensor_names):
        raise ValueError(
            f"sensor_names has {len(sensor_names)} names but readings has "
            f"{X_raw.shape[1]} columns."
        )
    if len(X_raw) < window_size + 5:
        raise ValueError(
            f"Need at least {window_size + 5} cycles. Got {len(X_raw)}."
        )

    # ── 2. Load pretrained LSTM ───────────────────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)
    base_weights = os.path.join(
        models_dir, f'lstm_target_only_{base_dataset_id}.keras'
    )
    if not os.path.exists(base_weights):
        raise FileNotFoundError(
            f"Pretrained weights not found: {base_weights}. "
            "Run Notebook 04 / Streamlit Step 4 first."
        )
    model = build_lstm_baseline(window_size=window_size, n_features=TARGET_DIM)
    model.load_weights(base_weights)

    # ── 3. Fit FeatureAligner ─────────────────────────────────────────────────
    aligner = FeatureAligner(target_dim=TARGET_DIM)
    aligner.fit(X_raw, feature_names=sensor_names)
    aligner_summary = aligner.summary()

    X_aligned    = aligner.transform(X_raw)
    aligner_path = os.path.join(models_dir, f'aligner_{machine_id}.joblib')
    aligner.save(aligner_path)

    # ── 4. Build sliding windows ──────────────────────────────────────────────
    feature_cols = [f'f_{i}' for i in range(TARGET_DIM)]
    df_tmp = pd.DataFrame(X_aligned, columns=feature_cols)
    df_tmp['unit_id'] = 1
    df_tmp['cycle']   = np.arange(1, len(df_tmp) + 1)
    df_tmp['RUL']     = y_raw

    X_win, y_win, _ = create_windows(
        df_tmp, feature_cols, window_size=window_size, rul_col='RUL'
    )
    if len(X_win) < 5:
        raise ValueError(
            f"Not enough windows ({len(X_win)}). Provide more cycles of data."
        )

    split   = max(1, int(len(X_win) * 0.8))
    X_tr    = X_win[:split].astype(np.float32)
    y_tr    = y_win[:split].astype(np.float32)
    X_val   = X_win[split:].astype(np.float32)
    y_val   = y_win[split:].astype(np.float32)

    # ── 5. Fine-tune ──────────────────────────────────────────────────────────
    tuner   = FewShotFineTuner(model, freeze_layers=['lstm'])
    history = tuner.fine_tune(
        X_new=X_tr, y_new=y_tr,
        X_val=X_val if len(X_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        batch_size=min(16, max(4, len(X_tr) // 4))
    )

    # ── 6. Save adapted weights + pipeline ───────────────────────────────────
    adapted_weights = os.path.join(models_dir, f'lstm_adapted_{machine_id}.keras')
    tuner.save_adapted_model(adapted_weights)

    pipeline = AdaptivePipeline(
        aligner=aligner,
        model_weights_path=os.path.abspath(adapted_weights),
        window_size=window_size,
        max_rul=125,
        cusum_threshold=5.0
    )
    pipeline_path = os.path.join(models_dir, f'pm_pipeline_{machine_id}.joblib')
    pipeline.save(pipeline_path)

    p1_loss = float(history['phase1_loss'][-1]) if history['phase1_loss'] else 0.0
    p2_loss = float(history['phase2_loss'][-1]) if history['phase2_loss'] else None

    return {
        'machine_id':        machine_id,
        'base_dataset_id':   base_dataset_id,
        'n_training_cycles': len(X_raw),
        'n_sensors_input':   aligner_summary['input_dim'],
        'alignment_method':  aligner_summary['method'],
        'phase1_final_loss': round(p1_loss, 4),
        'phase2_final_loss': round(p2_loss, 4) if p2_loss else None,
        'pipeline_path':     pipeline_path,
        'message': (
            f"Fine-tuning complete. Adapted pipeline saved for '{machine_id}'. "
            f"Use POST /predict/adapted with machine_id='{machine_id}'."
        )
    }


def run_adapted_prediction(
    machine_id: str,
    readings:   list,
    models_dir: str = 'models/saved'
) -> dict:
    """
    Run RUL prediction using a fine-tuned machine-specific pipeline.

    Args:
        machine_id : Must match the machine_id used in /adapt
        readings   : Raw sensor data — same column order as used during /adapt
        models_dir : Directory containing saved pipelines

    Returns:
        Standard prediction dict + machine_id field
    """
    from src.models.adaptive_lstm import AdaptivePipeline

    machine_id = _normalize_machine_id(machine_id)

    pipeline_path = os.path.join(models_dir, f'pm_pipeline_{machine_id}.joblib')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"No adapted pipeline for '{machine_id}'. Run POST /adapt first."
        )

    pipeline = AdaptivePipeline.load(pipeline_path)
    X_raw    = np.array(readings, dtype=np.float32)
    result   = pipeline.predict(X_raw)
    result['machine_id'] = machine_id
    return result
