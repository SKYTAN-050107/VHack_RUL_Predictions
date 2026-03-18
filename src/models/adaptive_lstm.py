import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.signal import savgol_filter


class AdaptivePipeline(BaseEstimator, RegressorMixin):
    """
    Transfer-learning-ready pipeline that accepts ANY number of sensor columns.

    Wraps:
        FeatureAligner  → maps any sensor count to fixed 24-dim space
        LSTM model      → pretrained on C-MAPSS, fine-tuned on new machine
        CUSUM detector  → health state classification

    Unlike the base PredictiveMaintenancePipeline (which requires exactly 24
    features), this pipeline uses the FeatureAligner so X_raw can have any
    number of sensor columns.

    Args:
        aligner             : Fitted FeatureAligner instance
        model_weights_path  : Path to LSTM .keras weights file
        window_size         : Must match training window size (default 30)
        max_rul             : RUL clip ceiling (default 125)
        cusum_threshold     : CUSUM detection threshold (default 5.0)
        sg_window           : Savitzky-Golay filter window length (default 11)
        sg_poly             : Savitzky-Golay polynomial order (default 3)
    """

    def __init__(self,
                  aligner,
                  model_weights_path: str,
                  window_size:        int   = 30,
                  max_rul:            int   = 125,
                  cusum_threshold:    float = 5.0,
                  sg_window:          int   = 11,
                  sg_poly:            int   = 3):
        self.aligner            = aligner
        self.model_weights_path = model_weights_path
        self.window_size        = window_size
        self.max_rul            = max_rul
        self.cusum_threshold    = cusum_threshold
        self.sg_window          = sg_window
        self.sg_poly            = sg_poly
        self._model             = None   # lazy-loaded, not serialised

    def _load_model(self):
        """Load LSTM weights on first predict() call."""
        if self._model is None:
            from src.models.lstm_baseline import build_lstm_baseline
            self._model = build_lstm_baseline(
                window_size=self.window_size,
                n_features=self.aligner.target_dim
            )
            self._model.load_weights(self.model_weights_path)

    def predict(self, X_raw: np.ndarray) -> dict:
        """
        Full inference with automatic feature alignment.

        Args:
            X_raw : np.ndarray of shape (n_cycles, ANY_n_sensors)
                    Columns must be in the same order as when aligner was fitted.

        Returns:
            dict with keys:
                rul_prediction        : float
                health_state          : str  ('Healthy' | 'Warning' | 'Critical')
                change_point_detected : bool
                change_point_step     : int or None
                n_input_sensors       : int
                alignment_method      : str ('pca' | 'zero_pad' | 'passthrough')
        """
        from src.changepoint import cusum_detector, classify_health_state

        self._load_model()
        X = X_raw.astype(np.float64).copy()

        # 1. Savitzky-Golay smoothing per column
        if len(X) >= self.sg_window:
            for j in range(X.shape[1]):
                X[:, j] = savgol_filter(X[:, j], self.sg_window, self.sg_poly)

        # 2. Feature alignment (normalise + PCA compress or zero-pad)
        X_aligned = self.aligner.transform(X)   # → (n_cycles, target_dim)

        # 3. Build last window
        T = len(X_aligned)
        if T < self.window_size:
            pad       = np.zeros((self.window_size - T, X_aligned.shape[1]))
            X_aligned = np.vstack([pad, X_aligned])
        window = X_aligned[-self.window_size:][np.newaxis].astype(np.float32)

        # 4. Predict RUL
        rul = float(np.clip(
            self._model.predict(window, verbose=0).flatten()[0],
            0, self.max_rul
        ))

        # 5. CUSUM change-point detection on most recent 50 cycles
        n_recent = min(50, len(X_aligned))
        cp = cusum_detector(
            X_aligned[-n_recent:, 0],   # use first aligned feature as proxy
            threshold=self.cusum_threshold
        )
        health = classify_health_state(rul, cp is not None)

        aligner_info = self.aligner.summary()
        return {
            'rul_prediction':        round(rul, 1),
            'health_state':          health,
            'change_point_detected': cp is not None,
            'change_point_step':     int(cp) if cp is not None else None,
            'n_input_sensors':       aligner_info['input_dim'],
            'alignment_method':      aligner_info['method']
        }

    # ── Joblib serialisation helpers ──────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None   # do not serialise Keras model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None       # will be lazy-loaded on next predict()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"AdaptivePipeline saved: {path}")

    @staticmethod
    def load(path: str) -> 'AdaptivePipeline':
        return joblib.load(path)
