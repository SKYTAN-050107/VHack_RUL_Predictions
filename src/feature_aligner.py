import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class FeatureAligner:
    """
    Maps an arbitrary number of input sensors to a fixed-size embedding
    so the LSTM always receives the same input shape regardless of machine type.

    Strategy:
      - If new machine has MORE sensors than target_dim:
          Use PCA to compress down to target_dim
      - If new machine has FEWER sensors than target_dim:
          Zero-pad the missing columns on the right
      - If new machine has EXACT match:
          Pass through with only normalisation

    Args:
        target_dim   : Fixed output dimension expected by the LSTM (default 24)
        scaler_type  : 'minmax' or 'standard'
    """

    def __init__(self, target_dim: int = 24, scaler_type: str = 'minmax'):
        self.target_dim    = target_dim
        self.scaler_type   = scaler_type
        self.scaler        = None
        self.pca           = None
        self.input_dim     = None
        self.feature_names = None
        self.is_fitted     = False

    def fit(self, X: np.ndarray, feature_names: list = None):
        """
        Fit the aligner on new machine training data.

        Args:
            X             : Raw sensor array of shape (n_cycles, n_sensors)
            feature_names : Optional list of sensor names for traceability
        """
        self.input_dim     = X.shape[1]
        self.feature_names = feature_names or [f'sensor_{i}' for i in range(self.input_dim)]

        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        self.scaler.fit(X)

        if self.input_dim > self.target_dim:
            from sklearn.decomposition import PCA
            X_scaled  = self.scaler.transform(X)
            self.pca  = PCA(n_components=self.target_dim, random_state=42)
            self.pca.fit(X_scaled)
            explained = self.pca.explained_variance_ratio_.sum()
            print(f"FeatureAligner: PCA {self.input_dim}→{self.target_dim} features, "
                  f"variance retained: {explained:.1%}")
        elif self.input_dim < self.target_dim:
            print(f"FeatureAligner: zero-padding {self.input_dim}→{self.target_dim} features")
        else:
            print(f"FeatureAligner: exact match at {self.target_dim} features")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Align raw sensor array to target_dim.

        Args:
            X : Raw sensor array of shape (n_cycles, n_sensors)

        Returns:
            Aligned array of shape (n_cycles, target_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        X_scaled = self.scaler.transform(X)

        if self.input_dim > self.target_dim:
            return self.pca.transform(X_scaled)
        elif self.input_dim < self.target_dim:
            pad = np.zeros((len(X_scaled), self.target_dim - self.input_dim))
            return np.hstack([X_scaled, pad])
        else:
            return X_scaled

    def fit_transform(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'FeatureAligner':
        return joblib.load(path)

    def summary(self) -> dict:
        return {
            'input_dim':     self.input_dim,
            'target_dim':    self.target_dim,
            'method':        ('pca' if self.pca else
                              'zero_pad' if self.input_dim < self.target_dim else
                              'passthrough'),
            'is_fitted':     self.is_fitted,
            'feature_names': self.feature_names
        }
