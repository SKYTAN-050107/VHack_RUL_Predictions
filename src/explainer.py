from __future__ import annotations

import numpy as np
import shap
import torch
from torch import nn


class ShapSequenceExplainer:
    """Small SHAP wrapper for sequence models that output a scalar RUL prediction."""

    def __init__(self, model: nn.Module, background: np.ndarray, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.background = np.asarray(background, dtype=np.float32)
        if self.background.ndim != 3:
            raise ValueError("background must have shape [n_samples, sequence_length, n_features]")
        self.sequence_length = self.background.shape[1]
        self.feature_count = self.background.shape[2]
        flattened_background = self.background.reshape(self.background.shape[0], -1)
        self.explainer = shap.KernelExplainer(self._predict_from_flattened, flattened_background)

    def explain(self, samples: np.ndarray, nsamples: int = 100) -> np.ndarray:
        sequence_batch = np.asarray(samples, dtype=np.float32)
        if sequence_batch.ndim != 3:
            raise ValueError("samples must have shape [n_samples, sequence_length, n_features]")
        flat_samples = sequence_batch.reshape(sequence_batch.shape[0], -1)
        shap_values = self.explainer.shap_values(flat_samples, nsamples=nsamples)
        return np.asarray(shap_values).reshape(sequence_batch.shape)

    def _predict_from_flattened(self, flat_samples: np.ndarray) -> np.ndarray:
        sequence_batch = np.asarray(flat_samples, dtype=np.float32).reshape(-1, self.sequence_length, self.feature_count)
        with torch.no_grad():
            inputs = torch.from_numpy(sequence_batch).to(self.device)
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        return outputs.detach().cpu().numpy().reshape(-1)
