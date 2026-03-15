from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChangePointResult:
    indices: np.ndarray
    scores: np.ndarray


class CUSUMDetector:
    def __init__(self, threshold: float = 5.0, drift: float = 0.0) -> None:
        self.threshold = threshold
        self.drift = drift

    def detect(self, signal: np.ndarray) -> ChangePointResult:
        values = np.asarray(signal, dtype=float)
        if values.ndim != 1:
            raise ValueError("CUSUMDetector expects a one-dimensional signal")

        mean = values.mean()
        pos_sum = 0.0
        neg_sum = 0.0
        indices: list[int] = []
        scores: list[float] = []

        for index, value in enumerate(values):
            centered = value - mean - self.drift
            pos_sum = max(0.0, pos_sum + centered)
            neg_sum = min(0.0, neg_sum + centered)
            score = max(pos_sum, abs(neg_sum))
            if score >= self.threshold:
                indices.append(index)
                scores.append(score)
                pos_sum = 0.0
                neg_sum = 0.0

        return ChangePointResult(np.asarray(indices, dtype=int), np.asarray(scores, dtype=float))


class BOCPDDetector:
    """Bayesian online changepoint detector with a Gaussian mean model."""

    def __init__(
        self,
        hazard: float = 200.0,
        mean0: float = 0.0,
        var0: float = 1.0,
        obs_var: float = 1.0,
        changepoint_threshold: float = 0.35,
    ) -> None:
        self.hazard = hazard
        self.mean0 = mean0
        self.var0 = var0
        self.obs_var = obs_var
        self.changepoint_threshold = changepoint_threshold

    def detect(self, signal: np.ndarray) -> ChangePointResult:
        values = np.asarray(signal, dtype=float)
        if values.ndim != 1:
            raise ValueError("BOCPDDetector expects a one-dimensional signal")

        hazard_probability = self._hazard_probability()
        run_probs = np.array([1.0], dtype=float)
        means = np.array([self.mean0], dtype=float)
        variances = np.array([self.var0], dtype=float)

        indices: list[int] = []
        scores: list[float] = []

        for index, value in enumerate(values):
            predictive = self._normal_pdf(value, means, variances + self.obs_var)
            growth_probs = run_probs * predictive * (1.0 - hazard_probability)
            cp_prob = np.sum(run_probs * predictive * hazard_probability)

            new_run_probs = np.empty(len(run_probs) + 1, dtype=float)
            new_run_probs[0] = cp_prob
            new_run_probs[1:] = growth_probs
            total = new_run_probs.sum()
            if total == 0.0:
                new_run_probs[:] = 0.0
                new_run_probs[0] = 1.0
            else:
                new_run_probs /= total

            if new_run_probs[0] >= self.changepoint_threshold:
                indices.append(index)
                scores.append(float(new_run_probs[0]))

            posterior_variances = 1.0 / (1.0 / variances + 1.0 / self.obs_var)
            posterior_means = posterior_variances * ((means / variances) + (value / self.obs_var))

            means = np.concatenate(([self.mean0], posterior_means))
            variances = np.concatenate(([self.var0], posterior_variances))
            run_probs = new_run_probs

        return ChangePointResult(np.asarray(indices, dtype=int), np.asarray(scores, dtype=float))

    def _hazard_probability(self) -> float:
        if self.hazard <= 0:
            raise ValueError("hazard must be positive")
        if self.hazard <= 1.0:
            return float(self.hazard)
        return 1.0 / float(self.hazard)

    def _normal_pdf(self, value: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        safe_variance = np.maximum(variance, 1e-8)
        scale = np.sqrt(2.0 * np.pi * safe_variance)
        exponent = -0.5 * ((value - mean) ** 2) / safe_variance
        return np.exp(exponent) / scale
