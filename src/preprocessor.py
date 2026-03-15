from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data_loader import OPERATING_COLUMNS, SENSOR_COLUMNS


@dataclass(slots=True)
class PreprocessorConfig:
    sensor_columns: Sequence[str] = field(default_factory=lambda: SENSOR_COLUMNS.copy())
    operating_columns: Sequence[str] = field(default_factory=lambda: OPERATING_COLUMNS.copy())
    group_column: str = "unit_id"
    smoothing_window: int = 5
    smoothing_polyorder: int = 2
    outlier_clip_quantiles: tuple[float, float] | None = (0.01, 0.99)
    normalize: bool = True


class CMAPSSPreprocessor:
    """Apply imputation, optional smoothing, clipping, and normalization."""

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.clip_bounds_: dict[str, tuple[float, float]] | None = None
        self.feature_columns_ = [*self.config.operating_columns, *self.config.sensor_columns]

    def fit(self, frame: pd.DataFrame) -> "CMAPSSPreprocessor":
        prepared = self._smooth(frame)
        if self.config.outlier_clip_quantiles is not None:
            lower_q, upper_q = self.config.outlier_clip_quantiles
            self.clip_bounds_ = {
                column: (
                    prepared[column].quantile(lower_q),
                    prepared[column].quantile(upper_q),
                )
                for column in self.config.sensor_columns
            }
            prepared = self._clip(prepared)
        feature_frame = prepared[self.feature_columns_]
        imputed = self.imputer.fit_transform(feature_frame)
        if self.config.normalize:
            self.scaler.fit(imputed)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._smooth(frame)
        if self.clip_bounds_ is not None:
            prepared = self._clip(prepared)
        feature_frame = prepared[self.feature_columns_]
        imputed = self.imputer.transform(feature_frame)
        if self.config.normalize:
            prepared.loc[:, self.feature_columns_] = self.scaler.transform(imputed)
        else:
            prepared.loc[:, self.feature_columns_] = imputed
        return prepared

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def _smooth(self, frame: pd.DataFrame) -> pd.DataFrame:
        smoothed = frame.copy()
        window = self._validated_window(self.config.smoothing_window)
        if window is None:
            return smoothed

        for column in self.config.sensor_columns:
            smoothed[column] = (
                smoothed.groupby(self.config.group_column, group_keys=False)[column]
                .transform(lambda values: self._smooth_series(values, window=window))
            )
        return smoothed

    def _clip(self, frame: pd.DataFrame) -> pd.DataFrame:
        clipped = frame.copy()
        for column, (lower, upper) in (self.clip_bounds_ or {}).items():
            clipped[column] = clipped[column].clip(lower=lower, upper=upper)
        return clipped

    def _validated_window(self, window: int) -> int | None:
        if window <= 2:
            return None
        if window % 2 == 0:
            return window + 1
        return window

    def _smooth_series(self, values: pd.Series, window: int) -> pd.Series:
        if len(values) < window or window <= self.config.smoothing_polyorder:
            return values.rolling(window=min(len(values), 3), min_periods=1).mean()
        filtered = savgol_filter(values.to_numpy(), window_length=window, polyorder=self.config.smoothing_polyorder)
        return pd.Series(filtered, index=values.index)
